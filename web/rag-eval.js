// Evaluation harness: run the pipeline over the labeled sample, score it against the gold labels, cache the run.
import { $, esc, pct, num, settings, M, onModelsReady, initModelCard, loadRecorded, pillIdle, download, yieldToBrowser, bars } from "./common.js";
import { loadCorpus, indexCorpus, retrieve, judge, runKey, evidenceRetrieved, expandRecorded, compactRun } from "./pipeline.js";
import { PRESETS, DEFAULT_PRESET, DEFAULT_THRESHOLD, gate, verdictOf } from "./judge.js";
import { evaluate } from "./metrics.js";
import * as db from "./store.js";

const S = {
  corpus: null, runs: new Map(), current: null, running: false, stop: false, open: new Set(),
  cutoff: settings.get("cutoff", 0.5), threshold: settings.get("threshold", DEFAULT_THRESHOLD),
};

const rag = () => ({ sentencesPerChunk: +$("spc").value, overlap: 0, k: +$("k").value, scope: $("scope").value });

function subsetClaims(which) {
  const cl = S.corpus.claims;
  if (which === "all") return cl;
  if (which === "1") {
    const out = [];
    for (const docId of Object.keys(S.corpus.documents)) {
      const mine = cl.filter((c) => c.docId === docId);
      out.push(...["supported", "not_supported"].map((l) => mine.find((c) => c.label === l)).filter(Boolean));
    }
    return out;
  }
  // quick: 2 documents per dataset, 1 supported + 1 unsupported each -> 12 claims
  const out = [];
  for (const ds of Object.keys(S.corpus.datasets)) {
    for (const docId of Object.keys(S.corpus.documents).filter((d) => S.corpus.documents[d].dataset === ds).slice(0, 2)) {
      const mine = cl.filter((c) => c.docId === docId);
      out.push(mine.find((c) => c.label === "supported"), mine.find((c) => c.label === "not_supported" && c.kind !== "unverifiable") || mine.find((c) => c.label === "not_supported"));
    }
  }
  return out.filter(Boolean);
}

// ---- runs -------------------------------------------------------------------------------------------

const runLabel = (r) => `${r.source === "recorded" ? "Recorded · " : ""}${r.model.variant} · ${PRESETS[r.presetId]?.name.split(":")[0] || r.presetId} · ${r.rag.sentencesPerChunk} sent/chunk · k=${r.rag.k} · ${r.rag.scope} · ${r.records.length} claims${r.complete ? "" : " (partial)"}`;

// Browser runs and recorded runs can share a configuration key; list them separately so neither hides the other.
const runId = (r) => (r.source === "recorded" ? "rec|" : "") + r.key;

function fillRunSelect() {
  const sel = $("runSel"); const cur = S.current && runId(S.current);
  const list = [...S.runs.values()].sort((a, b) => (a.source === "recorded") - (b.source === "recorded") || String(b.createdAt).localeCompare(String(a.createdAt)));
  sel.innerHTML = list.length ? list.map((r) => `<option value="${esc(runId(r))}">${esc(runLabel(r))}</option>`).join("") : "<option value=''>No runs yet</option>";
  if (cur && S.runs.has(cur)) sel.value = cur;
  $("exportBtn").disabled = !S.current;
}

function select(key) {
  const r = S.runs.get(key); if (!r) return;
  S.current = r; S.open.clear();
  // reflect the run's settings in the controls, so "Run evaluation" resumes / reproduces it
  $("preset").value = r.presetId; $("spc").value = String(r.rag.sentencesPerChunk); $("k").value = String(r.rag.k); $("scope").value = r.rag.scope;
  fillRunSelect(); render();
}

/** A subset run never throws away claims a stored run with the same key already judged; merge them in. */
function withPrev(run, prev) {
  if (!prev || prev.source === "recorded") return run;
  const ids = new Set(run.records.map((x) => x.id));
  return { ...run, records: [...run.records, ...prev.records.filter((x) => !ids.has(x.id))], complete: run.complete || prev.complete };
}

async function runEval() {
  if (S.running) return;
  if (!M.laya) { $("runHint").textContent = "Load the models first (button above). Recorded runs are listed under Stored runs."; return; }
  const r = rag(), presetId = $("preset").value, subset = $("subset").value;
  const key = runKey({ modelVariant: M.model.variant, presetId, rag: r });
  const claims = subsetClaims(subset);
  // Resume: reuse any claim already judged under exactly this configuration (browser-made runs only).
  const prev = S.runs.get(key);
  const have = new Map(prev && prev.source !== "recorded" ? prev.records.map((x) => [x.id, x]) : []);
  const run = {
    key, source: "browser", createdAt: new Date().toISOString(), presetId, rag: r, subset,
    model: { variant: M.model.variant, backend: M.model.backend, threads: M.model.threads, embedder: M.embedder.id },
    records: [], complete: false,
  };
  S.runs.set(key, run); S.current = run; S.open.clear(); fillRunSelect();
  S.running = true; S.stop = false; $("runBtn").disabled = true; $("stopBtn").disabled = false;
  const progress = (i, n, msg) => { $("evalBar").style.width = `${(100 * i / n).toFixed(1)}%`; $("evalCount").textContent = `${i} of ${n}`; if (msg) $("runHint").textContent = msg; };
  try {
    progress(0, claims.length, "Indexing the corpus (chunk + embed, stored in IndexedDB)…");
    const ix = await indexCorpus(S.corpus, M.embedder, r, (i, n) => { $("runHint").textContent = `Indexing the corpus: ${i} / ${n} documents…`; });
    $("runHint").textContent = `Index ready: ${ix.chunks} passages (${ix.embedded ? `${ix.embedded} newly embedded` : "all from IndexedDB"}). Judging…`;
    for (let i = 0; i < claims.length; i++) {
      if (S.stop) break;
      const c = claims[i];
      let rec = have.get(c.id);
      if (!rec) {
        const ret = await retrieve(M.embedder, c.claim, { rag: r, docId: c.docId });
        const j = await judge(M.laya, c.claim, ret.passages, presetId);
        rec = {
          id: c.id, dataset: S.corpus.documents[c.docId].dataset, docId: c.docId, kind: c.kind, gold: c.label, claim: c.claim,
          passages: ret.passages.map((p) => ({ docId: p.docId, chunk: p.chunk, text: p.text, score: +p.score.toFixed(4) })),
          pSupported: j.pSupported, confidence: j.confidence, probabilities: j.probabilities, layaMs: j.layaMs, embedMs: ret.embedMs,
          evidenceHit: evidenceRetrieved(ret.passages, c.evidence),
        };
      }
      run.records.push(rec);
      progress(i + 1, claims.length, `Judged ${i + 1} of ${claims.length}: “${c.claim.slice(0, 60)}${c.claim.length > 60 ? "…" : ""}”`);
      if ((i + 1) % 4 === 0 || i === claims.length - 1) render();
      if ((i + 1) % 8 === 0) db.putMany("runs", [withPrev(run, prev)]).catch(() => {}); // checkpoint, so a closed tab can resume
      await yieldToBrowser(); // keep the tab responsive between model calls
    }
    run.complete = !S.stop && run.records.length === claims.length;
    Object.assign(run, withPrev(run, prev));
    await db.putMany("runs", [run]);
    $("runHint").textContent = S.stop ? `Stopped after ${run.records.length} claims (saved; press Run to resume).` : `Done. ${run.records.length} claims judged and saved in this browser.`;
  } catch (e) {
    console.error(e); $("runHint").textContent = "Error: " + (e?.message || e);
  } finally {
    S.running = false; $("runBtn").disabled = false; $("stopBtn").disabled = true;
    fillRunSelect(); render();
  }
}

// ---- rendering -------------------------------------------------------------------------------------

function kpi(label, value, sub = "") { return `<div class="kpi"><div class="l">${label}</div><div class="v">${value}${sub ? ` <small>${sub}</small>` : ""}</div></div>`; }

function render() {
  const run = S.current;
  if (!run || !run.records.length) {
    $("runTitle").textContent = "No run yet"; $("runMeta").textContent = "Load the models and press Run evaluation, or pick a recorded run.";
    for (const id of ["kpis", "strip", "gates", "coverage", "byDataset", "byKind", "claims"]) $(id).innerHTML = "";
    return;
  }
  const rs = run.records;
  const m = evaluate(rs, { cutoff: S.cutoff, threshold: S.threshold });
  $("runTitle").textContent = `${PRESETS[run.presetId]?.name || run.presetId}`;
  $("runMeta").textContent = `${run.source === "recorded" ? "Recorded" : "Run in this browser"} · Laya ${run.model.variant}${run.model.backend ? ` on ${run.model.backend}` : ""} · ${run.rag.sentencesPerChunk} sentence${run.rag.sentencesPerChunk > 1 ? "s" : ""}/chunk, top-${run.rag.k}, ${run.rag.scope === "corpus" ? "whole corpus" : "claim's document"} · ${rs.length} claims${run.complete ? "" : " (partial)"}`;
  $("kpis").innerHTML = [
    kpi("Accuracy", pct(m.accuracy, 1), `${m.confusion.tp + m.confusion.tn}/${m.n}`),
    kpi("Balanced accuracy", pct(m.balancedAccuracy, 1)),
    kpi("False-verification rate", pct(m.falseVerificationRate, 1), "unsupported → supported"),
    kpi("AUROC", num(m.auroc), "threshold-free"),
    kpi("Evidence recall@k", pct(m.evidenceRecall, 0), "gold span retrieved"),
    kpi("Automated", pct(m.automatedShare, 0), `accuracy ${pct(m.automatedAccuracy, 0)}`),
    kpi("Judge latency", m.latency.median == null ? "–" : `${m.latency.median.toFixed(0)}`, `ms median · p95 ${m.latency.p95?.toFixed(0)}`),
  ].join("");
  $("bestHint").textContent = m.bestCutoff.balancedAccuracy >= 0 ? `On this run, a cutoff of ${m.bestCutoff.cutoff.toFixed(2)} gives the best balanced accuracy (${pct(m.bestCutoff.balancedAccuracy, 1)}). It is fit on the same claims, so it is optimistic.` : "";

  $("strip").innerHTML = strip(rs);
  const G = m.gates, n = rs.length;
  $("gates").innerHTML = bars({ AUTO: G.AUTO / n, BLOCK: G.BLOCK / n, HOLD: G.HOLD / n }, null).replace(/<span class="v">([\d.]+)<\/span>/g, (_, v) => `<span class="v">${Math.round(+v * n)}</span>`)
    + `<p class="caption">AUTO = confident "supported", BLOCK = confident "not supported", HOLD = below the confidence bar (${S.threshold.toFixed(2)}), sent to a person. Accuracy of the automated verdicts: <b>${pct(m.automatedAccuracy, 1)}</b>.</p>`;

  $("coverage").innerHTML = `<table class="grid"><thead><tr><th>Confidence ≥</th><th class="r">Claims kept</th><th class="r">Coverage</th><th class="r">Accuracy</th><th class="r">False-verif.</th></tr></thead><tbody>${
    m.coverage.map((c) => `<tr${Math.abs(c.threshold - S.threshold) < 1e-9 ? ' class="cur"' : ""}><td>${c.threshold.toFixed(2)}</td><td class="r">${c.kept}</td><td class="r">${pct(c.coverage)}</td><td class="r">${pct(c.accuracy, 1)}</td><td class="r">${pct(c.falseVerificationRate, 1)}</td></tr>`).join("")}</tbody></table>`;
  const groupTable = (rows, names) => `<table class="grid"><thead><tr><th></th><th class="r">n</th><th class="r">Accuracy</th><th class="r">Balanced</th><th class="r">False-verif.</th></tr></thead><tbody>${
    rows.map((g) => `<tr><td>${esc(names?.[g.name] ? g.name : g.name)}</td><td class="r">${g.n}</td><td class="r">${pct(g.accuracy, 1)}</td><td class="r">${pct(g.balancedAccuracy, 1)}</td><td class="r">${pct(g.falseVerificationRate, 1)}</td></tr>`).join("")}</tbody></table>`;
  $("byDataset").innerHTML = groupTable(m.byDataset, S.corpus.datasets);
  const order = Object.keys(S.corpus.kinds);
  $("byKind").innerHTML = groupTable([...m.byKind].sort((a, b) => order.indexOf(a.name) - order.indexOf(b.name)), S.corpus.kinds)
    + `<p class="caption">${order.map((k) => `<b>${k}</b>: ${esc(S.corpus.kinds[k].split(": ")[1])}`).join(" · ")}</p>`;
  renderClaims();
}

function strip(rs) {
  const W = 520, H = 116, L = 110, R = 12, T = 10, rowH = 34;
  const x = (p) => L + p * (W - L - R);
  const rows = [["supported", "Gold: supported", "var(--auto)"], ["not_supported", "Gold: not supported", "var(--block)"]];
  let s = `<svg viewBox="0 0 ${W} ${H}" class="strip" role="img" aria-label="p(supported) for each claim, by gold label">`;
  rows.forEach(([lab, name, col], ri) => {
    const cy = T + ri * rowH + rowH / 2;
    s += `<text x="0" y="${cy + 4}">${name}</text><line x1="${L}" x2="${W - R}" y1="${cy}" y2="${cy}" stroke="var(--line)"/>`;
    rs.filter((r) => r.gold === lab).forEach((r, i) => {
      const jitter = ((i * 7919) % 13 - 6) * 1.6;
      const wrong = verdictOf(r.pSupported, S.cutoff) !== r.gold;
      s += `<circle cx="${x(r.pSupported).toFixed(1)}" cy="${(cy + jitter).toFixed(1)}" r="4" fill="${col}" fill-opacity="${wrong ? 0.35 : 0.85}" stroke="${wrong ? col : "none"}"><title>${esc(r.claim)} — p ${r.pSupported.toFixed(3)}</title></circle>`;
    });
  });
  const ay = T + 2 * rowH + 8;
  s += `<line x1="${x(S.cutoff)}" x2="${x(S.cutoff)}" y1="${T - 4}" y2="${ay - 4}" stroke="var(--ink)" stroke-dasharray="3 3"/>`;
  for (const t of [0, 0.25, 0.5, 0.75, 1]) s += `<text x="${x(t)}" y="${ay + 10}" text-anchor="middle">${t}</text>`;
  s += `<text x="${x(S.cutoff) + 4}" y="${T + 2}" style="fill:var(--ink)">cutoff ${S.cutoff.toFixed(2)}</text></svg>`;
  return s;
}

function renderClaims() {
  const run = S.current; if (!run) return;
  const onlyWrong = $("onlyWrong").checked, ds = $("fDataset").value;
  const rows = run.records.filter((r) => (!ds || r.dataset === ds) && (!onlyWrong || verdictOf(r.pSupported, S.cutoff) !== r.gold));
  let html = `<thead><tr><th>Claim</th><th>Gold</th><th>Type</th><th>p(supported)</th><th>Verdict</th><th>Gate</th><th class="r">ms</th></tr></thead><tbody>`;
  for (const r of rows) {
    const v = verdictOf(r.pSupported, S.cutoff), ok = v === r.gold, g = gate(r.pSupported, r.confidence, { cutoff: S.cutoff, threshold: S.threshold });
    html += `<tr class="claimrow" data-id="${r.id}" tabindex="0" aria-expanded="${S.open.has(r.id)}"><td class="claim">${esc(r.claim)}</td><td>${r.gold === "supported" ? "supported" : "not supp."}</td><td><span class="tag">${esc(r.kind)}</span></td>
      <td><span class="pbar"><i style="width:${(r.pSupported * 100).toFixed(0)}%"></i></span>${r.pSupported.toFixed(3)}</td>
      <td><span class="tag ${ok ? "ok" : "bad"}">${v === "supported" ? "supported" : "not supp."}</span></td><td><span class="tag ${g === "AUTO" ? "ok" : g === "BLOCK" ? "bad" : "warn"}">${g}</span></td><td class="r">${r.layaMs != null ? r.layaMs.toFixed(0) : "–"}</td></tr>`;
    if (S.open.has(r.id)) {
      const ev = S.corpus.claims.find((c) => c.id === r.id)?.evidence;
      html += `<tr class="detailrow"><td colspan="7"><div style="font-size:12.5px;color:var(--ink2);margin-bottom:4px">${esc(S.corpus.documents[r.docId].title)} · gold evidence ${r.evidenceHit == null ? "n/a (claim is unverifiable)" : r.evidenceHit ? "retrieved ✓" : "NOT retrieved"} · confidence ${r.confidence.toFixed(3)}</div>
        <ol class="passages">${r.passages.map((p, i) => `<li class="passage${ev && p.text.includes(ev) ? " gold" : ""}"><span class="rank">${i + 1}</span><div><div class="src">${esc(S.corpus.documents[p.docId]?.title || p.docId)} · passage ${p.chunk + 1} · cos ${p.score.toFixed(3)}</div><div class="txt">${ev && p.text.includes(ev) ? esc(p.text).replace(esc(ev), `<mark class="ev">${esc(ev)}</mark>`) : esc(p.text)}</div></div></li>`).join("")}</ol>
        ${bars(r.probabilities, Object.entries(r.probabilities).sort((a, b) => b[1] - a[1])[0][0])}</td></tr>`;
    }
  }
  $("claims").innerHTML = html + (rows.length ? "" : `<tr><td colspan="7" class="placeholder">No claims match.</td></tr>`) + "</tbody>";
}

// ---- wiring ------------------------------------------------------------------------------------------

function wire() {
  $("preset").innerHTML = Object.entries(PRESETS).map(([k, p]) => `<option value="${k}">${esc(p.name)}</option>`).join("");
  $("preset").value = settings.get("preset", DEFAULT_PRESET);
  $("spc").value = String(settings.get("spc", 2)); $("k").value = String(settings.get("k", 3)); $("scope").value = settings.get("scope", "document");
  for (const id of ["preset", "spc", "k", "scope"]) $(id).addEventListener("change", () => settings.set(id, ["spc", "k"].includes(id) ? +$(id).value : $(id).value));
  $("runBtn").addEventListener("click", runEval);
  $("stopBtn").addEventListener("click", () => { S.stop = true; $("runHint").textContent = "Stopping after the current claim…"; });
  $("runSel").addEventListener("change", (e) => select(e.target.value));
  $("exportBtn").addEventListener("click", () => { const r = S.current; if (r) download(`laya-rag-judge-${r.presetId}-s${r.rag.sentencesPerChunk}k${r.rag.k}-${r.rag.scope}.json`, JSON.stringify(r, null, 2)); });
  $("clearBtn").addEventListener("click", async () => {
    if (!confirm("Delete the stored passage index, claim embeddings and evaluation runs in this browser? Recorded runs stay.")) return;
    await db.clearAll();
    for (const [k, r] of S.runs) if (r.source !== "recorded") S.runs.delete(k);
    S.current = [...S.runs.values()][0] || null; fillRunSelect(); render();
  });
  const slider = (id, key, valId) => {
    $(id).value = S[key]; $(valId).textContent = (+S[key]).toFixed(2);
    $(id).addEventListener("input", () => { S[key] = +$(id).value; $(valId).textContent = S[key].toFixed(2); settings.set(key, S[key]); render(); });
  };
  slider("cutoff", "cutoff", "cutoffVal");
  slider("thresh", "threshold", "threshVal");
  $("onlyWrong").addEventListener("change", renderClaims);
  $("fDataset").addEventListener("change", renderClaims);
  $("claims").addEventListener("click", (e) => { const tr = e.target.closest(".claimrow"); if (!tr) return; const id = tr.dataset.id; S.open.has(id) ? S.open.delete(id) : S.open.add(id); renderClaims(); });
  $("claims").addEventListener("keydown", (e) => { if ((e.key === "Enter" || e.key === " ") && e.target.classList.contains("claimrow")) { e.preventDefault(); e.target.click(); } });
  onModelsReady(() => { $("runHint").textContent = "Models ready. Pick settings and press Run evaluation."; });
}

// Console hook: export every browser run as a recorded.json (see README, "Re-recording").
window.__lrj = {
  state: S, models: M, runEval,
  recorded() {
    const runs = [...S.runs.values()].filter((r) => r.source !== "recorded" && r.complete);
    if (!runs.length) throw new Error("no complete runs in this browser");
    const variant = runs[0].model.variant;
    return { model: { variant, judge: "VishalMysore/layaForWebTrained", embedder: runs[0].model.embedder }, createdAt: new Date().toISOString(),
      runs: Object.fromEntries(runs.filter((r) => r.model.variant === variant).map((r) => [r.key, compactRun(r)])) };
  },
};

(async () => {
  wire();
  initModelCard();
  try { S.corpus = await loadCorpus(); } catch (e) { $("runHint").textContent = "Could not load the corpus: " + e.message; return; }
  $("fDataset").innerHTML = `<option value="">All datasets</option>` + Object.entries(S.corpus.datasets).map(([k, v]) => `<option value="${k}">${esc(v)}</option>`).join("");
  const recorded = expandRecorded(await loadRecorded(), S.corpus);
  if (recorded) for (const r of Object.values(recorded.runs)) S.runs.set("rec|" + r.key, { ...r, source: "recorded" });
  pillIdle(!!recorded);
  try { for (const r of await db.getAll("runs")) S.runs.set(r.key, r); } catch { /* storage blocked */ }
  const want = runKey({ modelVariant: recorded?.model.variant || "q8e8", presetId: $("preset").value, rag: rag() });
  S.current = S.runs.get(want) || S.runs.get("rec|" + want) || [...S.runs.values()][0] || null;
  if (S.current) select(runId(S.current)); else { fillRunSelect(); render(); }
})();
