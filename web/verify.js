// "Verify a claim" page: one claim through the whole pipeline, with every intermediate step on screen.
import { $, esc, num, settings, M, onModelsReady, initModelCard, loadRecorded, pillIdle, bars, GATE_TEXT } from "./common.js";
import { loadCorpus, indexCorpus, retrieve, judge, adHocPool, runKey, evidenceRetrieved, expandRecorded } from "./pipeline.js";
import { PRESETS, DEFAULT_PRESET, DEFAULT_THRESHOLD, gate, verdictOf } from "./judge.js";

const S = {
  corpus: null, recorded: null, last: null, busy: false, indexed: new Set(),
  cutoff: settings.get("cutoff", 0.5), threshold: settings.get("threshold", DEFAULT_THRESHOLD),
  selected: null, // claim object from the corpus, if the textarea still holds it
};

const rag = () => ({ sentencesPerChunk: +$("spc").value, overlap: 0, k: +$("k").value, scope: source() === "corpus" ? "corpus" : "document" });
const source = () => document.querySelector("input[name=source]:checked").value;
const preset = () => $("preset").value;

// ---- setup ----------------------------------------------------------------------------------------

function fillControls() {
  const c = S.corpus;
  $("dataset").innerHTML = Object.entries(c.datasets).map(([k, v]) => `<option value="${k}">${esc(v)} (${c.claims.filter((x) => c.documents[x.docId].dataset === k).length})</option>`).join("");
  $("doc").innerHTML = Object.entries(c.datasets).map(([k, v]) => `<optgroup label="${esc(v)}">${Object.entries(c.documents).filter(([, d]) => d.dataset === k).map(([id, d]) => `<option value="${id}">${esc(d.title)}</option>`).join("")}</optgroup>`).join("");
  $("preset").innerHTML = Object.entries(PRESETS).map(([k, p]) => `<option value="${k}">${esc(p.name.split(":")[0])}</option>`).join("");
  $("preset").value = settings.get("preset", DEFAULT_PRESET);
  $("spc").value = String(settings.get("spc", 2)); $("k").value = String(settings.get("k", 3));
  presetHint();
  $("dataset").value = settings.get("dataset", Object.keys(c.datasets)[0]);
  fillExamples();
}

function fillExamples() {
  const c = S.corpus, ds = $("dataset").value;
  settings.set("dataset", ds);
  const list = c.claims.filter((x) => c.documents[x.docId].dataset === ds);
  $("examples").innerHTML = "";
  for (const cl of list) {
    const b = document.createElement("button"); b.type = "button";
    b.textContent = cl.claim.length > 44 ? cl.claim.slice(0, 42) + "…" : cl.claim;
    b.title = `${c.documents[cl.docId].title}: ${cl.claim}`;
    b.dataset.id = cl.id;
    b.addEventListener("click", () => pick(cl));
    $("examples").appendChild(b);
  }
  markActive();
}

function pick(cl) {
  S.selected = cl;
  $("claim").value = cl.claim;
  $("doc").value = cl.docId;
  if (source() === "pasted") document.querySelector("input[name=source][value=document]").checked = true;
  sourceChanged();
  markActive(); hint();
  // Show a recorded result straight away when the models are not loaded yet.
  const r = !M.laya && recordedFor();
  if (r) show(r);
  else { S.last = null; $("placeholder").hidden = false; $("result").hidden = true; }
}

function markActive() { [...$("examples").children].forEach((b) => b.classList.toggle("active", b.dataset.id === S.selected?.id)); }
function presetHint() { $("presetHint").textContent = PRESETS[preset()].name + ". The evaluation page compares the phrasings."; }
function sourceChanged() { const s = source(); $("docField").hidden = s !== "document"; $("pasteField").hidden = s !== "pasted"; hint(); }

function recordedFor() {
  const cl = S.selected;
  if (!cl || !S.recorded || source() === "pasted" || $("claim").value.trim() !== cl.claim) return null;
  const key = runKey({ modelVariant: S.recorded.model.variant, presetId: preset(), rag: rag() });
  const run = S.recorded.runs[key];
  const rec = run?.records.find((r) => r.id === cl.id);
  if (!rec || (source() === "document" && $("doc").value !== cl.docId)) return null;
  return { ...rec, source: "recorded", question: PRESETS[preset()].question, state: { claim: rec.claim, evidence: rec.passages.map((p) => p.text) }, rag: run.rag, docId: cl.docId, presetId: preset() };
}

function hint() {
  const h = $("runHint");
  if (S.busy) return;
  if (M.laya) h.textContent = "Runs on this device with the loaded models.";
  else if (recordedFor()) h.textContent = "Models not loaded: this sample claim plays back a result recorded from the same models at these settings.";
  else h.textContent = "This needs the models (a custom claim, pasted text, or settings without a recording). Load them above; they download once, then stay cached.";
}

// ---- running ---------------------------------------------------------------------------------------

async function verify() {
  if (S.busy) return;
  const claim = $("claim").value.trim();
  if (!claim) { $("runHint").textContent = "Write a claim first."; return; }
  if (!M.laya) {
    const r = recordedFor();
    if (r) return show(r);
    $("runHint").textContent = "Load the models first (button above). Sample claims at the default settings play back recorded results without them.";
    return;
  }
  S.busy = true; $("verifyBtn").disabled = true;
  try {
    const r = rag(), src = source();
    let pool = null, docId = null;
    if (src === "pasted") {
      const text = $("pasted").value.trim();
      if (!text) throw new Error("Paste some evidence text first.");
      $("runHint").textContent = "Chunking and embedding your text…";
      pool = await adHocPool(M.embedder, text, r);
    } else {
      docId = $("doc").value;
      const cfgKey = `${r.sentencesPerChunk}`;
      if (!S.indexed.has(cfgKey)) {
        await indexCorpus(S.corpus, M.embedder, r, (i, n) => { $("runHint").textContent = `Indexing the corpus into IndexedDB: ${i} / ${n} documents…`; });
        S.indexed.add(cfgKey);
      }
    }
    $("runHint").textContent = "Retrieving evidence…";
    const ret = await retrieve(M.embedder, claim, { rag: r, docId, pool });
    $("runHint").textContent = "Asking Laya…";
    const j = await judge(M.laya, claim, ret.passages, preset());
    const cl = S.selected && S.selected.claim === claim ? S.selected : null;
    show({
      source: "model", claim, passages: ret.passages, candidates: ret.candidates, embedMs: ret.embedMs, searchMs: ret.searchMs,
      pSupported: j.pSupported, confidence: j.confidence, probabilities: j.probabilities, layaMs: j.layaMs, tokens: j.tokens,
      question: j.question, state: j.state, rag: r, docId: src === "pasted" ? null : (r.scope === "corpus" ? null : docId), pasted: src === "pasted" ? $("pasted").value : null,
      gold: cl?.label, kind: cl?.kind, evidence: cl?.evidence, id: cl?.id, presetId: preset(),
    });
    $("runHint").textContent = "Runs on this device with the loaded models.";
  } catch (e) {
    console.error(e);
    $("runHint").textContent = "Error: " + (e?.message || e);
  } finally { S.busy = false; $("verifyBtn").disabled = false; }
}

// ---- rendering ---------------------------------------------------------------------------------------

function show(res) {
  S.last = res;
  $("placeholder").hidden = true; $("result").hidden = false;
  const c = S.corpus;
  const cl = res.id ? c.claims.find((x) => x.id === res.id) : null;
  const gold = res.gold ?? cl?.label, evidence = res.evidence ?? cl?.evidence;
  const g = gate(res.pSupported, res.confidence, { cutoff: S.cutoff, threshold: S.threshold });
  const verdict = verdictOf(res.pSupported, S.cutoff);
  const G = GATE_TEXT[g];
  const correct = gold ? verdict === gold : null;
  $("outcome").innerHTML = `<div class="outcome ${G.cls}"><div class="k">${G.k}</div><div class="t">${esc(G.t)}</div>
    <div class="d">p(supported) ${num(res.pSupported)} ${verdict === "supported" ? "≥" : "<"} cutoff ${num(S.cutoff)} → <b>${verdict.replace("_", " ")}</b>; confidence ${num(res.confidence, 3)} ${res.confidence >= S.threshold ? "≥" : "<"} ${num(S.threshold)}. ${esc(G.d)}</div>
    ${gold ? `<div class="verdictline">Gold label: <b>${gold.replace("_", " ")}</b> <span class="tag">${esc(res.kind || cl?.kind || "")}</span> ${correct ? '<span class="tag ok">judge is right</span>' : '<span class="tag bad">judge is wrong</span>'}${res.source === "recorded" ? ' <span class="tag warn">recorded result</span>' : ""}</div>` : res.source === "recorded" ? '<div class="verdictline"><span class="tag warn">recorded result</span></div>' : ""}
  </div>`;

  const hit = evidence ? evidenceRetrieved(res.passages, evidence) : null;
  $("retrMeta").textContent = `top ${res.passages.length}${res.candidates ? ` of ${res.candidates} passages` : ""} by cosine similarity · ${res.rag.sentencesPerChunk} sentence${res.rag.sentencesPerChunk > 1 ? "s" : ""} per chunk${res.embedMs == null ? "" : res.embedMs >= 1 ? ` · claim embedded in ${res.embedMs.toFixed(0)} ms` : " · claim embedding reused from IndexedDB"}${hit == null ? "" : hit ? " · gold evidence retrieved ✓" : " · gold evidence NOT retrieved"}`;
  $("passages").innerHTML = res.passages.map((p, i) => {
    const doc = c.documents[p.docId];
    const isGold = evidence && p.text.includes(evidence);
    return `<li class="passage${isGold ? " gold" : ""}"><span class="rank">${i + 1}</span><div>
      <div class="src"><span>${esc(doc ? doc.title : "Your text")} · passage ${p.chunk + 1}</span><span class="score"><span class="tr"><span class="f" style="width:${Math.max(0, p.score) * 100}%"></span></span>cos ${p.score.toFixed(3)}</span>${isGold ? '<span class="tag ok">contains gold evidence</span>' : ""}</div>
      <div class="txt">${highlight(p.text, evidence)}</div></div></li>`;
  }).join("");

  $("judgeMeta").textContent = `${PRESETS[res.presetId || preset()].name}${res.layaMs != null ? ` · ${res.layaMs.toFixed(0)} ms` : ""}${res.tokens ? ` · ${res.tokens} tokens` : ""}`;
  const sel = Object.entries(res.probabilities).sort((a, b) => b[1] - a[1])[0][0];
  $("probs").innerHTML = bars(res.probabilities, sel);
  $("qjson").textContent = JSON.stringify({ question: res.question, state: res.state }, null, 2);

  // Source document with retrieved passages marked (one document, or each document that contributed).
  const docIds = res.pasted != null ? [] : [...new Set(res.passages.map((p) => p.docId))];
  if (res.pasted != null) {
    $("docview").innerHTML = res.passages.map((p, i) => `<mark>${esc(p.text)}<sup>${i + 1}</sup></mark>`).join(" … ");
  } else {
    $("docview").innerHTML = docIds.map((id) => {
      const d = c.documents[id];
      let html = esc(d.text);
      res.passages.forEach((p, i) => { if (p.docId === id) html = html.replace(esc(p.text), `<mark>${esc(p.text)}<sup>${i + 1}</sup></mark>`); });
      if (evidence && id === (cl?.docId || res.docId)) html = html.replace(esc(evidence), `<mark class="ev">${esc(evidence)}</mark>`);
      return `<div style="margin-bottom:8px"><b>${esc(d.title)}</b> <span class="tag">${esc(c.datasets[d.dataset])}</span><br>${html}</div>`;
    }).join("");
  }
  $("evNote").textContent = evidence ? "; the gold evidence span is underlined" : "";
  markActive(); hint();
}

function highlight(text, ev) {
  if (!ev || !text.includes(ev)) return esc(text);
  return esc(text).replace(esc(ev), `<mark class="ev">${esc(ev)}</mark>`);
}

// ---- wiring ------------------------------------------------------------------------------------------

function wire() {
  $("dataset").addEventListener("change", fillExamples);
  $("verifyBtn").addEventListener("click", verify);
  $("claim").addEventListener("input", () => { if (S.selected && $("claim").value.trim() !== S.selected.claim) { S.selected = null; markActive(); } hint(); });
  $("claim").addEventListener("keydown", (e) => { if ((e.ctrlKey || e.metaKey) && e.key === "Enter") verify(); });
  document.querySelectorAll("input[name=source]").forEach((r) => r.addEventListener("change", sourceChanged));
  $("doc").addEventListener("change", hint);
  for (const [id, key] of [["spc", "spc"], ["k", "k"], ["preset", "preset"]]) $(id).addEventListener("change", () => { settings.set(key, id === "preset" ? $(id).value : +$(id).value); presetHint(); hint(); });
  const slider = (id, key, valId) => {
    $(id).value = S[key]; $(valId).textContent = (+S[key]).toFixed(2);
    $(id).addEventListener("input", () => { S[key] = +$(id).value; $(valId).textContent = S[key].toFixed(2); settings.set(key, S[key]); if (S.last) show(S.last); });
  };
  slider("cutoff", "cutoff", "cutoffVal");
  slider("thresh", "threshold", "threshVal");
  onModelsReady(hint);
}

window.__lrj = { state: S, models: M, verify };

(async () => {
  wire();
  initModelCard();
  try { S.corpus = await loadCorpus(); } catch (e) { $("runHint").textContent = "Could not load the corpus: " + e.message; return; }
  S.recorded = expandRecorded(await loadRecorded(), S.corpus);
  pillIdle(!!S.recorded);
  fillControls();
  const first = S.corpus.claims.find((x) => S.corpus.documents[x.docId].dataset === $("dataset").value);
  if (first) pick(first);
})();
