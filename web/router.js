// Router page: Laya picks the knowledge base for a question, retrieval runs inside it, and Laya checks whether the
// passages answer the question. Routing decisions are re-derived from stored probabilities when a slider moves.
import { $, esc, pct, num, settings, M, onModelsReady, initModelCard, pillIdle, bars, yieldToBrowser } from "./common.js";
import { loadCorpus, indexCorpus, runRouter, ROUTER_RAG, compactRouterRecord, expandRouterRecorded } from "./pipeline.js";
import { routerQuestion, ANSWER_CHECK, outcome, routerMetrics, DEFAULT_DECLINE_AT, DEFAULT_KB_THRESHOLD, DEFAULT_ANSWER_CUTOFF } from "./router-core.js";
import { questionKey } from "./judge.js";
import * as db from "./store.js";

const S = {
  corpus: null, router: null, routes: [], fp: "", records: new Map(), sources: new Map(), last: null, selected: null,
  busy: false, stop: false, indexed: false,
  declineAt: settings.get("decl", DEFAULT_DECLINE_AT), kbThreshold: settings.get("kbt", DEFAULT_KB_THRESHOLD), cutoff: settings.get("acut", DEFAULT_ANSWER_CUTOFF),
};
const routeName = (id) => S.routes.find((r) => r.id === id)?.name || id;
const runKey = () => `router|${M.model?.variant || "q8e8"}|${S.fp}`;
const opts = () => ({ declineAt: S.declineAt, kbThreshold: S.kbThreshold, cutoff: S.cutoff });

// ---- setup -------------------------------------------------------------------------------------------

function fillSide() {
  $("routes").innerHTML = S.routes.map((r) => `<li><b>${esc(r.name)}</b><br>${esc(r.note)}</li>`).join("");
  $("examples").innerHTML = "";
  for (const q of S.router.questions) {
    const b = document.createElement("button"); b.type = "button"; b.dataset.id = q.id;
    b.textContent = `${q.question.length > 40 ? q.question.slice(0, 38) + "…" : q.question} [${q.route}]`;
    b.title = `${q.question}\nCorrect route: ${routeName(q.route)}${q.route !== "none" ? (q.answerable ? " (answer is in it)" : " (answer is NOT in it)") : ""}`;
    b.addEventListener("click", () => pick(q));
    $("examples").appendChild(b);
  }
}

function pick(q) {
  S.selected = q; $("question").value = q.question;
  markActive(); hint();
  const rec = S.records.get(q.id);
  if (rec) show(rec); else { S.last = null; $("placeholder").hidden = false; $("result").hidden = true; }
}
function markActive() { [...$("examples").children].forEach((b) => b.classList.toggle("active", b.dataset.id === S.selected?.id)); }
function hint() {
  if (S.busy) return;
  const q = $("question").value.trim(), rec = S.selected && S.selected.question === q && S.records.get(S.selected.id);
  $("runHint").textContent = M.laya ? "Runs on this device with the loaded models."
    : rec ? `Models not loaded: showing a ${S.sources.get(S.selected.id) === "recorded" ? "recorded" : "stored"} result for this sample question.`
    : "This needs the models. Load them above; they download once, then stay cached.";
}

// ---- running ---------------------------------------------------------------------------------------

async function ensureIndex() {
  if (S.indexed) return;
  await indexCorpus(S.corpus, M.embedder, ROUTER_RAG, (i, n) => { $("runHint").textContent = `Indexing the knowledge bases into IndexedDB: ${i} / ${n} documents…`; });
  S.indexed = true;
}

async function routeOne() {
  if (S.busy) return;
  const question = $("question").value.trim();
  if (!question) { $("runHint").textContent = "Write a question first."; return; }
  if (!M.laya) {
    const rec = S.selected && S.selected.question === question && S.records.get(S.selected.id);
    if (rec) return show(rec);
    $("runHint").textContent = "Load the models first (button above). Sample questions play back recorded results without them.";
    return;
  }
  S.busy = true; $("routeBtn").disabled = true;
  try {
    await ensureIndex();
    const res = await runRouter({ laya: M.laya, embedder: M.embedder, routes: S.routes, question, onStep: (m) => { $("runHint").textContent = m; } });
    const gold = S.selected && S.selected.question === question ? S.selected : null;
    const rec = { ...(gold || { id: null }), ...res };
    if (gold) { S.records.set(gold.id, rec); S.sources.set(gold.id, "live"); await saveRun(); renderEval(); }
    show(rec);
    $("runHint").textContent = "Runs on this device with the loaded models.";
  } catch (e) { console.error(e); $("runHint").textContent = "Error: " + (e?.message || e); }
  finally { S.busy = false; $("routeBtn").disabled = false; }
}

async function runAll() {
  if (S.busy) return;
  if (!M.laya) { $("evalMeta").textContent = "Load the models first to run the questions live."; return; }
  S.busy = true; S.stop = false; $("runAllBtn").disabled = true; $("stopBtn").disabled = false; $("routeBtn").disabled = true;
  const qs = S.router.questions;
  try {
    await ensureIndex();
    for (let i = 0; i < qs.length && !S.stop; i++) {
      const q = qs[i];
      if (S.sources.get(q.id) !== "live") {
        const res = await runRouter({ laya: M.laya, embedder: M.embedder, routes: S.routes, question: q.question });
        S.records.set(q.id, { ...q, ...res }); S.sources.set(q.id, "live");
      }
      $("evalBar").style.width = `${(100 * (i + 1) / qs.length).toFixed(1)}%`; $("evalCount").textContent = `${i + 1} of ${qs.length}`;
      if ((i + 1) % 4 === 0 || i === qs.length - 1) { renderEval(); await saveRun(); }
      await yieldToBrowser();
    }
  } catch (e) { console.error(e); $("evalMeta").textContent = "Error: " + (e?.message || e); }
  finally { S.busy = false; $("runAllBtn").disabled = false; $("stopBtn").disabled = true; $("routeBtn").disabled = false; await saveRun(); renderEval(); hint(); }
}

async function saveRun() {
  const live = [...S.records.values()].filter((r) => S.sources.get(r.id) === "live").map(({ routerInput, ...r }) => r);
  if (live.length) await db.putMany("runs", [{ key: runKey(), kind: "router", records: live, savedAt: new Date().toISOString() }]).catch(() => {});
}

// ---- rendering ---------------------------------------------------------------------------------------

const hl = (text, ev) => (ev && text.includes(ev) ? esc(text).replace(esc(ev), `<mark class="ev">${esc(ev)}</mark>`) : esc(text));

function show(rec) {
  S.last = rec;
  $("placeholder").hidden = true; $("result").hidden = false;
  const o = outcome(rec, opts());
  const src = rec.id ? S.sources.get(rec.id) : "live";
  // outcome box
  let cls, k, t, d;
  if (o.kind === "declined") { cls = "human"; k = "NO RETRIEVAL"; t = "Not a question about the documents"; d = `p(No retrieval) ${num(o.pNone)} ≥ ${num(S.declineAt)}, so nothing was searched. The assistant would answer directly or decline.`; }
  else if (o.kind === "answered") { cls = "auto"; k = `ANSWER FOUND · ${routeName(o.answer.route).toUpperCase()}`; t = "The passages answer the question"; d = `p(answered) ${num(o.answer.pAnswered)} ≥ ${num(S.cutoff)}. The top passage below is the evidence an LLM would answer from.`; }
  else if (o.smallTalk) { cls = "human"; k = "NOTHING IN THE DOCUMENTS"; t = "No document answers it, and it may not need one"; d = `"No retrieval" was the router's top option (p ${num(o.pNone)}) but below the ${num(S.declineAt)} bar, so it searched ${o.searched.map(routeName).join(" and ")} to be safe. Best p(answered) ${o.answer ? num(o.answer.pAnswered) : "–"} < ${num(S.cutoff)}: no document passage is served. The assistant should answer without the documents, or say it doesn't know.`; }
  else { cls = "block"; k = "NOT FOUND"; t = "No searched knowledge base answers it"; d = `Best p(answered) ${o.answer ? num(o.answer.pAnswered) : "–"} < ${num(S.cutoff)} in ${o.searched.map(routeName).join(" and ")}. The assistant should say it doesn't know instead of guessing.`; }
  const gold = rec.route ? `<div class="verdictline">Correct route: <b>${esc(routeName(rec.route))}</b>${rec.route !== "none" ? ` <span class="tag">${rec.answerable ? "answer is in it" : "answer is not in it"}</span>` : ""} ${o.correct ? '<span class="tag ok">router got this right</span>' : '<span class="tag bad">router got this wrong</span>'}${src === "recorded" ? ' <span class="tag warn">recorded result</span>' : ""}</div>` : "";
  $("outcome").innerHTML = `<div class="outcome ${cls}"><div class="k">${esc(k)}</div><div class="t">${esc(t)}</div><div class="d">${esc(d)}</div>${gold}</div>`;

  // router
  $("routerMeta").textContent = `${o.declined ? `p(No retrieval) ${num(o.pNone)} ≥ ${num(S.declineAt)}: skip retrieval` : `p(No retrieval) ${num(o.pNone)} < ${num(S.declineAt)}: search. ${routeName(o.topKb)} has ${pct(o.share)} of the knowledge-base probability, ${o.confident ? `≥ ${pct(S.kbThreshold)}: search it alone` : `< ${pct(S.kbThreshold)}: search the top two`}`}${rec.routerMs ? ` · ${Math.round(rec.routerMs)} ms` : ""}`;
  $("routeflow").innerHTML = S.routes.map((r) => {
    const c = o.declined && r.id === "none" ? "declined" : o.searched.includes(r.id) ? "searched" : "";
    return `<div class="rt ${c}${r.id === o.primary ? " top" : ""}"><b>${esc(r.name)}</b>${pct(rec.routeProbs[r.id])}${o.searched.includes(r.id) ? " · searched" : o.declined && r.id === "none" ? " · chosen" : ""}</div>`;
  }).join("");
  $("routeProbs").innerHTML = bars(Object.fromEntries(S.routes.map((r) => [r.name, rec.routeProbs[r.id]])), routeName(o.primary));
  const inp = rec.routerInput;
  $("rseqNote").textContent = inp ? `${inp.ids.length} tokens. Each [MASK] is one route; Laya scores all four at once.` : "Recorded result: load the models and route the question again to see the exact token sequence.";
  $("rseq").textContent = inp ? inp.text.replace(/\s*(\[SEP\])\s*/g, "\n$1\n").replace(/\s*(\[MASK\])/g, "\n$1").trim() : "";

  // retrieval + answer check per knowledge base (the two most likely ones are always computed)
  const ids = Object.keys(rec.perRoute).sort((a, b) => rec.routeProbs[b] - rec.routeProbs[a]);
  $("searchMeta").textContent = o.declined ? "nothing searched" : `searched: ${o.searched.map(routeName).join(" + ")}`;
  $("searched").innerHTML = ids.map((id) => {
    const x = rec.perRoute[id], route = S.routes.find((r) => r.id === id), searched = o.searched.includes(id);
    const best = o.answer && o.answer.route === id && o.kind === "answered";
    const passages = x.passages.map((p, i) => `<li class="passage${rec.evidence && p.text.includes(rec.evidence) ? " gold" : ""}"><span class="rank">${i + 1}</span><div>
      <div class="src"><span>${esc(S.corpus.documents[p.docId]?.title || p.docId)} · passage ${p.chunk + 1}</span><span>cos ${num(p.cosine, 3)}</span>${p.keyword != null ? `<span>BM25 ${num(p.keyword, 2)}</span><span>hybrid ${num(p.score, 3)}</span>` : ""}${rec.evidence && p.text.includes(rec.evidence) ? '<span class="tag ok">contains the answer</span>' : ""}</div>
      <div class="txt">${hl(p.text, rec.evidence)}</div></div></li>`).join("");
    return `<div class="routecol${best ? " best" : ""}" style="${searched ? "" : "opacity:.55"}">
      <h4>${esc(route.name)} <span class="tag">${route.retriever === "hybrid" ? "hybrid: cosine + BM25" : "dense: cosine"}</span>${searched ? '<span class="tag ok">searched</span>' : '<span class="tag">not searched at this threshold</span>'}${best ? '<span class="tag ok">answer taken from here</span>' : ""}</h4>
      <ol class="passages">${passages}</ol>
      <div style="margin-top:6px">${bars({ answered: x.pAnswered, not_answered: 1 - x.pAnswered }, x.pAnswered >= S.cutoff ? "answered" : "not_answered")}</div></div>`;
  }).join("") || '<p class="hint">No knowledge base was searched.</p>';
  markActive();
}

function renderEval() {
  const recs = S.router.questions.map((q) => S.records.get(q.id)).filter(Boolean);
  const nLive = recs.filter((r) => S.sources.get(r.id) === "live").length;
  $("evalMeta").textContent = recs.length ? `${recs.length} of ${S.router.questions.length} questions · ${nLive === recs.length ? "all run in this browser" : nLive ? `${nLive} run in this browser, the rest recorded` : "recorded results"}` : "No results yet. Load the models and press Run all questions.";
  if (!recs.length) { for (const id of ["rkpis", "confusion", "perRoute", "rtable"]) $(id).innerHTML = ""; return; }
  const m = routerMetrics(recs, S.routes, opts());
  const kpi = (l, v, s = "") => `<div class="kpi"><div class="l">${l}</div><div class="v">${v}${s ? ` <small>${s}</small>` : ""}</div></div>`;
  $("rkpis").innerHTML = [
    kpi("End to end", pct(m.endToEnd, 0), "right final outcome"),
    kpi("Knowledge-base choice", pct(m.kbChoiceAccuracy, 0), "top KB correct"),
    kpi("Right KB searched", pct(m.goldSearched, 0), `incl. ${m.fanOut} fan-outs`),
    kpi("Small talk handled", pct(m.smallTalkHandled, 0), `${m.smallTalkSkipped} skipped outright`),
    kpi("Answer check", pct(m.answerCheckAccuracy, 0), "answerable or not"),
    kpi("Evidence recall@3", pct(m.evidenceRecall, 0), "answer retrieved"),
  ].join("");
  const ids = S.routes.map((r) => r.id);
  $("confusion").innerHTML = `<table class="grid"><thead><tr><th>Correct ↓ / routed →</th>${ids.map((id) => `<th class="r">${esc(routeName(id).replace(" RAG", ""))}</th>`).join("")}</tr></thead><tbody>${
    ids.map((g) => `<tr><td>${esc(routeName(g))}</td>${ids.map((p) => `<td class="r${g === p ? " diag" : ""}">${m.confusion[g][p] || (g === p ? 0 : "·")}</td>`).join("")}</tr>`).join("")}</tbody></table>`;
  $("perRoute").innerHTML = `<table class="grid"><thead><tr><th>Correct route</th><th class="r">Questions</th><th class="r">Sent there</th><th class="r">Final outcome right</th></tr></thead><tbody>${
    m.perRoute.map((r) => `<tr><td>${esc(routeName(r.id))}</td><td class="r">${r.n}</td><td class="r">${pct(r.routed, 0)}</td><td class="r">${pct(r.correct, 0)}</td></tr>`).join("")}</tbody></table>
    <p class="caption">Answerable questions answered from the right passage: <b>${pct(m.answeredCorrectly, 0)}</b>. Unanswerable ones correctly reported as not found: <b>${pct(m.notFoundCorrectly, 0)}</b>. Knowledge questions wrongly skipped: ${m.wronglySkipped}. For small talk, "final outcome right" means no document passage was served as its answer.</p>`;
  $("rtable").innerHTML = `<thead><tr><th>Question</th><th>Correct route</th><th>Sent to</th><th>Searched</th><th>Outcome</th><th></th></tr></thead><tbody>${
    m.outs.map(({ r, o }) => `<tr class="rrow" data-id="${r.id}" tabindex="0"><td>${esc(r.question)}</td><td>${esc(routeName(r.route))}${r.route !== "none" ? (r.answerable ? "" : " <span class=\"tag\">no answer</span>") : ""}</td>
      <td>${esc(routeName(o.primary))} <small>${o.declined ? num(o.pNone) : pct(o.share)}</small></td><td>${o.declined ? "–" : o.searched.map((id) => esc(routeName(id).replace(" RAG", ""))).join(" + ")}</td>
      <td>${o.kind === "answered" ? `answered (${esc(routeName(o.answer.route).replace(" RAG", ""))})` : o.kind === "declined" ? "no retrieval" : "not found"}</td><td><span class="tag ${o.correct ? "ok" : "bad"}">${o.correct ? "right" : "wrong"}</span></td></tr>`).join("")}</tbody>`;
}

// ---- wiring -------------------------------------------------------------------------------------------

function wire() {
  $("routeBtn").addEventListener("click", routeOne);
  $("runAllBtn").addEventListener("click", runAll);
  $("stopBtn").addEventListener("click", () => { S.stop = true; });
  $("question").addEventListener("input", () => { if (S.selected && $("question").value.trim() !== S.selected.question) { S.selected = null; markActive(); } hint(); });
  $("question").addEventListener("keydown", (e) => { if ((e.ctrlKey || e.metaKey) && e.key === "Enter") routeOne(); });
  const slider = (id, key, valId) => {
    $(id).value = S[key]; $(valId).textContent = (+S[key]).toFixed(2);
    $(id).addEventListener("input", () => { S[key] = +$(id).value; $(valId).textContent = S[key].toFixed(2); settings.set(id, S[key]); if (S.last) show(S.last); renderEval(); });
  };
  slider("decl", "declineAt", "declVal");
  slider("kbt", "kbThreshold", "kbtVal");
  slider("acut", "cutoff", "acutVal");
  $("rtable").addEventListener("click", (e) => {
    const tr = e.target.closest(".rrow"); if (!tr) return;
    const q = S.router.questions.find((x) => x.id === tr.dataset.id); pick(q);
    $("resultCard").scrollIntoView({ behavior: "smooth", block: "start" });
  });
  $("rtable").addEventListener("keydown", (e) => { if ((e.key === "Enter" || e.key === " ") && e.target.classList.contains("rrow")) { e.preventDefault(); e.target.click(); } });
  onModelsReady(hint);
}

// Console hook: export the live run as router-recorded.json (see README).
window.__lrj = {
  state: S, models: M, routeOne, runAll,
  routerRecorded() {
    const recs = S.router.questions.map((q) => S.records.get(q.id)).filter((r) => r && S.sources.get(r.id) === "live");
    return { model: { variant: M.model?.variant || "q8e8", judge: "VishalMysore/layaForWebTrained", embedder: M.embedder?.id }, fingerprint: S.fp, createdAt: new Date().toISOString(), records: recs.map(compactRouterRecord) };
  },
};

(async () => {
  wire();
  initModelCard();
  try {
    S.corpus = await loadCorpus();
    S.router = await (await fetch("./router.json", { cache: "no-cache" })).json();
  } catch (e) { $("runHint").textContent = "Could not load the data: " + e.message; return; }
  S.routes = S.router.routes;
  S.fp = questionKey({ router: routerQuestion(S.routes), check: ANSWER_CHECK });
  let recorded = null;
  try { const r = await fetch("./router-recorded.json", { cache: "no-cache" }); if (r.ok) recorded = await r.json(); } catch { /* optional */ }
  if (recorded && recorded.fingerprint === S.fp) {
    for (const rec of expandRouterRecorded(recorded, S.corpus, S.router).records) { S.records.set(rec.id, rec); S.sources.set(rec.id, "recorded"); }
  }
  try {
    const run = await db.get("runs", runKey());
    if (run) for (const rec of run.records) { const q = S.router.questions.find((x) => x.id === rec.id); if (q) { S.records.set(rec.id, { ...rec, ...q }); S.sources.set(rec.id, "live"); } }
  } catch { /* storage blocked */ }
  pillIdle(S.records.size > 0);
  fillSide();
  renderEval();
  pick(S.router.questions[0]);
})();
