import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import { bm25, hybridTopK, chunkText, evidenceRetrieved } from "../web/rag.js";
import { routerQuestion, decideRoute, outcome, routerMetrics } from "../web/router-core.js";

const load = (f) => JSON.parse(fs.readFileSync(new URL(`../web/${f}`, import.meta.url), "utf8"));
const corpus = load("corpus.json"), router = load("router.json");

test("router data: routes map to corpus datasets; questions are well formed and evidence survives chunking", () => {
  const ids = new Set(router.routes.map((r) => r.id));
  for (const r of router.routes) if (r.dataset) assert.ok(corpus.datasets[r.dataset], `${r.id}: unknown dataset`);
  const seen = new Set();
  for (const q of router.questions) {
    assert.ok(!seen.has(q.id), `duplicate ${q.id}`); seen.add(q.id);
    assert.ok(ids.has(q.route), `${q.id}: unknown route`);
    if (q.route === "none") { assert.equal(q.docId, null); continue; }
    const doc = corpus.documents[q.docId];
    assert.ok(doc, `${q.id}: unknown doc`);
    assert.equal(doc.dataset, router.routes.find((r) => r.id === q.route).dataset, `${q.id}: doc is not in its route's knowledge base`);
    if (q.answerable) {
      assert.ok(doc.text.includes(q.evidence), `${q.id}: evidence not in doc`);
      assert.ok(evidenceRetrieved(chunkText(doc.text, { sentencesPerChunk: 2 }), q.evidence), `${q.id}: evidence split by chunking`);
    } else assert.equal(q.evidence, null);
  }
  for (const id of ids) assert.ok(router.questions.filter((q) => q.route === id).length >= 6, `route ${id} needs examples`);
});

test("router question has one option per route", () => {
  const q = routerQuestion(router.routes);
  assert.equal(q.type, "choice");
  assert.deepEqual(Object.keys(q.criteria), router.routes.map((r) => r.id));
});

test("bm25 rewards exact terms; hybrid mixes cosine and keywords", () => {
  const s = bm25("HTTP 429 limit", ["Requests receive an HTTP 429 response.", "The museum is open in May."]);
  assert.ok(s[0] > 0 && s[1] === 0);
  const items = [{ text: "HTTP 429 response on overload", vec: [0, 1] }, { text: "unrelated words here", vec: [1, 0] }];
  assert.equal(hybridTopK("HTTP 429", [1, 0], items, 1, 0.3)[0].item.text, items[0].text, "keywords win at low cosine weight");
  assert.equal(hybridTopK("HTTP 429", [1, 0], items, 1, 1)[0].item.text, items[1].text, "cosine wins at weight 1");
});

test("decideRoute: skip retrieval only when p(none) is high; pick the KB among KBs; fan out when its share is low", () => {
  // "none" is the single top option but below declineAt: still searched, in the leading knowledge base
  const lean = decideRoute({ none: 0.45, news: 0.4, policy: 0.1, encyclopedia: 0.05 });
  assert.ok(!lean.declined); assert.deepEqual(lean.searched, ["news"]); assert.ok(lean.share > 0.7);
  const none = decideRoute({ none: 0.8, news: 0.1, policy: 0.05, encyclopedia: 0.05 });
  assert.ok(none.declined); assert.deepEqual(none.searched, []); assert.equal(none.primary, "none");
  const unsure = decideRoute({ none: 0.2, news: 0.35, policy: 0.3, encyclopedia: 0.15 });
  assert.ok(!unsure.confident); assert.deepEqual(unsure.searched, ["news", "policy"]);
});

test("outcome and metrics score routing, the answer check and the end result", () => {
  const P = (text) => ({ text });
  const recs = [
    { route: "news", answerable: true, evidence: "7 to 2", routeProbs: { news: 0.8, policy: 0.1, encyclopedia: 0.05, none: 0.05 },
      perRoute: { news: { pAnswered: 0.9, passages: [P("voted 7 to 2")] }, policy: { pAnswered: 0.2, passages: [P("x")] } } },
    { route: "policy", answerable: false, evidence: null, routeProbs: { policy: 0.35, news: 0.3, encyclopedia: 0.25, none: 0.1 },
      perRoute: { policy: { pAnswered: 0.3, passages: [P("y")] }, news: { pAnswered: 0.35, passages: [P("z")] } } },
    { route: "none", routeProbs: { none: 0.9, news: 0.05, policy: 0.03, encyclopedia: 0.02 }, perRoute: {} },
    // small talk that isn't confidently skipped: searched, but the answer check finds nothing, so it's still right
    { route: "none", routeProbs: { none: 0.5, news: 0.3, policy: 0.1, encyclopedia: 0.1 }, perRoute: { news: { pAnswered: 0.1, passages: [P("q")] }, policy: { pAnswered: 0.2, passages: [P("r")] } } },
  ];
  assert.equal(outcome(recs[0]).kind, "answered"); assert.ok(outcome(recs[0]).correct);
  assert.equal(outcome(recs[1]).kind, "not_found"); assert.ok(outcome(recs[1]).correct); assert.ok(!outcome(recs[1]).confident);
  assert.equal(outcome(recs[2]).kind, "declined"); assert.ok(outcome(recs[2]).correct);
  assert.equal(outcome(recs[3]).kind, "not_found"); assert.ok(outcome(recs[3]).smallTalk); assert.ok(outcome(recs[3]).correct);
  const m = routerMetrics(recs, router.routes);
  assert.equal(m.kbChoiceAccuracy, 1); assert.equal(m.endToEnd, 1); assert.equal(m.fanOut, 1);
  assert.equal(m.smallTalkHandled, 1); assert.equal(m.smallTalkSkipped, 1);
  assert.equal(m.confusion.news.news, 1); assert.equal(m.answerCheckAccuracy, 1);
});

test("router-recorded.json matches the current routing question, answer check and corpus, and reproduces its score", async () => {
  const { questionKey } = await import("../web/judge.js");
  const { expandRouterRecorded } = await import("../web/pipeline.js");
  const { ANSWER_CHECK } = await import("../web/router-core.js");
  const rec = load("router-recorded.json");
  assert.equal(rec.fingerprint, questionKey({ router: routerQuestion(router.routes), check: ANSWER_CHECK }), "question wording changed since recording; re-record (see README)");
  const ex = expandRouterRecorded(rec, corpus, router);
  assert.equal(ex.records.length, router.questions.length, "every question recorded and every passage rebuilt from the corpus");
  const m = routerMetrics(ex.records, router.routes);
  assert.equal(m.endToEnd, 30 / 32);
  assert.equal(m.kbChoiceAccuracy, 23 / 24);
});
