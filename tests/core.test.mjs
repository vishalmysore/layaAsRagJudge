import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import { splitSentences, chunkText, cosine, topK, evidenceRetrieved } from "../web/rag.js";
import { PRESETS, interpret, gate, verdictOf, binaryConfidence, buildState } from "../web/judge.js";
import { evaluate, auroc, bestCutoff, percentile, coverageTable } from "../web/metrics.js";

const corpus = JSON.parse(fs.readFileSync(new URL("../web/corpus.json", import.meta.url), "utf8"));

test("sentence splitter keeps decimals, abbreviations and numbers intact", () => {
  const s = splitSentences("It covers 6.5 square kilometres. The lead author, Dr. Sofia Lindqvist, agreed. Trains start at 9:00 daily. It cost 1.1 billion dollars. A ferry ran from the 1850s. J. Smith agreed.");
  assert.deepEqual(s, [
    "It covers 6.5 square kilometres.",
    "The lead author, Dr. Sofia Lindqvist, agreed.",
    "Trains start at 9:00 daily.",
    "It cost 1.1 billion dollars.",
    "A ferry ran from the 1850s.",
    "J. Smith agreed.",
  ]);
});

test("chunking covers every sentence exactly once without overlap, and slides with overlap", () => {
  const text = "One ant. Two bees. Three cats. Four dogs. Five eels.";
  const c2 = chunkText(text, { sentencesPerChunk: 2 });
  assert.deepEqual(c2.map((c) => c.text), ["One ant. Two bees.", "Three cats. Four dogs.", "Five eels."]);
  const c3o1 = chunkText(text, { sentencesPerChunk: 3, overlap: 1 });
  assert.deepEqual(c3o1.map((c) => c.firstSentence), [0, 2]);
  assert.equal(c3o1.at(-1).text, "Three cats. Four dogs. Five eels.");
});

test("cosine and topK rank by similarity", () => {
  assert.equal(Math.round(cosine([1, 0], [1, 0]) * 1e6) / 1e6, 1);
  assert.equal(cosine([1, 0], [0, 1]), 0);
  const items = [{ id: "a", vec: [0, 1] }, { id: "b", vec: [1, 0.1] }, { id: "c", vec: [0.7, 0.7] }];
  assert.deepEqual(topK([1, 0], items, 2).map((x) => x.item.id), ["b", "c"]);
});

test("corpus: every claim has a document, a valid label/kind, and a verbatim evidence span", () => {
  const ids = new Set();
  for (const c of corpus.claims) {
    assert.ok(!ids.has(c.id), `duplicate id ${c.id}`); ids.add(c.id);
    const doc = corpus.documents[c.docId];
    assert.ok(doc, `${c.id}: unknown document ${c.docId}`);
    assert.ok(corpus.datasets[doc.dataset], `${c.id}: unknown dataset`);
    assert.ok(["supported", "not_supported"].includes(c.label), `${c.id}: bad label`);
    assert.ok(corpus.kinds[c.kind], `${c.id}: unknown kind ${c.kind}`);
    const sup = c.kind === "paraphrase" || c.kind === "inference";
    assert.equal(c.label === "supported", sup, `${c.id}: kind ${c.kind} does not match label ${c.label}`);
    if (c.kind === "unverifiable") assert.equal(c.evidence, null, `${c.id}: unverifiable claims have no evidence span`);
    else assert.ok(doc.text.includes(c.evidence), `${c.id}: evidence span not found in ${c.docId}`);
  }
  const sup = corpus.claims.filter((c) => c.label === "supported").length;
  assert.equal(sup * 2, corpus.claims.length, "sample is balanced");
});

test("corpus: every evidence span survives chunking (lies inside one chunk) at 1-4 sentences per chunk", () => {
  for (const n of [1, 2, 3, 4]) {
    for (const c of corpus.claims.filter((x) => x.evidence)) {
      const chunks = chunkText(corpus.documents[c.docId].text, { sentencesPerChunk: n });
      assert.ok(evidenceRetrieved(chunks, c.evidence), `${c.id} evidence split across chunks at ${n} sentences`);
    }
  }
});

test("judge: interpret reads p(supported) for each preset; gate routes AUTO / BLOCK / HOLD", () => {
  assert.equal(interpret({ type: "choice", probabilities: { supported: 0.8, not_supported: 0.2 } }, "choice2").pSupported, 0.8);
  assert.equal(interpret({ type: "choice", probabilities: { supported: 0.5, contradicted: 0.3, not_mentioned: 0.2 } }, "choice3").pSupported, 0.5);
  const n = interpret({ type: "noul", noul: 0.3, confidence: 0.7 }, "noul");
  assert.equal(n.pSupported, 0.3); assert.equal(n.probabilities.not_supported, 0.7);
  assert.equal(binaryConfidence(0.5), 0);
  assert.ok(binaryConfidence(0.99) > 0.9);
  assert.equal(gate(0.95, binaryConfidence(0.95), { threshold: 0.5 }), "AUTO");
  assert.equal(gate(0.05, binaryConfidence(0.05), { threshold: 0.5 }), "BLOCK");
  assert.equal(gate(0.6, binaryConfidence(0.6), { threshold: 0.5 }), "HOLD");
  assert.equal(verdictOf(0.4, 0.35), "supported");
  assert.deepEqual(buildState(" x ", [{ text: "a" }, "b"]), { claim: "x", evidence: ["a", "b"] });
  for (const p of Object.values(PRESETS)) assert.ok(p.question.type && p.question.instructions);
});

test("metrics: accuracy, balanced accuracy, false-verification rate, AUROC, coverage", () => {
  const R = (gold, p, extra = {}) => ({ gold, pSupported: p, confidence: binaryConfidence(p), dataset: "d", kind: "k", layaMs: 10, embedMs: 1, ...extra });
  const rs = [R("supported", 0.9), R("supported", 0.7), R("supported", 0.4), R("not_supported", 0.6), R("not_supported", 0.2), R("not_supported", 0.1)];
  const m = evaluate(rs, { cutoff: 0.5, threshold: 0 });
  assert.deepEqual(m.confusion, { tp: 2, fp: 1, tn: 2, fn: 1 });
  assert.equal(m.accuracy, 4 / 6);
  assert.equal(m.falseVerificationRate, 1 / 3);
  assert.equal(m.balancedAccuracy, (2 / 3 + 2 / 3) / 2);
  assert.equal(auroc(rs), 8 / 9);
  assert.ok(bestCutoff(rs).balancedAccuracy >= m.balancedAccuracy);
  assert.equal(percentile([1, 2, 3, 4], 0.5), 2.5);
  const cov = coverageTable(rs, 0.5, [0, 0.99]);
  assert.equal(cov[0].coverage, 1); assert.equal(cov[1].kept, 0);
  assert.equal(m.gates.AUTO + m.gates.BLOCK + m.gates.HOLD, rs.length);
});

test("recorded.json matches the corpus and the current question wording, and its metrics reproduce", async () => {
  const { expandRecorded, runKey } = await import("../web/pipeline.js"); // store.js touches indexedDB only when called
  const rec = JSON.parse(fs.readFileSync(new URL("../web/recorded.json", import.meta.url), "utf8"));
  const ex = expandRecorded(rec, corpus);
  assert.deepEqual(Object.keys(ex.runs), Object.keys(rec.runs), "every recorded run still matches the corpus chunking");
  for (const run of Object.values(ex.runs)) {
    assert.equal(run.key, runKey({ modelVariant: run.model.variant, presetId: run.presetId, rag: run.rag }), `${run.key}: question wording changed since recording; re-record (see README)`);
    assert.equal(run.records.length, corpus.claims.length);
    for (const r of run.records) {
      const c = corpus.claims.find((x) => x.id === r.id);
      assert.equal(r.gold, c.label); assert.equal(r.evidenceHit, evidenceRetrieved(r.passages, c.evidence));
    }
  }
  const m = evaluate(ex.runs["q8e8|choice2|3e873b77|s2o0|k3|document"].records, { cutoff: 0.5, threshold: 0.1 });
  assert.equal(m.accuracy, 54 / 72);
});
