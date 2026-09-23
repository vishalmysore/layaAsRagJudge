// The RAG claim-verification pipeline, shared by both pages:
//   load corpus -> chunk each document -> embed each chunk into IndexedDB -> embed the claim -> cosine top-k
//   -> one typed Laya question (claim + retrieved passages) -> p(supported) -> gate.
import * as db from "./store.js";
import { chunkText, chunkConfigKey, topK, evidenceRetrieved, RAG_DEFAULTS } from "./rag.js";
import { PRESETS, buildState, interpret, questionKey } from "./judge.js";
import { buildSequence, toInternal } from "./laya-core.js";

const CORPUS_URL = "./corpus.json";

/** Fetch the corpus (network first so edits show up; the IndexedDB copy makes it work offline). */
export async function loadCorpus() {
  try {
    const r = await fetch(CORPUS_URL, { cache: "no-cache" });
    if (!r.ok) throw new Error(`corpus.json: HTTP ${r.status}`);
    const corpus = await r.json();
    db.putMany("meta", [{ key: "corpus", corpus }]).catch(() => {});
    return corpus;
  } catch (e) {
    const hit = await db.get("meta", "corpus");
    if (hit) return hit.corpus;
    throw e;
  }
}

/** Store key prefix for passages: which embedding model and which chunking produced them. */
export const passageCfg = (embedder, rag) => `${embedder.id}|${chunkConfigKey(rag)}`;

/**
 * Make sure every document is chunked and embedded for this configuration. Documents already in IndexedDB are
 * skipped, so after the first visit this is a no-op.
 * @returns { cfg, chunks, embedded, ms }
 */
export async function indexCorpus(corpus, embedder, rag, onProgress = () => {}) {
  const cfg = passageCfg(embedder, rag);
  const t0 = performance.now();
  const docs = Object.entries(corpus.documents);
  let chunks = 0, embedded = 0;
  for (let i = 0; i < docs.length; i++) {
    const [docId, doc] = docs[i];
    const have = await db.passages(cfg, docId);
    const parts = chunkText(doc.text, rag);
    chunks += parts.length;
    if (have.length !== parts.length || have.some((p, j) => p.text !== parts[j].text)) {
      const vecs = await embedder.embed(parts.map((p) => p.text));
      await db.putMany("passages", parts.map((p, j) => ({
        key: `${cfg}|${doc.dataset}|${docId}|${p.index}`, cfg, dataset: doc.dataset, docId, chunk: p.index, text: p.text, vec: vecs[j],
      })));
      embedded += parts.length;
    }
    onProgress(i + 1, docs.length);
  }
  return { cfg, chunks, embedded, ms: performance.now() - t0 };
}

/** Embed a claim, reusing the IndexedDB copy if this exact text was embedded before. */
export async function claimVector(embedder, text) {
  const key = `${embedder.id}|${text}`;
  const hit = await db.get("claims", key);
  if (hit) return { vec: hit.vec, ms: 0, cached: true };
  const t0 = performance.now();
  const [vec] = await embedder.embed([text]);
  const ms = performance.now() - t0;
  db.putMany("claims", [{ key, vec }]).catch(() => {});
  return { vec, ms, cached: false };
}

/**
 * Top-k passages for a claim. scope "document" scans only the claim's own document (the requirements' default);
 * "corpus" scans every passage, which makes retrieval part of the test.
 */
export async function retrieve(embedder, claim, { rag = RAG_DEFAULTS, docId = null, pool = null } = {}) {
  const { vec, ms } = await claimVector(embedder, claim);
  const t0 = performance.now();
  const items = pool || await db.passages(passageCfg(embedder, rag), rag.scope === "corpus" ? null : docId);
  const ranked = topK(vec, items, items.length).map(({ item, score }) => ({ docId: item.docId, chunk: item.chunk, text: item.text, score }));
  return { passages: ranked.slice(0, rag.k), ranked, candidates: items.length, embedMs: ms, searchMs: performance.now() - t0 };
}

/** Ask Laya one typed question: does the evidence support the claim? */
export async function judge(laya, claim, passages, presetId) {
  const preset = PRESETS[presetId];
  const state = buildState(claim, passages);
  const res = await laya.systemOne(state, { verdict: preset.question });
  const answer = res.answers.verdict;
  return { ...interpret(answer, presetId), answer, state, question: preset.question, layaMs: res.latency_ms, tokens: res.usage?.input_tokens, input: modelInput(laya, state, preset.question) };
}

/**
 * The exact token sequence Laya reads for this question, decoded back to text: the same buildSequence() call that
 * systemOne() makes, so special tokens ([CLS] [SEP] and one [MASK] per option) and any truncation are visible.
 */
export function modelInput(laya, state, question) {
  const cfg = laya.cfg;
  const { ids, markers } = buildSequence(laya.tok, laya.sp, state, toInternal(question), cfg.max_len ?? 512, cfg.head_max_len ?? 192);
  const tokens = ids.map((id) => laya.tok.decode([id], { skip_special_tokens: false }));
  return { ids, markers, text: laya.tok.decode(ids, { skip_special_tokens: false }), tokens, maxLen: cfg.max_len ?? 512, truncated: ids.length >= (cfg.max_len ?? 512) };
}

/** Embed and index a pasted evidence text in memory (not stored), for the "your own evidence" mode. */
export async function adHocPool(embedder, text, rag) {
  const parts = chunkText(text, rag);
  const vecs = await embedder.embed(parts.map((p) => p.text));
  return parts.map((p, j) => ({ docId: "pasted", chunk: p.index, text: p.text, vec: vecs[j] }));
}

/** Everything that identifies an evaluation run's model outputs (NOT the cutoff/threshold, which are applied later). */
export function runKey({ modelVariant, presetId, rag }) {
  return [modelVariant, presetId, questionKey(PRESETS[presetId].question), chunkConfigKey(rag), `k${rag.k}`, rag.scope].join("|");
}

/**
 * recorded.json stores passages as { docId, chunk, score } and no claim text, to stay small; rebuild the text from
 * the corpus (chunking is deterministic). Runs whose documents changed since recording are dropped.
 */
export function expandRecorded(recorded, corpus) {
  if (!recorded) return null;
  const chunks = new Map();
  const textOf = (docId, chunk, rag) => {
    const k = `${docId}|${chunkConfigKey(rag)}`;
    if (!chunks.has(k)) chunks.set(k, corpus.documents[docId] ? chunkText(corpus.documents[docId].text, rag) : []);
    return chunks.get(k)[chunk]?.text;
  };
  const byId = new Map(corpus.claims.map((c) => [c.id, c]));
  const runs = {};
  for (const [key, run] of Object.entries(recorded.runs)) {
    const records = run.records.map((r) => ({ ...r, claim: byId.get(r.id)?.claim, passages: r.passages.map((p) => ({ ...p, text: textOf(p.docId, p.chunk, run.rag) })) }));
    if (records.every((r) => r.claim && r.passages.every((p) => p.text))) runs[key] = { ...run, source: "recorded", records };
  }
  return { ...recorded, runs };
}

/** The compact form written to recorded.json (see expandRecorded). */
export function compactRun(run) {
  const r4 = (x) => Math.round(x * 1e4) / 1e4;
  return {
    ...run, source: "recorded",
    records: run.records.map((r) => ({
      id: r.id, dataset: r.dataset, docId: r.docId, kind: r.kind, gold: r.gold,
      passages: r.passages.map((p) => ({ docId: p.docId, chunk: p.chunk, score: r4(p.score) })),
      pSupported: r4(r.pSupported), confidence: r4(r.confidence), probabilities: Object.fromEntries(Object.entries(r.probabilities).map(([k, v]) => [k, r4(v)])),
      layaMs: Math.round(r.layaMs), embedMs: Math.round(r.embedMs), evidenceHit: r.evidenceHit,
    })),
  };
}

export { evidenceRetrieved };
