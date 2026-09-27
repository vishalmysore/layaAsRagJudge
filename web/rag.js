// Retrieval primitives: sentence-aware chunking and brute-force cosine top-k. Pure functions, no DOM or model,
// so they run unchanged in Node for the unit tests.

export const RAG_DEFAULTS = { sentencesPerChunk: 2, overlap: 0, k: 3, scope: "document" };

// Tokens that end in a period but do not end a sentence.
const ABBREV = new Set(["dr", "mr", "mrs", "ms", "st", "no", "vs", "etc", "e.g", "i.e", "approx", "inc", "ltd", "co", "prof", "jr", "sr"]);

/** Split prose into sentences. Handles decimals ("6.5"), common abbreviations ("Dr. Lindqvist") and quotes. */
export function splitSentences(text) {
  const src = String(text || "").replace(/\s+/g, " ").trim();
  if (!src) return [];
  const out = [];
  let start = 0;
  const re = /[.!?]["'”’)]*(?=\s+["'“‘(]?[A-Z0-9])/g;
  let m;
  while ((m = re.exec(src))) {
    const end = m.index + m[0].length;
    const before = src.slice(start, m.index);
    const lastWord = (before.match(/(?:^|\s)(\S+)$/) || [, ""])[1];
    if (m[0][0] === "." && ABBREV.has(lastWord.toLowerCase())) continue;
    if (m[0][0] === "." && /^[A-Z]$/.test(lastWord)) continue; // initials: "J. Smith"
    out.push(src.slice(start, end).trim());
    start = end;
  }
  const tail = src.slice(start).trim();
  if (tail) out.push(tail);
  return out;
}

/**
 * Split a document into passages of `sentencesPerChunk` sentences, sliding by (sentencesPerChunk - overlap).
 * @returns [{ index, text, firstSentence, sentences }]
 */
export function chunkText(text, { sentencesPerChunk = RAG_DEFAULTS.sentencesPerChunk, overlap = RAG_DEFAULTS.overlap } = {}) {
  const sents = splitSentences(text);
  const size = Math.max(1, sentencesPerChunk | 0);
  const step = Math.max(1, size - Math.max(0, Math.min(overlap | 0, size - 1)));
  const chunks = [];
  for (let i = 0; i < sents.length; i += step) {
    const part = sents.slice(i, i + size);
    chunks.push({ index: chunks.length, text: part.join(" "), firstSentence: i, sentences: part.length });
    if (i + size >= sents.length) break;
  }
  return chunks;
}

/** A stable key for a chunking configuration (part of every stored passage's key). */
export const chunkConfigKey = ({ sentencesPerChunk, overlap }) => `s${sentencesPerChunk}o${overlap || 0}`;

export function cosine(a, b) {
  let dot = 0, na = 0, nb = 0;
  const n = Math.min(a.length, b.length);
  for (let i = 0; i < n; i++) { dot += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i]; }
  return na && nb ? dot / Math.sqrt(na * nb) : 0;
}

/** Brute-force scan: score every item's `vec` against `query`, return the best k as [{ item, score }], best first. */
export function topK(query, items, k = RAG_DEFAULTS.k) {
  return items
    .map((item) => ({ item, score: cosine(query, item.vec) }))
    .sort((a, b) => b.score - a.score)
    .slice(0, Math.max(1, k));
}

const STOP = new Set("a an and are as at be but by can could did do does for from had has have how i if in is it its many much of on or our so that the their them then there these they this to was we were what when where which who why will with would you your".split(" "));
export const terms = (s) => (String(s).toLowerCase().match(/[a-z0-9]+(?:'[a-z]+)?/g) || []).filter((t) => !STOP.has(t));

/** BM25 keyword scores of `query` against each text (k1 = 1.2, b = 0.75). Exact terms and numbers count here. */
export function bm25(query, texts, { k1 = 1.2, b = 0.75 } = {}) {
  const docs = texts.map(terms);
  const N = docs.length, avg = docs.reduce((a, d) => a + d.length, 0) / Math.max(1, N);
  const df = new Map();
  for (const d of docs) for (const t of new Set(d)) df.set(t, (df.get(t) || 0) + 1);
  const q = [...new Set(terms(query))];
  return docs.map((d) => {
    const tf = new Map(); for (const t of d) tf.set(t, (tf.get(t) || 0) + 1);
    let s = 0;
    for (const t of q) {
      const f = tf.get(t); if (!f) continue;
      const idf = Math.log(1 + (N - df.get(t) + 0.5) / (df.get(t) + 0.5));
      s += idf * (f * (k1 + 1)) / (f + k1 * (1 - b + b * d.length / (avg || 1)));
    }
    return s;
  });
}

/**
 * Hybrid ranking: cosine and BM25, each min-max normalized over the candidates, mixed with weight `w` on cosine.
 * @returns [{ item, score, cosine, keyword }] best first
 */
export function hybridTopK(query, qvec, items, k, w = 0.6) {
  const cos = items.map((it) => cosine(qvec, it.vec));
  const kw = bm25(query, items.map((it) => it.text));
  const norm = (xs) => { const lo = Math.min(...xs), hi = Math.max(...xs); return xs.map((x) => (hi > lo ? (x - lo) / (hi - lo) : 0)); };
  const nc = norm(cos), nk = norm(kw);
  return items.map((item, i) => ({ item, score: w * nc[i] + (1 - w) * nk[i], cosine: cos[i], keyword: kw[i] }))
    .sort((a, b) => b.score - a.score).slice(0, Math.max(1, k));
}

/** True if any retrieved passage contains the gold evidence span (a verbatim substring of the document). */
export function evidenceRetrieved(passages, evidence) {
  if (!evidence) return null;
  const norm = (s) => s.replace(/\s+/g, " ").toLowerCase();
  const e = norm(evidence);
  return passages.some((p) => norm(p.text).includes(e));
}
