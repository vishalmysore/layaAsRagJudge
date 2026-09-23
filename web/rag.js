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

/** True if any retrieved passage contains the gold evidence span (a verbatim substring of the document). */
export function evidenceRetrieved(passages, evidence) {
  if (!evidence) return null;
  const norm = (s) => s.replace(/\s+/g, " ").toLowerCase();
  const e = norm(evidence);
  return passages.some((p) => norm(p.text).includes(e));
}
