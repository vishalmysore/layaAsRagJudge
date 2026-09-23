// Sentence embeddings for retrieval: all-MiniLM-L6-v2 (384-d, int8 ONNX, ~23 MB), run by the SAME ONNX Runtime Web
// and Tokenizers.js that run Laya, so the page ships one runtime. Mean pooling over the attention mask, then L2
// normalization, exactly like sentence-transformers. Files are pinned to a commit and kept in Cache Storage, so after
// the first visit the embedder loads offline.
import * as ort from "./vendor/ort.min.mjs";
import { Tokenizer } from "./vendor/tokenizers.min.mjs";

export const EMBED_MODEL = {
  id: "Xenova/all-MiniLM-L6-v2",
  page: "https://huggingface.co/Xenova/all-MiniLM-L6-v2",
  base: "https://huggingface.co/Xenova/all-MiniLM-L6-v2/resolve/751bff37182d3f1213fa05d7196b954e230abad9/",
  onnx: "onnx/model_quantized.onnx",
  dim: 384,
  maxTokens: 256,
};
const CACHE = "rag-embedder-minilm-751bff3";

async function cachedFetch(url, onProgress) {
  let cache = null;
  try { cache = await caches.open(CACHE); const hit = await cache.match(url); if (hit) return { bytes: new Uint8Array(await hit.arrayBuffer()), cached: true }; } catch { /* no Cache Storage */ }
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url.split("/").pop()}: HTTP ${res.status}`);
  const total = Number(res.headers.get("content-length")) || 0;
  const reader = res.body.getReader(); const parts = []; let got = 0;
  for (;;) { const { done, value } = await reader.read(); if (done) break; parts.push(value); got += value.length; onProgress?.(got, total); }
  const bytes = new Uint8Array(got); let o = 0; for (const p of parts) { bytes.set(p, o); o += p.length; }
  try { if (cache) await cache.put(url, new Response(bytes)); } catch { /* quota: fine */ }
  return { bytes, cached: false };
}
const json = (b) => JSON.parse(new TextDecoder().decode(b));

export async function embedderCached() {
  try { const c = await caches.open(CACHE); return !!(await c.match(EMBED_MODEL.base + EMBED_MODEL.onnx)); } catch { return false; }
}

/** @returns { embed(texts: string[]) => Promise<Float32Array[]>, dim, id, info } */
export async function loadEmbedder({ onStatus = () => {}, onProgress = () => {} } = {}) {
  ort.env.wasm.wasmPaths = new URL("./vendor/", import.meta.url).href;
  ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 2) : 1;
  const t0 = performance.now();
  onStatus("Loading the embedding model…");
  const [tj, tc] = await Promise.all(["tokenizer.json", "tokenizer_config.json"].map((f) => cachedFetch(EMBED_MODEL.base + f).then((r) => json(r.bytes))));
  const tokenizer = new Tokenizer(tj, tc);
  const model = await cachedFetch(EMBED_MODEL.base + EMBED_MODEL.onnx, (got, total) => {
    onProgress(total ? got / total : null);
    onStatus(`Downloading the embedding model: ${(got / 1048576).toFixed(1)} / ${(total / 1048576).toFixed(1)} MB`);
  });
  const session = await ort.InferenceSession.create(model.bytes, { executionProviders: ["wasm"], graphOptimizationLevel: "all" });
  const needsTypes = session.inputNames.includes("token_type_ids");

  const encode = (text) => {
    let ids = tokenizer.encode(String(text)).ids;
    if (ids.length > EMBED_MODEL.maxTokens) ids = [...ids.slice(0, EMBED_MODEL.maxTokens - 1), ids.at(-1)]; // keep [SEP]
    return ids;
  };

  // One ORT session runs one inference at a time; queue calls like model.js does for Laya.
  let queue = Promise.resolve();
  async function run(texts) {
    const enc = texts.map(encode);
    const n = enc.length, L = Math.max(...enc.map((e) => e.length));
    const ids = new BigInt64Array(n * L), mask = new BigInt64Array(n * L);
    enc.forEach((e, i) => e.forEach((t, j) => { ids[i * L + j] = BigInt(t); mask[i * L + j] = 1n; }));
    const feeds = { input_ids: new ort.Tensor("int64", ids, [n, L]), attention_mask: new ort.Tensor("int64", mask, [n, L]) };
    if (needsTypes) feeds.token_type_ids = new ort.Tensor("int64", new BigInt64Array(n * L), [n, L]);
    const out = await session.run(feeds);
    const h = (out.last_hidden_state || out[session.outputNames[0]]).data; // [n, L, dim]
    const D = EMBED_MODEL.dim;
    return enc.map((e, i) => {
      const v = new Float32Array(D);
      for (let j = 0; j < e.length; j++) for (let d = 0; d < D; d++) v[d] += h[(i * L + j) * D + d];
      let norm = 0; for (let d = 0; d < D; d++) { v[d] /= e.length; norm += v[d] * v[d]; }
      norm = Math.sqrt(norm) || 1; for (let d = 0; d < D; d++) v[d] /= norm;
      return v;
    });
  }
  /** Embed texts in small batches (padding waste stays low; the page stays responsive between batches). */
  async function embed(texts, batch = 8) {
    const p = queue.then(async () => {
      const out = [];
      for (let i = 0; i < texts.length; i += batch) out.push(...await run(texts.slice(i, i + batch)));
      return out;
    });
    queue = p.catch(() => {});
    return p;
  }
  await embed(["warm up"]);
  return { embed, dim: EMBED_MODEL.dim, id: EMBED_MODEL.id, info: { loadMs: performance.now() - t0, fromCache: model.cached } };
}
