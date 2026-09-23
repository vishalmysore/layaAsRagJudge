// Loads the Laya typed-decisions ONNX model in the browser. Adapted from layaForWeb's web/app.js loadModel().
//
// The weights live on Hugging Face (VishalMysore/layaForWebTrained): a small ONNX graph plus the external-data
// file split into 24 MiB parts listed in manifest.json. Parts are cached in Cache Storage keyed by the file hash,
// so a repeat visit (or a visit to the layaForWeb demo with the same build) skips the download.
// Override the location with ?modelBase=https://.../ (the host must send CORS headers).
import * as ort from "./vendor/ort.min.mjs";
import { Tokenizer } from "./vendor/tokenizers.min.mjs";
import { Laya } from "./laya-core.js";

export const DEFAULT_MODEL_BASE = "https://huggingface.co/VishalMysore/layaForWebTrained/resolve/main/";
export const MODEL_PAGE = "https://huggingface.co/VishalMysore/layaForWebTrained";

export function modelBase() {
  const q = new URLSearchParams(location.search).get("modelBase");
  return (q || DEFAULT_MODEL_BASE).replace(/\/?$/, "/");
}

export async function fetchManifest(base = modelBase()) {
  const r = await fetch(base + "manifest.json");
  if (!r.ok) throw new Error(`manifest.json: HTTP ${r.status}`);
  return r.json();
}

async function fetchBytes(url, onProgress) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url.split("/").pop()}: HTTP ${res.status}`);
  const total = Number(res.headers.get("content-length")) || 0;
  const reader = res.body.getReader();
  const chunks = []; let got = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value); got += value.length; onProgress?.(got, total);
  }
  const out = new Uint8Array(got); let o = 0;
  for (const c of chunks) { out.set(c, o); o += c.length; }
  return out;
}

async function openCache(entry) {
  try { return await caches.open("laya-" + entry.sha256.slice(0, 16)); } catch { return null; }
}

/** How many of this build's parts are already in the browser cache (0..parts.length). */
export async function cachedParts(base, entry) {
  const cache = await openCache(entry);
  if (!cache) return 0;
  let n = 0;
  for (const p of entry.parts) { try { if (await cache.match(base + p)) n++; } catch { /* ignore */ } }
  return n;
}

async function fetchParts(base, entry, onProgress) {
  const cache = await openCache(entry);
  const out = new Uint8Array(entry.size); let off = 0, fromCache = 0;
  for (const part of entry.parts) {
    const url = base + part;
    let bytes = null;
    try { const hit = cache && await cache.match(url); if (hit) { bytes = new Uint8Array(await hit.arrayBuffer()); fromCache++; } } catch { /* ignore */ }
    if (!bytes) {
      bytes = await fetchBytes(url, (got) => onProgress(off + got, entry.size));
      try { if (cache) await cache.put(url, new Response(bytes)); } catch { /* quota exceeded: fine */ }
    }
    if (off + bytes.length > entry.size) throw new Error("weights are larger than the manifest says");
    out.set(bytes, off); off += bytes.length; onProgress(off, entry.size);
  }
  if (off !== entry.size) throw new Error(`weights incomplete: got ${off} of ${entry.size} bytes`);
  return { data: out, fromCache };
}

export async function hasWebGPU() {
  try { return !!navigator.gpu && !!(await navigator.gpu.requestAdapter()); } catch { return false; }
}

/**
 * @param opts { base, manifest, variant: "q8e8"|"q4e8", backend: "auto"|"wasm"|"webgpu", onStatus(msg), onProgress(fraction|null) }
 * @returns { laya, backend, info }
 */
export async function loadLaya(opts) {
  const base = opts.base || modelBase();
  const manifest = opts.manifest || await fetchManifest(base);
  const key = opts.variant || "q8e8";
  const v = manifest.variants[key];
  if (!v) throw new Error(`unknown build "${key}"`);
  const status = opts.onStatus || (() => {}), progress = opts.onProgress || (() => {});

  // ONNX Runtime's WebGPU MatMulNBits kernel supports 2- and 4-bit weights only, so the int8 build is WASM-only.
  const gpu = await hasWebGPU();
  const int8 = key === "q8e8";
  let backend = opts.backend || "auto";
  if (backend === "auto") backend = gpu && !int8 ? "webgpu" : "wasm";
  if (backend === "webgpu" && int8) throw new Error("The int8 build cannot run on WebGPU. Pick the int4 build, or WASM.");
  if (backend === "webgpu" && !gpu) throw new Error("WebGPU is not available in this browser. Use WASM.");

  ort.env.wasm.wasmPaths = new URL("./vendor/", import.meta.url).href;
  ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 2) : 1;

  status("Loading tokenizer and config…"); progress(0);
  const [tj, tc, cfg] = await Promise.all(["tokenizer.json", "tokenizer_config.json", "rl_agent_config.json"].map((f) =>
    fetch(base + f).then((r) => { if (!r.ok) throw new Error(`${f}: HTTP ${r.status}`); return r.json(); })));
  const tokenizer = new Tokenizer(tj, tc);

  const t0 = performance.now();
  const graph = await fetchBytes(base + v.onnx);
  const { data, fromCache } = await fetchParts(base, v.data, (got, total) => {
    progress(total ? got / total : null);
    status(`Downloading weights: ${(got / 1048576).toFixed(0)} / ${(total / 1048576).toFixed(0)} MB`);
  });
  const dlMs = performance.now() - t0;

  status("Creating inference session (first time can take a while)…"); progress(null);
  const t1 = performance.now();
  const session = await ort.InferenceSession.create(graph, {
    executionProviders: backend === "webgpu" ? ["webgpu", "wasm"] : ["wasm"],
    graphOptimizationLevel: "all",
    externalData: [{ path: v.data.name, data }],
  });
  const initMs = performance.now() - t1;
  const laya = new Laya(ort, session, tokenizer, cfg);
  // An ONNX Runtime session can run only one inference at a time ("Session already started"), so queue calls.
  const run = laya.systemOne.bind(laya);
  let queue = Promise.resolve();
  laya.systemOne = (state, questions) => { const p = queue.then(() => run(state, questions)); queue = p.catch(() => {}); return p; };
  status("Warming up…");
  const w0 = performance.now();
  await laya.systemOne("warm up", { w: { type: "noul", instructions: "This is a warm-up call" } });
  return {
    laya, backend,
    info: {
      variant: key, label: v.label, source: manifest.source, model: cfg.model_name, threads: ort.env.wasm.numThreads,
      crossOriginIsolated: self.crossOriginIsolated, downloadMs: dlMs, initMs, warmMs: performance.now() - w0,
      fromCache, parts: v.data.parts.length, maxLen: cfg.max_len, headMaxLen: cfg.head_max_len,
    },
  };
}
