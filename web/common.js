// Shared by both pages: the model card (Laya + the embedding model), small DOM helpers and per-browser settings.
import { loadEmbedder, embedderCached, EMBED_MODEL } from "./embedder.js";

export const $ = (id) => document.getElementById(id);
export const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
export const pct = (p, d = 0) => (p == null ? "–" : `${(p * 100).toFixed(d)}%`);
export const num = (x, d = 2) => (x == null || !Number.isFinite(x) ? "–" : x.toFixed(d));
// Yield a macrotask so the page can paint and handle input. MessageChannel, not setTimeout: background tabs clamp
// setTimeout to >= 1 s, which would stall a batch run whenever the tab is not in front.
export const yieldToBrowser = () => new Promise((r) => { const ch = new MessageChannel(); ch.port1.onmessage = () => r(); ch.port2.postMessage(0); });

export const settings = {
  get(k, d) { try { const v = localStorage.getItem("lrj." + k); return v == null ? d : JSON.parse(v); } catch { return d; } },
  set(k, v) { try { localStorage.setItem("lrj." + k, JSON.stringify(v)); } catch { /* fine */ } },
};

export const M = { laya: null, embedder: null, model: null, manifest: null, loading: false, listeners: [] };
export const onModelsReady = (fn) => M.listeners.push(fn);

let modelMod = null;
const modelApi = () => (modelMod ||= import("./model.js"));

function setPill(kind, text) { const p = $("modelPill"); p.className = "modelpill " + kind; $("modelPillText").textContent = text; }
export function setStatus(msg, warn = false) { const s = $("status"); s.textContent = msg; s.classList.toggle("warn", warn); }
function setProgress(f) { $("progress").hidden = f == null; if (f != null) $("progressBar").style.width = (f * 100).toFixed(1) + "%"; }
export function pillIdle(recorded) { if (!M.laya) setPill(recorded ? "replay" : "", recorded ? "Recorded results · load the models to run your own" : "Models not loaded"); }

export async function initModelCard() {
  $("loadBtn").addEventListener("click", loadModels);
  $("embedLink").href = EMBED_MODEL.page; $("embedLink").textContent = EMBED_MODEL.id;
  const m = await modelApi();
  const base = m.modelBase();
  const link = $("modelLink"); link.href = base === m.DEFAULT_MODEL_BASE ? m.MODEL_PAGE : base; link.textContent = base === m.DEFAULT_MODEL_BASE ? "VishalMysore/layaForWebTrained" : base;
  try {
    M.manifest = await m.fetchManifest(base);
    const sel = $("variant"); sel.innerHTML = "";
    const saved = settings.get("variant", "q8e8");
    for (const [k, v] of Object.entries(M.manifest.variants)) {
      const cached = await m.cachedParts(base, v.data);
      const tag = cached === v.data.parts.length ? ", cached" : cached ? `, ${cached}/${v.data.parts.length} parts cached` : "";
      sel.add(new Option(`${k === "q8e8" ? "int8 (recommended)" : k === "q4e8" ? "int4 (smaller, WebGPU)" : v.label} · ${Math.round(v.data.size / 1048576)} MB${tag}`, k));
    }
    if (M.manifest.variants[saved]) sel.value = saved;
    const e = await embedderCached();
    $("embedState").textContent = e ? " (cached)" : " (about 23 MB)";
    $("loadBtn").disabled = false;
  } catch (e) {
    setStatus(`Could not read the model manifest from ${base} (${e.message}). Recorded results still work.`, true);
  }
}

export async function loadModels() {
  if (M.loading) return;
  M.loading = true; $("loadBtn").disabled = true; $("badges").innerHTML = "";
  setPill("busy", "Loading models…");
  settings.set("variant", $("variant").value);
  try {
    const m = await modelApi();
    if (!M.embedder) M.embedder = await loadEmbedder({ onStatus: setStatus, onProgress: setProgress });
    const { laya, backend, info } = await m.loadLaya({
      base: m.modelBase(), manifest: M.manifest, variant: $("variant").value, backend: $("backend").value,
      onStatus: setStatus, onProgress: setProgress,
    });
    M.laya = laya; M.model = { backend, ...info };
    setStatus(`Ready. Laya: download ${(info.downloadMs / 1000).toFixed(1)} s${info.fromCache ? ` (${info.fromCache}/${info.parts} parts from cache)` : ""}, session ${(info.initMs / 1000).toFixed(1)} s. Embedder ${(M.embedder.info.loadMs / 1000).toFixed(1)} s${M.embedder.info.fromCache ? " (cached)" : ""}.`);
    const badge = (t, ok) => { const b = document.createElement("span"); b.className = "badge" + (ok ? " ok" : ""); b.textContent = t; $("badges").appendChild(b); };
    badge(backend === "webgpu" ? "WebGPU" : `WASM · ${info.threads} thread${info.threads > 1 ? "s" : ""}`, true);
    badge(`judge: ${info.model || info.source} · ${info.variant}`); badge(`retriever: ${EMBED_MODEL.id} · ${EMBED_MODEL.dim}-d`);
    badge(`context ${info.maxLen} tokens`);
    setPill("ready", `Models ready · ${backend === "webgpu" ? "WebGPU" : "WASM"}`);
    $("loadBtn").textContent = "Reload models";
    for (const fn of M.listeners) fn();
  } catch (e) {
    console.error(e);
    setStatus("Could not load the models: " + (e?.message || e), true);
    pillIdle(false);
  } finally { M.loading = false; $("loadBtn").disabled = false; setProgress(null); }
}

/** Recorded results (produced by the same models) so the pages show something before anything is downloaded. */
export async function loadRecorded() {
  try { const r = await fetch("./recorded.json", { cache: "no-cache" }); if (r.ok) return await r.json(); } catch { /* optional */ }
  return null;
}

export function download(name, text, type = "application/json") {
  const url = URL.createObjectURL(new Blob([text], { type }));
  const a = Object.assign(document.createElement("a"), { href: url, download: name });
  document.body.appendChild(a); a.click(); a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

/** Probability bars (reuses the .bar styles). */
export function bars(probs, selected) {
  return `<div class="bars">${Object.entries(probs).map(([k, v]) => `<div class="bar${k === selected ? " sel" : ""}"><span class="n">${esc(k.replace(/_/g, " "))}</span><span class="tr"><span class="f" style="width:${(v * 100).toFixed(1)}%"></span></span><span class="v">${v.toFixed(2)}</span></div>`).join("")}</div>`;
}

export const GATE_TEXT = {
  AUTO: { cls: "auto", k: "AUTO · supported", t: "Evidence supports the claim", d: "Confident enough to show the answer with its citation." },
  BLOCK: { cls: "block", k: "BLOCK · not supported", t: "Evidence does not support the claim", d: "Confident enough to withhold or regenerate the answer." },
  HOLD: { cls: "human", k: "HOLD · review", t: "Judge is unsure", d: "Below the confidence threshold, so a person reviews it." },
};
