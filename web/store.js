// Browser-local vector store on IndexedDB. Two object stores, as in the requirements:
//   passages: { key, cfg, dataset, docId, chunk, text, vec: Float32Array }   index "byDoc" on [cfg, docId], "byCfg" on cfg
//   runs:     { key, ... }  cached evaluation runs keyed by (model build, judge preset, chunking, k, scope)
// plus "claims" (claim embeddings, so re-running an evaluation never re-embeds) and "meta" (the corpus JSON). If IndexedDB is unavailable
// (private window, blocked storage) everything falls back to an in-memory map for the life of the tab.

const DB_NAME = "laya-rag-judge", VERSION = 1;
const STORES = ["passages", "runs", "claims", "meta"];
let dbp = null;
const mem = Object.fromEntries(STORES.map((s) => [s, new Map()]));

function open() {
  if (dbp) return dbp;
  dbp = new Promise((resolve) => {
    let req;
    try { req = indexedDB.open(DB_NAME, VERSION); } catch { return resolve(null); }
    req.onupgradeneeded = () => {
      const db = req.result;
      const p = db.createObjectStore("passages", { keyPath: "key" });
      p.createIndex("byDoc", ["cfg", "docId"]);
      p.createIndex("byCfg", "cfg");
      db.createObjectStore("runs", { keyPath: "key" });
      db.createObjectStore("claims", { keyPath: "key" });
      db.createObjectStore("meta", { keyPath: "key" });
    };
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => { console.warn("IndexedDB unavailable, using memory:", req.error); resolve(null); };
    req.onblocked = () => resolve(null);
  });
  return dbp;
}

const done = (req) => new Promise((res, rej) => { req.onsuccess = () => res(req.result); req.onerror = () => rej(req.error); });
const txDone = (tx) => new Promise((res, rej) => { tx.oncomplete = () => res(); tx.onerror = () => rej(tx.error); tx.onabort = () => rej(tx.error); });

export async function persistent() { return !!(await open()); }

export async function putMany(store, rows) {
  const db = await open();
  if (!db) { for (const r of rows) mem[store].set(r.key, r); return; }
  const tx = db.transaction(store, "readwrite");
  for (const r of rows) tx.objectStore(store).put(r);
  await txDone(tx);
}
export async function get(store, key) {
  const db = await open();
  if (!db) return mem[store].get(key) ?? null;
  return (await done(db.transaction(store).objectStore(store).get(key))) ?? null;
}
export async function getAll(store) {
  const db = await open();
  if (!db) return [...mem[store].values()];
  return done(db.transaction(store).objectStore(store).getAll());
}
/** Passages for one chunking config, optionally one document. */
export async function passages(cfg, docId = null) {
  const db = await open();
  if (!db) return [...mem.passages.values()].filter((p) => p.cfg === cfg && (docId == null || p.docId === docId)).sort((a, b) => a.chunk - b.chunk);
  const os = db.transaction("passages").objectStore("passages");
  const rows = docId == null ? await done(os.index("byCfg").getAll(cfg)) : await done(os.index("byDoc").getAll([cfg, docId]));
  return rows.sort((a, b) => a.chunk - b.chunk);
}
export async function countPassages(cfg) {
  const db = await open();
  if (!db) return [...mem.passages.values()].filter((p) => p.cfg === cfg).length;
  return done(db.transaction("passages").objectStore("passages").index("byCfg").count(cfg));
}
export async function clearAll() {
  const db = await open();
  for (const s of Object.keys(mem)) mem[s].clear();
  if (!db) return;
  const tx = db.transaction(STORES, "readwrite");
  for (const s of STORES) tx.objectStore(s).clear();
  await txDone(tx);
}
