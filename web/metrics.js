// Scoring an evaluation run against the gold labels. Pure functions over per-claim records:
//   { id, dataset, kind, gold: "supported"|"not_supported", pSupported, confidence, layaMs, embedMs, evidenceHit }
// Everything here is recomputed when a slider moves, so the model never has to run again.
import { verdictOf, gate } from "./judge.js";

// Fine steps at the low end: this checkpoint's probabilities are compressed, so entropy confidence rarely passes 0.4.
export const COVERAGE_THRESHOLDS = [0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.9];

const ratio = (a, b) => (b ? a / b : null);

export function percentile(values, q) {
  const v = values.filter((x) => Number.isFinite(x)).sort((a, b) => a - b);
  if (!v.length) return null;
  const pos = (v.length - 1) * q, lo = Math.floor(pos), hi = Math.ceil(pos);
  return v[lo] + (v[hi] - v[lo]) * (pos - lo);
}

/** Confusion counts with "supported" as the positive class. */
export function confusion(records, cutoff) {
  const c = { tp: 0, fp: 0, tn: 0, fn: 0 };
  for (const r of records) {
    const pred = verdictOf(r.pSupported, cutoff) === "supported", gold = r.gold === "supported";
    if (pred && gold) c.tp++; else if (pred && !gold) c.fp++; else if (!pred && !gold) c.tn++; else c.fn++;
  }
  return c;
}

function scores(c) {
  const n = c.tp + c.fp + c.tn + c.fn;
  const tpr = ratio(c.tp, c.tp + c.fn), tnr = ratio(c.tn, c.tn + c.fp);
  return {
    n,
    accuracy: ratio(c.tp + c.tn, n),
    balancedAccuracy: tpr == null || tnr == null ? (tpr ?? tnr) : (tpr + tnr) / 2,
    // share of genuinely unsupported claims the judge let through as "supported"
    falseVerificationRate: ratio(c.fp, c.fp + c.tn),
    precision: ratio(c.tp, c.tp + c.fp),
    recall: tpr,
  };
}

/** Area under the ROC curve of pSupported (threshold-free: how well the score separates the two classes). */
export function auroc(records) {
  const pos = records.filter((r) => r.gold === "supported").map((r) => r.pSupported);
  const neg = records.filter((r) => r.gold !== "supported").map((r) => r.pSupported);
  if (!pos.length || !neg.length) return null;
  let s = 0;
  for (const p of pos) for (const q of neg) s += p > q ? 1 : p === q ? 0.5 : 0;
  return s / (pos.length * neg.length);
}

/** The cutoff on pSupported that maximizes balanced accuracy (a calibration hint, fit on the same data). */
export function bestCutoff(records) {
  const cands = [...new Set(records.map((r) => r.pSupported))].sort((a, b) => a - b);
  let best = { cutoff: 0.5, balancedAccuracy: -1 };
  for (const c of cands) {
    const b = scores(confusion(records, c)).balancedAccuracy ?? -1;
    if (b > best.balancedAccuracy + 1e-12) best = { cutoff: c, balancedAccuracy: b };
  }
  return best;
}

export function groupBy(records, key, cutoff) {
  const groups = {};
  for (const r of records) (groups[r[key] ?? "?"] ||= []).push(r);
  return Object.entries(groups).map(([name, rs]) => ({ name, ...scores(confusion(rs, cutoff)) }));
}

/** At each confidence threshold: how many claims clear the bar (coverage) and how accurate those are. */
export function coverageTable(records, cutoff, thresholds = COVERAGE_THRESHOLDS) {
  return thresholds.map((t) => {
    const kept = records.filter((r) => r.confidence >= t);
    const s = scores(confusion(kept, cutoff));
    return { threshold: t, kept: kept.length, coverage: ratio(kept.length, records.length), accuracy: s.accuracy, falseVerificationRate: s.falseVerificationRate };
  });
}

export function evaluate(records, { cutoff = 0.5, threshold = 0.5 } = {}) {
  const overall = scores(confusion(records, cutoff));
  const gates = { AUTO: 0, BLOCK: 0, HOLD: 0 };
  let gatedWrong = 0, gatedN = 0;
  for (const r of records) {
    const g = gate(r.pSupported, r.confidence, { cutoff, threshold });
    gates[g]++;
    if (g !== "HOLD") { gatedN++; if (verdictOf(r.pSupported, cutoff) !== r.gold) gatedWrong++; }
  }
  const hits = records.filter((r) => r.evidenceHit != null);
  return {
    ...overall,
    confusion: confusion(records, cutoff),
    auroc: auroc(records),
    bestCutoff: bestCutoff(records),
    byDataset: groupBy(records, "dataset", cutoff),
    byKind: groupBy(records, "kind", cutoff),
    coverage: coverageTable(records, cutoff),
    gates, automatedAccuracy: ratio(gatedN - gatedWrong, gatedN), automatedShare: ratio(gatedN, records.length),
    evidenceRecall: ratio(hits.filter((r) => r.evidenceHit).length, hits.length),
    latency: {
      median: percentile(records.map((r) => r.layaMs), 0.5), p95: percentile(records.map((r) => r.layaMs), 0.95),
      embedMedian: percentile(records.map((r) => r.embedMs), 0.5),
    },
  };
}
