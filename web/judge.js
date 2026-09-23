// The Laya "judge": the claim plus its retrieved passages become the state of ONE typed question, and the answer is
// reduced to p(supported). Several phrasings ("presets") are available because this checkpoint's probabilities are
// sensitive to wording; the evaluation page measures each against the gold labels.

export const LABELS = ["supported", "not_supported"];

export const PRESETS = {
  choice2: {
    name: "Two options: supported / not supported",
    question: {
      type: "choice",
      instructions: "Does the evidence support the claim?",
      criteria: {
        supported: "The evidence states the claim or clearly implies it",
        not_supported: "The evidence contradicts the claim or does not mention it",
      },
    },
    pSupported: (a) => a.probabilities.supported,
  },
  choice3: {
    name: "Three options: supported / contradicted / not mentioned",
    question: {
      type: "choice",
      instructions: "What does the evidence say about the claim?",
      criteria: {
        supported: "The evidence states the claim or clearly implies it",
        contradicted: "The evidence says something different from the claim",
        not_mentioned: "The evidence does not say whether the claim is true",
      },
    },
    pSupported: (a) => a.probabilities.supported,
  },
  noul: {
    name: "Yes / no: the evidence supports the claim",
    question: { type: "noul", instructions: "The evidence fully supports the claim" },
    pSupported: (a) => a.noul,
  },
};
export const DEFAULT_PRESET = "choice2";

// Default confidence bar for acting automatically. This checkpoint's probabilities are compressed (a verdict of
// p = 0.83 has confidence 0.34), so the 0.90 used elsewhere would hold every claim for review. Measured on the
// sample with the two-option question: confidence >= 0.10 keeps ~44% of claims at ~84% accuracy (see README).
export const DEFAULT_THRESHOLD = 0.1;

/** The state Laya sees. Kept as a plain object; laya-core serializes it the way the Python original does. */
export function buildState(claim, passages) {
  return { claim: String(claim).trim(), evidence: passages.map((p) => (typeof p === "string" ? p : p.text)) };
}

/** 1 - normalized binary entropy of p (the layaForWeb confidence convention, applied to the supported split). */
export function binaryConfidence(p) {
  const q = Math.min(Math.max(p, 1e-12), 1 - 1e-12);
  const h = -(q * Math.log(q) + (1 - q) * Math.log(1 - q));
  return Math.min(Math.max(1 - h / Math.log(2), 0), 1);
}

/** Reduce a Laya answer to { pSupported, confidence, probabilities }. */
export function interpret(answer, presetId = DEFAULT_PRESET) {
  const preset = PRESETS[presetId];
  const p = preset.pSupported(answer);
  const probabilities = answer.type === "noul" ? { supported: p, not_supported: 1 - p } : answer.probabilities;
  return { pSupported: p, confidence: binaryConfidence(p), probabilities };
}

export const verdictOf = (pSupported, cutoff = 0.5) => (pSupported >= cutoff ? "supported" : "not_supported");

/**
 * Confidence gating, same idea as the workflows page: act automatically only when the judge is sure.
 *   AUTO  = confident "supported"      -> answer can be shown with its citation
 *   BLOCK = confident "not supported"  -> answer is withheld / regenerated
 *   HOLD  = below the threshold        -> flagged for a person to review
 */
export function gate(pSupported, confidence, { cutoff = 0.5, threshold = 0.5 } = {}) {
  if (confidence < threshold) return "HOLD";
  return verdictOf(pSupported, cutoff) === "supported" ? "AUTO" : "BLOCK";
}

/** Fingerprint of a question, so cached/recorded results are only reused if the wording is unchanged. */
export function questionKey(q) {
  const s = JSON.stringify(q);
  let h = 2166136261;
  for (let i = 0; i < s.length; i++) { h ^= s.charCodeAt(i); h = Math.imul(h, 16777619); }
  return (h >>> 0).toString(16);
}
