// RAG router: one Laya choice question picks the knowledge base, a second checks whether the retrieved passages
// actually answer the question. Pure functions (no DOM, no model), unit-tested in Node.

/** The routing question: one option per route, described in plain words. */
export function routerQuestion(routes) {
  return {
    type: "choice",
    instructions: "Which knowledge base should answer this question?",
    criteria: Object.fromEntries(routes.map((r) => [r.id, r.description])),
  };
}

/** After retrieval: do these passages contain the answer? */
export const ANSWER_CHECK = {
  type: "choice",
  instructions: "Do the passages answer the question?",
  criteria: {
    answered: "The passages contain the answer to the question",
    not_answered: "The passages do not contain the answer",
  },
};

// Measured on the sample: among the three knowledge bases Laya picks the right one 23 times out of 24, but
// "no retrieval" vs "look it up" overlaps (knowledge questions get p(none) 0.28-0.54, small talk 0.48-0.75). So the
// router makes two separate decisions instead of one argmax:
//   1. skip retrieval only when p(none) is clearly high; otherwise search, and let the answer check (which rejects
//      small talk reliably) decide whether anything useful came back
//   2. choose the knowledge base among the KB options alone; fan out to the top two when the leader's share is low
export const DEFAULT_DECLINE_AT = 0.6;
export const DEFAULT_KB_THRESHOLD = 0.55;
export const DEFAULT_ANSWER_CUTOFF = 0.5;

/**
 * @returns { primary, pNone, topKb, share, declined, confident, searched: [routeId] }
 *   primary: "none" when retrieval is skipped, else the leading knowledge base
 *   share:   the leading knowledge base's share of the knowledge-base probability
 */
export function decideRoute(probs, { declineAt = DEFAULT_DECLINE_AT, kbThreshold = DEFAULT_KB_THRESHOLD } = {}) {
  const pNone = probs.none ?? 0;
  const kbs = Object.entries(probs).filter(([id]) => id !== "none").sort((a, b) => b[1] - a[1]);
  const sum = kbs.reduce((a, [, p]) => a + p, 0) || 1;
  const topKb = kbs[0]?.[0] ?? null, share = kbs.length ? kbs[0][1] / sum : 0;
  const declined = pNone >= declineAt || !topKb;
  const confident = share >= kbThreshold;
  const searched = declined ? [] : confident ? [topKb] : kbs.slice(0, 2).map(([id]) => id);
  return { primary: declined ? "none" : topKb, pNone, topKb, share, declined, confident, searched };
}

/** Pick the best answer among the searched routes: the route whose passages most likely answer the question. */
export function bestAnswer(perRoute, searched, cutoff = DEFAULT_ANSWER_CUTOFF) {
  const cands = searched.map((id) => ({ id, ...perRoute[id] })).filter((x) => x.pAnswered != null);
  if (!cands.length) return null;
  const best = cands.sort((a, b) => b.pAnswered - a.pAnswered)[0];
  return { route: best.id, pAnswered: best.pAnswered, answered: best.pAnswered >= cutoff, passages: best.passages };
}

/**
 * What the whole router did with one question, given a stored record (router probabilities, plus retrieval and
 * answer-check results for the top knowledge bases). Re-run on every slider move; no model calls.
 */
export function outcome(rec, { declineAt = DEFAULT_DECLINE_AT, kbThreshold = DEFAULT_KB_THRESHOLD, cutoff = DEFAULT_ANSWER_CUTOFF } = {}) {
  const d = decideRoute(rec.routeProbs, { declineAt, kbThreshold });
  const missing = d.searched.filter((id) => !rec.perRoute?.[id]);
  const ans = d.declined ? null : bestAnswer(rec.perRoute || {}, d.searched.filter((id) => rec.perRoute?.[id]), cutoff);
  const kind = d.declined ? "declined" : ans?.answered ? "answered" : "not_found";
  // searched but nothing answered, and "none" was the model's single most likely route: treat as small talk
  const smallTalk = kind === "not_found" && d.pNone >= Math.max(...Object.values(rec.routeProbs));
  const r = { ...d, answer: ans, kind, smallTalk, missing };
  if (rec.route) {
    r.routeCorrect = d.primary === rec.route;
    r.kbCorrect = rec.route === "none" ? null : d.topKb === rec.route;
    r.goldSearched = rec.route === "none" ? null : d.searched.includes(rec.route);
    // small talk is handled correctly as long as no knowledge-base passage is served as its answer
    if (rec.route === "none") r.correct = kind !== "answered";
    else if (rec.answerable) r.correct = kind === "answered" && ans.route === rec.route && !!ans.passages?.some((p) => rec.evidence && p.text.includes(rec.evidence));
    else r.correct = kind === "not_found";
  }
  return r;
}

const ratio = (a, b) => (b ? a / b : null);

export function routerMetrics(records, routes, opts = {}) {
  const outs = records.map((r) => ({ r, o: outcome(r, opts) }));
  const ids = routes.map((x) => x.id);
  const confusion = Object.fromEntries(ids.map((g) => [g, Object.fromEntries(ids.map((p) => [p, 0]))]));
  for (const { r, o } of outs) confusion[r.route][o.primary]++;
  const kb = outs.filter(({ r }) => r.route !== "none");
  const answerable = kb.filter(({ r }) => r.answerable), unanswerable = kb.filter(({ r }) => !r.answerable);
  // answer check measured on the gold knowledge base's own results, independent of routing
  const check = kb.filter(({ r }) => r.perRoute?.[r.route]);
  const small = outs.filter(({ r }) => r.route === "none");
  return {
    n: records.length,
    kbChoiceAccuracy: ratio(kb.filter(({ o }) => o.kbCorrect).length, kb.length),
    goldSearched: ratio(kb.filter(({ o }) => o.goldSearched).length, kb.length),
    wronglySkipped: kb.filter(({ o }) => o.declined).length,
    smallTalkHandled: ratio(small.filter(({ o }) => o.correct).length, small.length),
    smallTalkSkipped: small.filter(({ o }) => o.declined).length,
    fanOut: outs.filter(({ o }) => !o.declined && !o.confident).length,
    perRoute: ids.map((id) => { const g = outs.filter(({ r }) => r.route === id); return { id, n: g.length, correct: ratio(g.filter(({ o }) => o.correct).length, g.length), routed: ratio(g.filter(({ o }) => o.routeCorrect).length, g.length) }; }),
    confusion,
    answerCheckAccuracy: ratio(check.filter(({ r }) => (r.perRoute[r.route].pAnswered >= (opts.cutoff ?? DEFAULT_ANSWER_CUTOFF)) === r.answerable).length, check.length),
    evidenceRecall: ratio(answerable.filter(({ r }) => r.perRoute?.[r.route]?.passages.some((p) => p.text.includes(r.evidence))).length, answerable.length),
    endToEnd: ratio(outs.filter(({ o }) => o.correct).length, outs.length),
    answeredCorrectly: ratio(answerable.filter(({ o }) => o.correct).length, answerable.length),
    notFoundCorrectly: ratio(unanswerable.filter(({ o }) => o.correct).length, unanswerable.length),
    outs,
  };
}
