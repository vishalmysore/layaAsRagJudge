# Jev or Laya: LLM as Judge for RAG Applications — Live Demo

**Live demo:** https://vishalmysore.github.io/layaAsRagJudge/
**Code:** https://github.com/vishalmysore/layaAsRagJudge

A RAG application retrieves some documents, hands them to a model, and gets back an answer that *sounds* grounded. Whether it actually is grounded is a separate question, and it's the one that matters. The usual fix is "LLM-as-judge": after generation, ask a second model whether each claim in the answer is supported by the retrieved evidence.

That second model is normally a large hosted LLM, called over an API, once per claim. This demo tries something much smaller. The judge is **Laya**, a typed-decision model built on ModernBERT-large. It doesn't generate text at all; it answers a multiple-choice question with a probability for each option. The whole pipeline runs **inside a browser tab**: chunking, embedding, the vector store, retrieval and the judgment. There's no server, no API key and no per-call cost, and no claim text leaves the page.

The question the demo answers is simple: *can a small, local, non-generative model do the judge's job, and how well?*

![The Verify page: a claim, the retrieved evidence, Laya's verdict and the gate](images/01-verify-overview.png)

## The pipeline

```
documents ─► sentence-aware chunking (2 sentences per passage)
          ─► all-MiniLM-L6-v2 embeddings (384-d) ─► IndexedDB
claim     ─► same embedder ─► cosine top-3 over IndexedDB
          ─► Laya: one typed question { claim, evidence: [3 passages] }
          ─► p(supported) ─► verdict ─► gate: AUTO / BLOCK / HOLD
```

Two models do two different jobs:

| | Retriever | Judge |
|---|---|---|
| Model | all-MiniLM-L6-v2 (int8 ONNX) | Laya typed-decisions (ModernBERT-large, int8 ONNX) |
| Size | ~23 MB | ~422 MB |
| Job | Find passages *about* the claim | Decide whether they *support* it |
| Output | A 384-number vector per text | A probability per option |
| Time | ~5–10 ms per claim | ~1.5–5 s per claim on WASM |

Both run on the same ONNX Runtime Web instance, streamed from Hugging Face once and then cached, so after the first visit the page works offline.

The split matters because **relevant is not the same as supporting**. An embedding model, or a reranker, scores how closely a passage matches a claim's topic. It has no way to say "this passage is on topic, and it says the opposite." That's the judge's job.

## Asking Laya a typed question

Laya doesn't take a prompt and write a reply. It takes a question, a set of named options, and a *state* (the data to reason about). It then scores each option at a `[MASK]` token. The Verify page shows the exact token sequence the model reads:

```
[CLS]choice question: Does the evidence support the claim?
[SEP]
[MASK] supported: The evidence states the claim or clearly implies it
[MASK] not_supported: The evidence contradicts the claim or does not mention it
[SEP]
{"claim": "All API keys on an account share a single rate limit.",
 "evidence": ["Rate limits are applied separately to each key, …", "…", "…"]}
[SEP]
```

That's 196 tokens, against a context limit of 1,024. The answer comes back as a probability split, for example `supported 0.64 / not_supported 0.36`. There's no parsing of free text and no risk of the judge rambling. The options are plain text too, so you can change the question (add "partially supported", say) without retraining anything.

![Live run: every passage, the question and state, and the exact decoded input Laya reads](images/09-live-contradiction.png)

## The gate: act, block, or ask a person

A verdict alone isn't enough; you also need to know when to trust it. The demo turns p(supported) into two things:

1. **A verdict:** "supported" if p(supported) ≥ the cutoff (0.50), otherwise "not supported".
2. **A confidence:** 1 − normalized entropy of the split. It's 0 at 50/50 and rises as the answer gets more lopsided.

The **gate** combines them:

| Gate | When | What a RAG app would do |
|---|---|---|
| **AUTO** | Confident "supported" | Show the answer with its citation |
| **BLOCK** | Confident "not supported" | Withhold or regenerate the answer |
| **HOLD** | Too close to call | Send it to a person to review |

With the default confidence bar of 0.10, this works out to a simple rule: **AUTO above p ≈ 0.69, BLOCK below p ≈ 0.31, HOLD in between.**

Three live runs show all three outcomes.

**BLOCK, a claim the evidence never makes.** "The study shows eating more carbs helps students sleep", checked against an article about school start times. Retrieval still returns the three most similar passages, because it always returns *something*. Laya gives it 0.23 supported, confidence 0.21, so the claim is blocked. A hallucination like this would be stopped before it reached the user.

![BLOCK: the claim is about carbs, the evidence is about school start times](images/10-live-unsupported-block.png)

**AUTO, your own evidence.** Paste any text, and it's chunked and embedded in memory without being stored. "The Kestrel 5 can ride 90 km per charge in eco mode" against a short product description scores 0.81 supported, confidence 0.31: AUTO.

![AUTO: a claim checked against pasted text](images/11-live-pasted-auto.png)

**HOLD, and why HOLD matters.** "All API keys on an account share a single rate limit." The top passage (cosine 0.647) says limits are applied *separately to each key*, the exact opposite. Laya gets this **wrong**: 0.64 supported. The claim and passage share almost every word (rate limit, keys, account) and differ only in logic, which is exactly where this model is weakest. But 0.64 is close to a coin flip, so confidence is 0.059 and the gate says HOLD. The wrong verdict is never acted on, and the claim goes to a person.

This is the whole point of the gate. Laya's mistakes are concentrated where it's unsure, so the gate catches most of them.

## How good is it? Scoring against an answer key

The Evaluate page runs the pipeline over **72 labeled claims** and compares every verdict with its gold label (the answer key). Each claim is also tagged with the kind of test it is: a paraphrase, a small inference, a changed number, a swapped entity, a direct contradiction, or something the document simply doesn't say.

![Evaluate: accuracy, false-verification rate, AUROC, and every claim as a dot](images/03-eval-summary.png)

In the dot chart, each dot is one claim, placed by p(supported). Green claims are truly supported, red ones aren't. A perfect judge would put every green dot right of the cutoff and every red dot left of it. Faded dots are wrong verdicts, and clicking any dot opens its passages and probabilities:

![Clicking a dot: the API-keys claim, its evidence, and its probabilities](images/05-eval-dot-detail.png)

Results (int8 build, WASM, 72 claims, answer key only; no other model is compared):

| Judge question | Search scope | Accuracy | False verification | AUROC | Evidence in top 3 |
|---|---|---|---|---|---|
| Two options (default) | Claim's document | **75.0%** | 27.8% | **0.80** | 99% |
| Two options | Whole corpus | 75.0% | 27.8% | 0.80 | 96% |
| Three options | Claim's document | 72.2% | **11.1%** | 0.80 | 99% |
| Yes / no | Claim's document | 69.4% | 19.4% | 0.74 | 99% |

*False verification* is the share of unsupported claims the judge called "supported", the hallucinations that would get through.

![Confidence vs coverage, and where the judge struggles](images/04-eval-coverage-breakdowns.png)

What the numbers say:

- **It works, within limits.** An AUROC of 0.80 means Laya separates supported from unsupported claims well above chance. It does this with nothing but a typed question and no task-specific fine-tuning.
- **The gate earns its keep.** At the default bar, 44% of claims are decided automatically, and those verdicts are **84%** accurate, against 75% overall. The rest go to review.
- **Retrieval isn't the bottleneck.** The sentence that proves or disproves the claim is in the top 3 passages for 99% of claims (96% when searching all 118 passages). The errors come from the judge.
- **Where it fails is predictable.** Swapped names were all caught. Contradictions (68%) and changed numbers (60%) are the weak spots, and so are policy texts, whose rules are phrased as exceptions ("cannot be returned", "not covered unless…").
- **The question wording matters.** Splitting "not supported" into *contradicted* and *not mentioned* makes Laya stricter: accuracy dips to 72%, but false verification drops from 28% to 11%. For RAG, where a hallucination getting through is the worse mistake, that's the better trade.

![Three options: fewer hallucinations get through](images/06-eval-three-options.png)

## Honest caveats

- **Laya isn't an LLM.** It's a 400M-parameter encoder answering typed questions. It plays the *role* of an LLM judge, and that's the point: a much smaller, local, structured model in a slot usually filled by a hosted LLM. A model trained specifically for fact-checking, such as MiniCheck or AlignScore, would probably score higher. That comparison is deliberately out of scope here.
- **72 claims is a smoke test, not a benchmark.** The documents are fictional and written for the demo, so a claim's answer depends only on its evidence and never on what the model already knows. The labels were written by a single author, not agreed by several annotators as in LLM-AggreFact.
- **The probabilities are compressed.** Laya rarely goes beyond about 85/15, so no claim in the sample reached a confidence of 0.5. That's why the default bar is 0.10, not the 0.90 you might expect.
- **Latency** is 1.5–5 s per claim on 4 WASM threads. That's fine for checking an answer, but slow for checking thousands of claims in a browser. The int4 build on WebGPU is faster.

## Try it

Open the [live demo](https://vishalmysore.github.io/layaAsRagJudge/). Before downloading anything, the sample claims play back results recorded from the same models. Press **Load models** (about 445 MB once, then cached) to check your own claims against the corpus or against text you paste. On the **Evaluate** page, drag the cutoff and confidence sliders to see how the gate trades automation for accuracy, without calling the model again.

It works on a phone too, and in dark mode:

![Evaluate page in dark mode at phone width](images/12-dark-mobile.png)

The code is Apache-2.0: https://github.com/vishalmysore/layaAsRagJudge
