# Requirements: Browser-Native RAG Claim Verification with Laya

Status: v1 implemented (standalone repo, not inside layaForWeb; see README for results and deviations)
Owner: Vishal
Related: layaForWeb (web/app.js, web/workflows.js, web/laya-core.js)

## 1. Goal

Build a fully client-side pipeline that retrieves supporting evidence for a
claim and asks the Laya typed-decisions model to judge whether the evidence
actually supports the claim, "supported" or "not supported." Every step,
embedding, storage, retrieval, and judgment, runs inside the browser tab.
Nothing is sent to a server, no API key is required, and no cost is incurred
beyond the visitor's own CPU (or GPU, if WebGPU is used).

## 2. Background

layaForWeb already runs the Laya decision model (general and fine-tuned
typed-decisions checkpoints) in-browser through ONNX Runtime Web, with a
typed-question interface (`systemOne(state, questions)`) that returns a
labeled choice with a probability distribution. That typed-question pattern
is a natural fit for a binary verification question: given a claim and a
passage of evidence, does the evidence support the claim.

What layaForWeb does not yet have is a retrieval layer. Today, any evidence
text has to be typed or pasted in by hand. A real RAG claim-verification
demo needs a small corpus of documents, a way to chunk and embed them, a way
to store those embeddings between page loads, and a way to pull the most
relevant passages for a given claim before Laya ever sees them.

## 3. Scope

In scope:
- A static, public benchmark of labeled claim/evidence pairs, used as the
  source corpus and as ground truth for evaluation.
- Client-side chunking of source documents into short passages.
- Client-side embedding of passages and claims using a small ONNX
  embedding model that runs the same way Laya does (WASM, browser-cached).
- A browser-local vector store built on IndexedDB, holding passage text,
  passage embeddings, and metadata (source dataset, example id).
- Cosine-similarity retrieval of the top-k passages for a given claim,
  computed in JavaScript against the IndexedDB-stored vectors.
- A typed "supported / not supported" question, asked of Laya with the
  claim and its retrieved evidence as the state.
- Confidence-based result gating, reusing the AUTO / HOLD / BLOCK pattern
  already built for the other workflows, so low-confidence verifications
  are flagged rather than trusted outright.
- An evaluation harness that runs the pipeline across a batch of labeled
  examples, scores the output against the gold labels, and renders a
  results table (per-dataset accuracy, false-verification rate, and a
  confidence-vs-coverage breakdown).
- Caching of computed embeddings and evaluation results in IndexedDB so a
  repeat visit does not redo the batch run from scratch.

Out of scope for this first pass:
- Any call to a hosted LLM or paid API. This build compares Laya against
  itself and against the labeled ground truth, not against another model.
- Any live web crawling or document upload flow. The corpus for v1 is the
  fixed benchmark sample described in Section 7.
- Multi-hop or multi-document reasoning across passages. Each claim is
  checked against its own top-k retrieved passages independently.
- Redistributing the full benchmark dataset. Only a small, clearly licensed
  sample ships with the demo, per Section 7.

## 4. Functional Requirements

### 4.1 Corpus loading
The demo ships a static JSON file containing a stratified sample of claims,
each with its labeled evidence document and its gold label (supported or
not supported). The file is fetched once and cached in IndexedDB.

### 4.2 Chunking
Each evidence document is split into passages of roughly fixed length
(sentence-boundary aware, target 2 to 4 sentences per chunk) before
embedding. Chunk size is a tunable constant, not hardcoded per document.

### 4.3 Embedding and indexing
A small sentence-embedding model, distinct from Laya's own backbone, is
loaded through ONNX Runtime Web (or transformers.js on top of it) the first
time the page runs, and its weights are cached by the browser the same way
Laya's weights already are. Every passage is embedded once and the
resulting vector is written to an IndexedDB object store keyed by dataset
id, example id, and chunk index.

### 4.4 Retrieval
Given a claim, the same embedding model encodes it, and a brute-force
cosine-similarity scan over the IndexedDB-stored passage vectors for that
example's document returns the top-k passages (k configurable, default 3).
Brute-force is acceptable at this corpus size; no external ANN library is
required for v1.

### 4.5 Claim verification
The claim plus its retrieved passages become the `state` for a Laya typed
question with two options, "supported" and "not supported." The model
returns a label and a probability distribution, from which a confidence
score (one minus normalized entropy) is derived, matching the convention
already used elsewhere in layaForWeb.

### 4.6 Confidence-based routing
Each verification result is classified the same way the existing workflows
classify their outcomes: above a configurable confidence threshold the
result is treated as reliable, below it the result is flagged for review.
The UI must make this threshold adjustable, mirroring the slider already
used on the workflows page.

### 4.7 Evaluation harness
A dedicated page runs the pipeline over the full loaded sample (not one
claim at a time), shows live progress (n of total processed), and on
completion renders:
- Overall accuracy and balanced accuracy against gold labels.
- Per-dataset accuracy, if the sample spans more than one source dataset.
- False-verification rate: the share of genuinely unsupported claims the
  model marked as supported.
- A confidence-vs-coverage table: at each of a few confidence thresholds,
  what fraction of claims clear the bar and what the accuracy is among
  those that do.
- Median and p95 latency per verification call.
Results are cached in IndexedDB, keyed by model checkpoint and threshold
settings, so switching the confidence slider does not require a full rerun
unless the underlying verifications changed.

## 5. Non-Functional Requirements

- Fully offline-capable after first load: once the corpus, embedding
  model, and Laya checkpoint are cached, the page should work with no
  network connection.
- No user data or claim text leaves the browser at any point.
- Works on the existing WASM backend layaForWeb already supports; WebGPU
  is a nice-to-have acceleration, not a requirement.
- The batch evaluation run must not freeze the page; use progress
  feedback and yield to the browser between calls so the tab stays
  responsive.
- Reuses existing layaForWeb conventions: the `BASES` / manifest loading
  pattern for model selection, the confidence-gating UI pattern from
  `workflows.js`, and the existing style and layout of the demo pages.

## 6. Technical Architecture

Components:
1. `web/rag-eval.html` and `web/rag-eval.js` (new): the evaluation page
   and its driving script.
2. An embedding module (new): loads and runs the embedding model, exposes
   an `embed(text) -> Float32Array` function.
3. An IndexedDB module (new): a thin wrapper for two object stores,
   `passages` (text, embedding, dataset id, example id, chunk index) and
   `runs` (cached evaluation results keyed by config).
4. Retrieval module (new): cosine similarity search over a given
   example's passage vectors.
5. Existing `laya-core.js` / `systemOne` call: unchanged, reused for the
   typed "supported / not supported" question.
6. Existing confidence-gating and threshold-slider UI: reused from
   `workflows.js`, adapted to this page's result shape.

Data flow: load corpus JSON -> chunk each document -> embed each chunk and
store in IndexedDB -> for each claim, embed the claim -> retrieve top-k
passages from IndexedDB -> build the typed question -> call Laya ->
classify by confidence -> compare to gold label -> aggregate into the
results table.

## 7. Data Requirements

- Source: a public, academically licensed claim-verification benchmark
  (for example, the LLM-AggreFact collection introduced by Tang, Laban and
  Durrett, EMNLP 2024, which aggregates several grounded-factuality
  datasets under one binary supported / not-supported label).
- Before shipping any sample, confirm the dataset's license permits
  redistributing a small static subset alongside this demo. If it does
  not, host only enough of it to demonstrate the pipeline, or generate a
  small hand-labeled sample instead, and say so plainly in the demo's
  documentation.
- Sample size for v1: small enough to run end to end in a few minutes in
  a browser tab, on the order of 50 to 150 examples, expandable later.

## 8. Evaluation Metrics

- Accuracy and balanced accuracy against gold labels.
- False-verification rate (unsupported claims marked supported).
- Confidence-vs-coverage table at a small set of threshold values.
- Latency per verification call (median, p95).
No comparison to any other model or vendor is part of this build; the
comparison is Laya's retrieval-augmented judgment against the benchmark's
own gold labels.

## 9. Open Questions

- Which embedding model gives the best size-to-quality tradeoff for
  browser deployment: a MiniLM-class model or something smaller.
- Whether chunk size and top-k should be exposed as UI controls or fixed
  constants for v1.
- Whether the confidence-gating threshold should default to the same 0.90
  used elsewhere in layaForWeb, or be tuned separately for this task.
- Final license check on the source benchmark before any sample ships.

## 10. Suggested Phases

1. Build the IndexedDB vector store and embedding pipeline, verify
   retrieval quality by hand on a handful of examples.
2. Wire retrieval output into a typed Laya question and confirm the
   judge's output is sane on the same handful of examples.
3. Build the batch evaluation harness and results table on a small
   sample (10 to 20 examples).
4. Scale the sample up, tune chunk size, top-k, and confidence threshold.
5. Polish the UI, add caching, and write up the results if they are
   worth publishing.
