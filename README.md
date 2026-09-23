# Laya RAG Judge

Browser-native RAG claim verification. A claim is checked against retrieved evidence, and the **Laya typed-decisions model** judges whether that evidence supports it: *supported* or *not supported*, with a probability and a confidence gate. Chunking, embedding, the vector store, retrieval and the judgment all run in the browser tab. There is no server, no API key and no cost, and no claim text leaves the page.

**Live demo:** https://vishalmysore.github.io/layaAsRagJudge/ · **Article:** [Jev or Laya: LLM as Judge for RAG Applications — Live Demo](docs/article.md)

Two pages:

- **Verify a claim** (`index.html`): one claim through the whole pipeline, with each step shown: the retrieved passages and their cosine scores, the exact typed question and state sent to Laya, the probabilities, the AUTO / BLOCK / HOLD gate, and the source document with the retrieved passages highlighted. Evidence can come from one corpus document, the whole corpus, or text you paste.
- **Evaluate** (`rag-eval.html`): runs the pipeline over all 72 labeled claims and scores it against the gold labels. It reports accuracy, balanced accuracy, false-verification rate, AUROC, evidence recall@k, a confidence-vs-coverage table, breakdowns by dataset and by claim type, and median / p95 latency. Moving the sliders re-scores the stored run without calling the model again.

Until the models are loaded, both pages play back results recorded from the same models (`web/recorded.json`).

## Pipeline

```
corpus.json ──► sentence-aware chunking (2 sentences / chunk)
            ──► all-MiniLM-L6-v2 embeddings (384-d) ──► IndexedDB "passages" store
claim ──► same embedder ──► brute-force cosine top-k over that store (k = 3)
      ──► Laya typed question { claim, evidence: [top-k passages] } ──► p(supported)
      ──► verdict (p ≥ cutoff) ──► gate: AUTO (confident supported) / BLOCK (confident not) / HOLD (review)
```

| Piece | File | Notes |
|---|---|---|
| Chunking, cosine, top-k | `web/rag.js` | Pure functions, unit-tested in Node |
| Embedder | `web/embedder.js` | `Xenova/all-MiniLM-L6-v2` int8 ONNX (~23 MB), run by the **same** ONNX Runtime Web and Tokenizers.js as Laya, so there is one runtime. Mean pooling + L2 norm (sentence-transformers). Pinned to a commit and kept in Cache Storage |
| Vector store | `web/store.js` | IndexedDB: `passages` (text, `Float32Array` vector, dataset, doc, chunk), `claims` (claim vectors), `runs` (evaluation runs), `meta` (corpus). Falls back to memory if IndexedDB is blocked |
| Judge | `web/judge.js`, `web/laya-core.js`, `web/model.js` | One typed question per claim; `laya-core.js` and `model.js` come unchanged from layaForWeb / layaForWorkflows |
| Metrics | `web/metrics.js` | Accuracy, balanced accuracy, false-verification rate, AUROC, coverage table, latency percentiles |
| Glue | `web/pipeline.js` | index → retrieve → judge, run keys, recorded-run compaction |

## Results on the sample (int8 build, WASM, 72 claims)

| Judge question | Retrieval scope | Accuracy | Balanced | False-verification | AUROC | Evidence recall@3 |
|---|---|---|---|---|---|---|
| **Two options** (default) | claim's document | **75.0%** | 75.0% | 27.8% | **0.80** | 99% |
| Two options | whole corpus (118 passages) | 75.0% | 75.0% | 27.8% | 0.80 | 96% |
| Three options (supported / contradicted / not mentioned) | claim's document | 72.2% | 72.2% | **11.1%** | 0.80 | 99% |
| Yes / no | claim's document | 69.4% | 69.4% | 19.4% | 0.74 | 99% |

Confidence vs coverage for the default question (confidence = 1 − normalized entropy of the supported / not-supported split):

| Confidence ≥ | Coverage | Accuracy of the kept verdicts |
|---|---|---|
| 0.00 | 100% | 75.0% |
| 0.05 | 63% | 80.0% |
| **0.10** (default gate) | **44%** | **84.4%** |
| 0.20 | 22% | 87.5% |
| 0.30 | 6% | 75.0% (4 claims) |

What these numbers say:

- **Laya works as a judge, within limits.** It separates supported from unsupported claims well above chance (AUROC 0.80), with nothing but a typed question and no fine-tuning for this task.
- **The three-option phrasing is the safer judge.** It gets the same AUROC but lets far fewer unsupported claims through (11% vs 28%). If a false "supported" is the costly mistake, which is usually the case for RAG answers, pick it on the Evaluate page. The two-option question stays the default because it has the best accuracy.
- **Retrieval is not the bottleneck here.** The gold evidence span is in the top 3 passages for 99% of claims (96% when searching the whole corpus). The errors come from the judge, mostly on contradictions and changed numbers, the same weakness with numbers that layaForWorkflows found.
- **The requirements' 0.90 confidence default does not fit this checkpoint.** Its probabilities are compressed: a verdict of p = 0.83 has an entropy confidence of only 0.34, and no claim in the sample reaches 0.5. The default gate is therefore **0.10**, which acts automatically on 44% of claims at 84% accuracy and holds the rest for review.

Caveats: 72 hand-written claims is a smoke test, not a benchmark. The "best cutoff" hint on the Evaluate page is fit on the same claims, so it is optimistic. Latency was measured in a background browser tab (median ≈ 2.6 s per judgment on 4 WASM threads) and is faster in a foreground tab or on WebGPU with the int4 build.

## The sample corpus

`web/corpus.json` holds 18 short documents in three genres (encyclopedic, news, policy / documentation) and 72 claims, 4 per document, balanced 36 supported / 36 not supported. Each claim is tagged by type: *paraphrase*, *inference*, *number*, *entity*, *contradiction* or *unverifiable*. Every non-unverifiable claim carries a verbatim evidence span, which is used to measure retrieval recall.

**Every document is fictional and was written for this demo.** The requirements suggested LLM-AggreFact. Its sub-datasets carry mixed licenses, and several are derived from news articles whose redistribution terms are unclear, so no benchmark text is shipped. With fictional documents, a claim's gold label depends only on its evidence document and never on what the model already knows. The unit tests check the corpus: labels match types, evidence spans exist verbatim and survive chunking at 1 to 4 sentences, and the sample is balanced.

## Run locally

```bash
npm ci
```

```bash
npm test
```

```bash
node scripts/prepare_site.mjs
```

```bash
python serve.py
```

Then open http://localhost:8000. `serve.py` sends the COOP/COEP headers so ONNX Runtime can use several WASM threads. On GitHub Pages, `coi.js` registers a small service worker that adds them instead.

The first model load downloads Laya (422 MB int8, or 278 MB int4 for WebGPU) and the embedder (23 MB). Both are cached, and after that the pages work offline: the corpus, passage vectors and runs are all in IndexedDB.

## Re-recording `recorded.json`

If you change the corpus or a judge question, the unit test on `recorded.json` fails; the run keys include a fingerprint of each question's wording. To refresh the file, load the models on the Evaluate page and run each configuration you want to ship. Then call `copy(JSON.stringify(__lrj.recorded()))` in the console and paste the result into `web/recorded.json`. Passage text and claim text are left out of the file and rebuilt from the corpus at load time.

## Deploy

`.github/workflows/deploy.yml` runs the unit tests, assembles `dist/` (the pages plus vendored ONNX Runtime Web and Tokenizers.js, about 30 MB) and publishes it to GitHub Pages. The models are not bundled; they stream from Hugging Face at runtime.

## License

Apache-2.0. See `NOTICE.md` for the models and third-party code.
