# Production RAG Assignment Report

## 1. Objective and corpus

The goal was to build and evaluate a complete local RAG system: ingest, extract, chunk, embed, store, retrieve, generate, cite, abstain, and expose the workflow through a REST API.

The frozen corpus contains three related NIST publications:

| Document | Form | Physical pages | Test purpose |
|---|---|---:|---|
| NIST AI RMF 1.0 | Native text PDF | 48 | Core definitions, categories, and cross-page questions |
| NIST AI 600-1 Generative AI Profile | Native text PDF | 64 | Similar terminology and exact technical identifiers |
| NIST SP 800-218A | Deliberately image-only PDF | 30 | OCR and multi-page scanned evidence |

This corpus made the evaluation harder than a collection of unrelated clean PDFs. Similar terminology tests ranking, the scanned document tests ingestion, and compound questions test evidence coverage.

## 2. Baseline

The first pipeline used `pypdf`, recursive `1000/200` character chunks, `mxbai-embed-large`, persistent Chroma cosine search, dense top-three retrieval, and `qwen3.5:2b` at temperature `0.2` with a 4096-token context and 256-token output budget.

The baseline achieved 59.62% Recall@3, 51.47% required-fact recall, 51.92% citation F1, and 50% unanswerable accuracy. Its p95 total latency was 37.42 seconds.

The main failure was not generation. `pypdf` returned no text for the 30 image-only pages, so those pages never reached chunking, embedding, or retrieval. No prompt can recover evidence that ingestion discarded.

## 3. Selective OCR

The system now tries native PDF extraction first and runs Tesseract only on pages with no text layer. OCR pages are rendered at 300 DPI. A hard gate rejects the upload when both methods return empty text for a page.

Selective OCR preserved cleaner and faster native extraction on 112 pages while recovering all 30 scanned pages. Recall@3 rose from 59.62% to 75.00%, fact recall from 51.47% to 66.26%, and citation F1 from 51.92% to 66.67%.

Universal OCR was rejected because it would spend time re-recognizing already-readable text and could introduce recognition errors into clean pages.

## 4. Chunking and embeddings

`RecursiveCharacterTextSplitter` attempts paragraph, line, sentence, word, and finally character boundaries in that order. The selected size is 1000 characters with 200 characters of overlap.

The `500/100` experiment slightly improved retrieval recall from 88.46% to 90.38%, but final answer fact recall fell from 68.89% to 59.37%. Small chunks found relevant pages but often omitted the surrounding explanation needed by the 2B generator. Retrieval-only optimization would have selected the wrong configuration.

`mxbai-embed-large` was retained as the evaluated production embedding model for the English technical corpus. It runs locally through Ollama and supports semantic paraphrase retrieval. 

## 5. Chroma

Chroma was the appropriate trade-off for a local single-machine assignment: persistent storage, cosine search, LangChain integration, and metadata attached to every vector without operating a database server.

Physical PDF page, filename, document ID, extraction method, and chunk number were required for duplicate handling, evaluation, and source attribution.


## 6. Hybrid retrieval and fusion

Dense retrieval handles meaning and paraphrases but may underweight exact identifiers, acronyms, and OCR-distorted technical phrases. BM25 provides exact lexical matching and rare-term weighting.

For each query, the improved pipeline retrieves up to 20 dense and 20 BM25 candidates. Equal-weight reciprocal-rank fusion combines ranks rather than incorrectly adding incompatible cosine distances and BM25 scores:

\[
RRF(d)=\sum_r \frac{1}{60+\operatorname{rank}_r(d)}
\]

Basic hybrid RRF reached 82.69% Recall@3 and 72.44% citation F1, but could spend all three context slots on overlapping chunks from one page.

## 7. Balanced top-three selection

The balanced selector reserves:

1. The strongest dense semantic page.
2. The strongest BM25 page from a different physical page.
3. The strongest remaining fused page from a third page.

This raised Recall@3 to 88.46% while keeping the 2B model's context focused. Dense top-five increased fact recall slightly to 69.98%, but p95 latency nearly doubled to 71.66 seconds and unanswerable accuracy fell to 75%.

The result supports a concrete principle: retrieval needs a broad candidate pool; generation needs a small, diverse evidence set.

## 8. Deterministic query decomposition

Questions explicitly shaped as “both X and Y” contain two information needs. One whole-question embedding may represent one side more strongly.

The system deterministically creates two subqueries, reserves a unique page for each, and fills the final slot using the original question. Ordinary questions are unchanged. The method does not inspect question IDs, reference answers, or gold evidence.

This raised fact recall from 67.57% to 68.89% on the saved run. It did not change aggregate Recall@3 and it reduced unanswerable accuracy from 100% to 75%. That regression must be reported, not hidden. The REST API retains decomposition for explicit multi-part coverage and returns an explicit abstention flag, but balanced hybrid without decomposition remains the safer control for refusal behavior.

## 9. Rejected experiments

| Experiment | Observation | Decision |
|---|---|---|
| Dense top five | Slightly higher fact coverage; p95 71.66 s and 75% unanswerable accuracy | Rejected |
| Basic hybrid RRF | Better exact-term retrieval; redundant physical pages | Repaired with balancing |
| Pseudo-relevance feedback | Fact recall stayed near 68.84%; citation F1 fell to 68.08% | Rejected |
| `500/100` chunks | Recall@3 rose; fact recall collapsed to 59.37% | Rejected |
| Parent-child context | More mappings and context with no preserved regression-free win | Rejected |
| Diversity/adaptive/evidence-matrix selectors | Useful diagnostics but no clean end-to-end improvement | Not promoted |
| Qwen reranking | Extra local-model calls and inconsistent ranking without a proven gain | Not promoted |

These techniques were unjustified for this corpus, model, hardware, and context budget.

## 10. Generation and grounding

`qwen3.5:2b` fit the local 8 GB constraint. The generation settings were frozen during controlled experiments:

- Temperature: `0.2`.
- Context window: `4096`.
- Maximum output: `256` tokens.
- Reasoning output: disabled.
- Final evidence budget: three distinct physical pages where possible.

The system prompt restricts Qwen to retrieved evidence, requires `[Source N]` labels, and defines an exact refusal response. The API returns `abstained` explicitly so evaluation and client code do not guess from wording.

Embedding and generation models are unloaded between stages. This trades latency for stable memory usage on the target machine.

## 11. Evaluation method

The 30-question set was frozen before optimization. It contains 26 answerable and 4 unanswerable questions across direct, paraphrased, multi-page, OCR, and OCR multi-page types.

The scorer measures Hit@k, Recall@k, MRR@k, token F1, required-fact recall, citation precision/recall/F1, explicit abstention accuracy, and median/p95 latency.

## 12. Multimodal branch

Five figure pages were rendered as images. `qwen3.5:2b` converted each figure into a description, `mxbai-embed-large` embedded those descriptions, and a separate Chroma visual index supplied one visual description plus two text pages for visual questions.

On 12 visual questions, automatic fact recall rose from 12.50% for the text control to 81.67% for the multimodal challenger; citation F1 rose from 20% to 80%.

Manual review overturned the apparent win. Some answers contained all required labels while placing them in the wrong lifecycle stages or spatial relationships. Keyword-based scoring rewarded the vocabulary and missed the contradiction.

The branch remains experimental and will be further developed.

## 13. REST API implementation

FastAPI now exposes:

- `POST /ingest`: validated multipart upload, selective extraction/OCR, chunking, embedding, Chroma persistence, content-hash deduplication, and extraction counts.
- `POST /query`: decomposition, dense and BM25 candidate retrieval, RRF, balanced selection, grounded generation, citations, explicit abstention, and latency.
- `GET /health`: index count and configured local models.
- `GET /`: service metadata and the interactive documentation route.

The API supports PDFs, common images, DOCX, and UTF-8 text-like formats. It rejects unsupported or empty documents, prevents path traversal through uploaded filenames, enforces a configurable size limit, and does not silently drop scanned PDF pages.

## 14. Final status and limitations

The required text system and REST API are implemented. The architecture follows measured failures rather than feature accumulation.

Known limitations are explicit:

- Real inference requires Ollama and both models running locally.
- OCR requires Tesseract and is slower than native extraction.
- BM25 is rebuilt in memory per query; acceptable for this corpus, not for a large service.
- The multimodal branch is not production-approved.
- Decomposition improved compound fact coverage but the saved run regressed refusal accuracy.

The strongest conclusion is not “maximum accuracy.” It is that each accepted component fixes a measured failure, and misleading improvements were rejected when end-to-end or human review exposed regressions.

