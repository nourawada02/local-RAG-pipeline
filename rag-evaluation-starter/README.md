# RAG Evaluation Starter

This starter freezes the evaluation target before retrieval changes are made. It contains:

- a reproducible three-document NIST corpus;
- a 30-question gold set;
- native-PDF, OCR, direct, paraphrased, multi-chunk, and unanswerable cases;
- deterministic retrieval, answer, citation, abstention, and latency metrics.

## Corpus

| Document | Indexed form | Pages | Purpose |
|---|---:|---:|---|
| NIST AI RMF 1.0 | Native PDF | 48 | Core terminology, categories, and cross-page questions |
| NIST AI 600-1 Generative AI Profile | Native PDF | 64 | Similar vocabulary, exact identifiers, and detailed GAI risks |
| NIST SP 800-218A | Rasterized PDF with no text layer | 30 | OCR robustness |

The corpus is intentionally coherent. Similar terminology across documents makes retrieval ranking meaningful. Do not index the clean SSDF reference copy; doing so would bypass the OCR test.

## 1. Prepare the corpus

From this directory:

```cmd
python -m pip install -r requirements.txt
python scripts\prepare_corpus.py
```

Index only the three PDFs created in `data\corpus`.

The preparation script:

1. downloads official NIST PDFs;
2. verifies their page counts;
3. turns NIST SP 800-218A into an image-only PDF;
4. verifies that the scan has no extractable text;
5. records SHA-256 hashes for reproducibility.

## 2. Validate the gold set

```cmd
python evaluation\validate_gold.py
python evaluation\test_metrics.py
```

Ground-truth page numbers are physical PDF page numbers starting at 1, not the page number printed inside the publication.

## 3. Normalize pipeline output

Run every gold question through the frozen pipeline and write one JSON object per line:

```json
{
  "id": "q001",
  "answer": "The functions are GOVERN, MAP, MEASURE, and MANAGE.",
  "abstained": false,
  "retrieved": [
    {
      "doc_id": "nist_ai_rmf_1_0",
      "pages": [7],
      "text": "..."
    }
  ],
  "citations": [
    {
      "doc_id": "nist_ai_rmf_1_0",
      "pages": [7]
    }
  ],
  "retrieval_ms": 240.5,
  "generation_ms": 1850.2,
  "total_ms": 2090.7
}
```

Every chunk must retain `doc_id` and physical `pages` metadata. If the current pipeline loses page metadata, fix that before benchmarking; otherwise retrieval and citation correctness cannot be measured.

For unanswerable questions, set:

```json
{
  "id": "q027",
  "answer": "I could not find this information in the provided documents.",
  "abstained": true,
  "retrieved": [],
  "citations": [],
  "retrieval_ms": 200.0,
  "generation_ms": 900.0,
  "total_ms": 1100.0
}
```

Do not infer abstention from wording in the evaluator. The pipeline should return an explicit boolean so the measure is stable.

## 4. Calculate metrics

```cmd
python evaluation\metrics.py ^
  --gold evaluation\gold_questions.jsonl ^
  --predictions results\baseline_predictions.jsonl ^
  --output results\baseline_metrics.json
```

The evaluator reports:

| Area | Measures | What they reveal |
|---|---|---|
| Retrieval | Hit@3, Hit@5 | Whether any correct evidence was returned |
| Retrieval | Recall@3, Recall@5 | How much required page-level evidence was returned |
| Ranking | MRR@3, MRR@5 | How early the first relevant result appeared |
| Answer | Exact match | Strict phrasing match; informative but not the main score |
| Answer | Token F1 | Lexical overlap with the reference answer |
| Answer | Required-fact recall | Whether the answer contains the facts required by the rubric |
| Citations | Precision, recall, F1 | Whether cited document-page pairs are correct and complete |
| Reliability | Abstention accuracy | Whether answerable and unanswerable questions are classified correctly |
| Performance | Median and p95 latency | Typical and slow-case retrieval, generation, and total time |

The main optimization measures should be `Recall@5`, `MRR@5`, required-fact recall, citation F1, unanswerable accuracy, and p95 total latency. Exact match is too brittle to use as the deciding score.

Copy each configuration's main results into `evaluation\experiment_log.csv`. Give every run a unique experiment ID and never overwrite the baseline row.

## Human faithfulness check

Faithfulness cannot be honestly reduced to token overlap. For the baseline and each final candidate configuration, manually score every answer:

- `0`: unsupported or contradicted by retrieved text;
- `1`: partly supported, with an unsupported or distorted material claim;
- `2`: fully supported by retrieved text.

Optionally add `human_faithfulness` and `human_correctness` fields, each from 0 to 2, to prediction rows. The evaluator will average any supplied values.

## Experimental rule

Generate and save predictions once per configuration. Never re-run the model while scoring. Keep the generator model, prompt, temperature, and evaluation questions fixed while comparing chunking or retrieval settings. Change one independent variable at a time.
