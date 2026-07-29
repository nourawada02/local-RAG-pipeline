from __future__ import annotations

import json
import time
from pathlib import Path

from langchain_ollama import ChatOllama

from build_index import COLLECTION_NAME, EMBEDDING_MODEL
from rag_query import GENERATION_MODEL, generate_answer, stop_ollama_model
from run_balanced_hybrid_evaluation import (
    FINAL_TOP_K,
    normalize_balanced_retrieved,
    page_key,
)
from run_chunking_retrieval_sweep import (
    CONFIGS,
    index_is_valid,
    open_index,
    paths_for,
)
from run_decomposed_hybrid_evaluation import (
    normalize_decomposed_retrieved,
    retrieve_with_decomposition,
)
from run_hybrid_evaluation import (
    BM25_B,
    BM25_CANDIDATE_K,
    BM25_K1,
    DENSE_CANDIDATE_K,
    RRF_CONSTANT,
    BM25Index,
    load_frozen_chunks,
)
from run_ocr_evaluation import (
    GOLD_FILE,
    RESULTS_DIR,
    extract_citations,
    is_abstention,
    read_jsonl,
    validate_inputs,
)


# Confirmation run for the provisional chunking-sweep winner. Only recursive
# chunk size and overlap differ from the frozen decomposed-hybrid control.
# OCR text, embedding model, Chroma cosine distance, balanced dense+BM25
# retrieval, decomposition, final top-k, prompt, and generator remain fixed.
SMALL_CONFIG = next(
    config for config in CONFIGS if config.name == "small_500_100"
)
PREDICTIONS_FILE = (
    RESULTS_DIR / "small_chunk_decomposed_hybrid_predictions.jsonl"
)
CONFIG_FILE = RESULTS_DIR / "small_chunk_decomposed_hybrid_run_config.json"


def load_index_manifest() -> dict:
    manifest_file = paths_for(SMALL_CONFIG)["index_manifest"]
    if not manifest_file.exists():
        raise FileNotFoundError(
            "The 500/100 sweep index manifest was not found at "
            f"{manifest_file}. Run run_chunking_retrieval_sweep.py first."
        )
    try:
        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(
            f"Invalid sweep index manifest: {manifest_file}"
        ) from error

    expected = {
        "experiment": "chunking_retrieval_sweep",
        "name": SMALL_CONFIG.name,
        "chunk_size_characters": SMALL_CONFIG.chunk_size_characters,
        "chunk_overlap_characters": SMALL_CONFIG.chunk_overlap_characters,
        "embedding_model": EMBEDDING_MODEL,
        "collection_name": COLLECTION_NAME,
        "complete": True,
    }
    mismatches = {
        key: {"expected": value, "actual": manifest.get(key)}
        for key, value in expected.items()
        if manifest.get(key) != value
    }
    if mismatches:
        raise ValueError(
            "The 500/100 sweep index does not match this experiment: "
            f"{mismatches}"
        )
    return manifest


def validate_small_index(manifest: dict) -> Path:
    database = paths_for(SMALL_CONFIG)["database"]
    if not database.exists():
        raise FileNotFoundError(
            "The 500/100 sweep index was not found at "
            f"{database}. Run run_chunking_retrieval_sweep.py first."
        )

    corpus_sha256 = manifest.get("corpus_sha256")
    chunk_count = int(manifest.get("chunk_count", 0))
    if (
        not isinstance(corpus_sha256, dict)
        or not corpus_sha256
        or chunk_count <= 0
        or not index_is_valid(
            SMALL_CONFIG,
            corpus_sha256,
            chunk_count,
        )
    ):
        raise ValueError(
            "The 500/100 sweep index failed its manifest validation. "
            "Rerun run_chunking_retrieval_sweep.py to rebuild it safely."
        )
    return database


def save_run_config(
    database: Path,
    index_manifest: dict,
    corpus_size: int,
) -> None:
    config = {
        "experiment_id": "small_chunk_decomposed_balanced_hybrid_top_3",
        "independent_variable": "recursive chunk size and overlap",
        "control_experiment": "ocr_decomposed_balanced_hybrid_top_3",
        "screening_experiment": "Part 2 controlled chunking retrieval sweep",
        "screening_consistency_check": {
            "purpose": (
                "diagnostic comparison with the frozen retrieval-only sweep"
            ),
            "policy": (
                "record exact chunk-level drift but do not abort; the "
                "validated index manifest and live retrieval define the "
                "confirmation run"
            ),
            "reason": (
                "separate Chroma/Ollama sessions can reorder near-tied dense "
                "candidates"
            ),
        },
        "reused_index": str(database),
        "collection_name": COLLECTION_NAME,
        "corpus_chunks": corpus_size,
        "chunking": "RecursiveCharacterTextSplitter",
        "chunk_size_characters": SMALL_CONFIG.chunk_size_characters,
        "chunk_overlap_characters": (
            SMALL_CONFIG.chunk_overlap_characters
        ),
        "average_chunk_characters": index_manifest.get(
            "average_chunk_characters"
        ),
        "embedding_model": EMBEDDING_MODEL,
        "vector_database": "Chroma",
        "distance": "cosine",
        "retrieval": {
            "dense": {
                "method": "Chroma cosine similarity",
                "candidate_k": DENSE_CANDIDATE_K,
            },
            "lexical": {
                "method": "BM25",
                "candidate_k": BM25_CANDIDATE_K,
                "k1": BM25_K1,
                "b": BM25_B,
                "storage": "in memory from 500/100 OCR chunks",
            },
            "fusion": {
                "candidate_scoring": "equal-weight Reciprocal Rank Fusion",
                "rrf_constant": RRF_CONSTANT,
                "decomposition_trigger": "explicit both X and Y syntax",
                "decomposed_selection": [
                    "best RRF page for first subquery",
                    "best RRF page for second subquery",
                    "best remaining full-question RRF page",
                ],
                "non_decomposed_selection": (
                    "balanced dense anchor, different BM25 page, "
                    "different RRF page"
                ),
                "page_deduplication": True,
                "uses_gold_labels_or_evidence": False,
            },
            "final_top_k": FINAL_TOP_K,
        },
        "generation_model": GENERATION_MODEL,
        "temperature": 0.2,
        "num_ctx": 4096,
        "num_predict": 256,
        "reasoning": False,
        "ocr": {
            "enabled": True,
            "strategy": (
                "reuse shared native extraction with selective "
                "Tesseract fallback"
            ),
        },
        "predictions_file": str(PREDICTIONS_FILE),
    }
    CONFIG_FILE.write_text(
        json.dumps(config, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def load_completed_predictions() -> dict[str, dict]:
    if not PREDICTIONS_FILE.exists():
        return {}

    completed: dict[str, dict] = {}
    for row in read_jsonl(PREDICTIONS_FILE):
        question_id = str(row["id"])
        if question_id in completed:
            raise ValueError(
                f"Duplicate prediction ID in {PREDICTIONS_FILE}: "
                f"{question_id}"
            )
        completed[question_id] = row
    return completed


def retrieval_signature(retrieved: list[dict]) -> list[tuple]:
    return [
        (
            int(item["rank"]),
            str(item["doc_id"]),
            tuple(int(page) for page in item.get("pages", [])),
            int(item["chunk"]),
            str(item["selection_role"]),
        )
        for item in retrieved
    ]


def load_screening_retrieval(gold_ids: set[str]) -> dict[str, dict]:
    screening_file = paths_for(SMALL_CONFIG)["predictions"]
    if not screening_file.exists():
        raise FileNotFoundError(
            "The 500/100 retrieval-screening results were not found at "
            f"{screening_file}. Run run_chunking_retrieval_sweep.py first."
        )
    by_id: dict[str, dict] = {}
    for row in read_jsonl(screening_file):
        question_id = str(row["id"])
        if question_id in by_id:
            raise ValueError(
                f"Duplicate screening ID in {screening_file}: "
                f"{question_id}"
            )
        by_id[question_id] = row
    if set(by_id) != gold_ids:
        raise ValueError(
            "The 500/100 screening IDs do not match the current gold set."
        )
    return by_id


def run_evaluation() -> None:
    validate_inputs()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    index_manifest = load_index_manifest()
    database = validate_small_index(index_manifest)
    gold_rows = read_jsonl(GOLD_FILE)
    completed = load_completed_predictions()
    gold_ids = {str(row["id"]) for row in gold_rows}
    screening_by_id = load_screening_retrieval(gold_ids)
    extra_ids = sorted(set(completed) - gold_ids)
    if extra_ids:
        raise ValueError(
            f"Predictions contain IDs not present in the gold set: {extra_ids}"
        )

    vector_store = open_index(SMALL_CONFIG)
    frozen_chunks = load_frozen_chunks(vector_store)
    expected_chunks = int(index_manifest["chunk_count"])
    if len(frozen_chunks) != expected_chunks:
        raise RuntimeError(
            f"Loaded {len(frozen_chunks)} chunks from the 500/100 index; "
            f"expected {expected_chunks}."
        )
    bm25_index = BM25Index(frozen_chunks)
    save_run_config(
        database=database,
        index_manifest=index_manifest,
        corpus_size=len(frozen_chunks),
    )

    llm = ChatOllama(
        model=GENERATION_MODEL,
        temperature=0.2,
        num_ctx=4096,
        num_predict=256,
        reasoning=False,
    )
    pending = [
        row for row in gold_rows if str(row["id"]) not in completed
    ]

    print("Controlled confirmation: 500/100 decomposed balanced hybrid")
    print(f"Reusing verified sweep index: {database.resolve()}")
    print(
        "Changed variable: recursive chunks 1000/200 -> 500/100 "
        "characters"
    )
    print(
        "Frozen controls: OCR, embeddings, Chroma cosine, balanced hybrid, "
        "decomposition, top-3 prompt, and Qwen settings"
    )
    print(f"BM25 corpus: {len(frozen_chunks)} verified 500/100 chunks")
    print(
        f"Gold questions: {len(gold_rows)} | "
        f"already completed: {len(completed)} | pending: {len(pending)}"
    )

    try:
        with PREDICTIONS_FILE.open("a", encoding="utf-8") as output:
            for position, gold in enumerate(pending, start=1):
                question_id = str(gold["id"])
                question = str(gold["question"])
                print(
                    f"\n[{position}/{len(pending)}] "
                    f"{question_id}: {question}"
                )
                total_start = time.perf_counter()

                stop_ollama_model(GENERATION_MODEL)
                retrieval_start = time.perf_counter()
                (
                    results,
                    decomposed_selected,
                    balanced_selected,
                    subqueries,
                    query_runs,
                ) = retrieve_with_decomposition(
                    vector_store=vector_store,
                    bm25_index=bm25_index,
                    question=question,
                )
                retrieval_ms = (
                    time.perf_counter() - retrieval_start
                ) * 1000

                if len(results) != FINAL_TOP_K:
                    raise RuntimeError(
                        f"Expected {FINAL_TOP_K} selected chunks, "
                        f"but received {len(results)}."
                    )
                if (
                    len(
                        {
                            page_key(document)
                            for document, _score in results
                        }
                    )
                    != FINAL_TOP_K
                ):
                    raise RuntimeError(
                        "Small-chunk fusion produced duplicate source pages."
                    )

                if decomposed_selected is None:
                    normalized_retrieved = normalize_balanced_retrieved(
                        balanced_selected
                    )
                else:
                    normalized_retrieved = normalize_decomposed_retrieved(
                        decomposed_selected
                    )

                expected_retrieved = screening_by_id[question_id]["retrieved"]
                expected_signature = retrieval_signature(expected_retrieved)
                actual_signature = retrieval_signature(normalized_retrieved)
                screening_exact_match = (
                    actual_signature == expected_signature
                )
                if not screening_exact_match:
                    print(
                        "Retrieval drift warning: live confirmation retrieval "
                        "differs from the frozen screening result."
                    )
                    print(f"  Expected: {expected_signature}")
                    print(f"  Actual:   {actual_signature}")
                    print(
                        "  Continuing with the actual retrieved context; "
                        "the validated 500/100 index and retrieval settings "
                        "are unchanged."
                    )

                stop_ollama_model(EMBEDDING_MODEL)
                generation_start = time.perf_counter()
                answer = generate_answer(
                    llm=llm,
                    question=question,
                    results=results,
                )
                generation_ms = (
                    time.perf_counter() - generation_start
                ) * 1000
                total_ms = (time.perf_counter() - total_start) * 1000

                prediction = {
                    "id": question_id,
                    "answer": answer,
                    "abstained": is_abstention(answer),
                    "chunking": {
                        "name": SMALL_CONFIG.name,
                        "size_characters": (
                            SMALL_CONFIG.chunk_size_characters
                        ),
                        "overlap_characters": (
                            SMALL_CONFIG.chunk_overlap_characters
                        ),
                    },
                    "decomposition": {
                        "applied": len(subqueries) > 1,
                        "subqueries": (
                            subqueries if len(subqueries) > 1 else []
                        ),
                    },
                    "retrieved": normalized_retrieved,
                    "screening_consistency": {
                        "exact_chunk_level_match": screening_exact_match,
                        "expected_signature": expected_signature,
                        "actual_signature": actual_signature,
                    },
                    "citations": extract_citations(
                        answer,
                        normalized_retrieved,
                    ),
                    "retrieval_ms": round(retrieval_ms, 3),
                    "generation_ms": round(generation_ms, 3),
                    "total_ms": round(total_ms, 3),
                }
                output.write(
                    json.dumps(prediction, ensure_ascii=False) + "\n"
                )
                output.flush()

                print(
                    "Decomposition: "
                    + (
                        " | ".join(subqueries)
                        if len(subqueries) > 1
                        else "not applied"
                    )
                )
                print(
                    "Retrieval queries: "
                    + ", ".join(
                        (
                            f"{run.query_index}:dense={run.dense_count},"
                            f"bm25={run.bm25_count}"
                        )
                        for run in query_runs
                    )
                )
                print(f"Answer: {answer}")
                print(
                    f"Latency: retrieval={retrieval_ms:.0f} ms, "
                    f"generation={generation_ms:.0f} ms, "
                    f"total={total_ms:.0f} ms"
                )
                stop_ollama_model(GENERATION_MODEL)
    finally:
        stop_ollama_model(EMBEDDING_MODEL)
        stop_ollama_model(GENERATION_MODEL)

    final_rows = load_completed_predictions()
    if len(final_rows) != len(gold_rows):
        raise RuntimeError(
            f"Only {len(final_rows)} of {len(gold_rows)} predictions exist."
        )

    print(f"\nCompleted all {len(gold_rows)} questions.")
    print(f"Predictions saved to: {PREDICTIONS_FILE.resolve()}")
    print(f"Run configuration saved to: {CONFIG_FILE.resolve()}")
    print("\nCalculate small-chunk confirmation metrics with:")
    print(
        r"python rag-evaluation-starter\evaluation\metrics.py "
        r"--gold rag-evaluation-starter\evaluation\gold_questions.jsonl "
        r"--predictions "
        r"rag-evaluation-starter\results"
        r"\small_chunk_decomposed_hybrid_predictions.jsonl "
        r"--output "
        r"rag-evaluation-starter\results"
        r"\small_chunk_decomposed_hybrid_metrics.json"
    )


def main() -> None:
    run_evaluation()


if __name__ == "__main__":
    main()
