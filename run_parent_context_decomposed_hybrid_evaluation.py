from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path

from langchain_core.documents import Document
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
    tokenize,
)
from run_ocr_evaluation import (
    GOLD_FILE,
    RESULTS_DIR,
    extract_citations,
    is_abstention,
    read_jsonl,
    validate_inputs,
)
from run_small_chunk_decomposed_hybrid_evaluation import (
    load_index_manifest,
    load_screening_retrieval,
    retrieval_signature,
    validate_small_index,
)


# Controlled child-parent retrieval experiment:
# - retrieve and rank the already-screened 500/100 child chunks;
# - map each selected child to the most overlapping frozen 1000/200 chunk on
#   the same source page;
# - send the three parent chunks to Qwen.
#
# Page selection, embeddings, hybrid fusion, decomposition, prompt, generator,
# and top-3 context count remain frozen. Gold answers and gold evidence are
# never used for retrieval, parent mapping, or generation.
CHILD_CONFIG = next(
    config for config in CONFIGS if config.name == "small_500_100"
)
PARENT_CONFIG = next(
    config for config in CONFIGS if config.name == "control_1000_200"
)
MIN_PARENT_TOKEN_COVERAGE = 0.50

PREDICTIONS_FILE = (
    RESULTS_DIR / "parent_context_decomposed_hybrid_predictions.jsonl"
)
CONFIG_FILE = RESULTS_DIR / "parent_context_decomposed_hybrid_run_config.json"


def source_page_key(document: Document) -> tuple[str, int]:
    return (
        str(document.metadata.get("source", "")),
        int(document.metadata.get("page", -1)),
    )


def load_parent_index_manifest(child_manifest: dict) -> tuple[dict, Path]:
    manifest_file = paths_for(PARENT_CONFIG)["index_manifest"]
    database = paths_for(PARENT_CONFIG)["database"]
    if not manifest_file.exists() or not database.exists():
        raise FileNotFoundError(
            "The 1000/200 parent index was not found. "
            "Run run_chunking_retrieval_sweep.py first."
        )

    try:
        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(
            f"Invalid parent index manifest: {manifest_file}"
        ) from error

    expected = {
        "experiment": "chunking_retrieval_sweep",
        "name": PARENT_CONFIG.name,
        "chunk_size_characters": PARENT_CONFIG.chunk_size_characters,
        "chunk_overlap_characters": PARENT_CONFIG.chunk_overlap_characters,
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
            "The 1000/200 parent index does not match this experiment: "
            f"{mismatches}"
        )
    if manifest.get("corpus_sha256") != child_manifest.get("corpus_sha256"):
        raise ValueError(
            "The child and parent indexes were built from different corpus "
            "fingerprints."
        )

    chunk_count = int(manifest.get("chunk_count", 0))
    corpus_sha256 = manifest.get("corpus_sha256")
    if (
        chunk_count <= 0
        or not isinstance(corpus_sha256, dict)
        or not index_is_valid(
            PARENT_CONFIG,
            corpus_sha256,
            chunk_count,
        )
    ):
        raise ValueError(
            "The 1000/200 parent index failed manifest validation. "
            "Rerun run_chunking_retrieval_sweep.py."
        )
    return manifest, database


def build_parent_lookup(
    parent_chunks: list[Document],
) -> dict[tuple[str, int], list[Document]]:
    by_page: dict[tuple[str, int], list[Document]] = {}
    for document in parent_chunks:
        by_page.setdefault(source_page_key(document), []).append(document)
    for documents in by_page.values():
        documents.sort(
            key=lambda item: int(item.metadata.get("chunk", -1))
        )
    return by_page


def token_coverage(child: Document, parent: Document) -> float:
    child_terms = Counter(tokenize(child.page_content))
    if not child_terms:
        return 0.0
    parent_terms = Counter(tokenize(parent.page_content))
    covered = sum((child_terms & parent_terms).values())
    return covered / sum(child_terms.values())


def expand_generation_context(
    child_results: list[tuple[Document, float]],
    parent_by_page: dict[tuple[str, int], list[Document]],
    normalized_retrieved: list[dict],
) -> tuple[list[tuple[Document, float]], list[dict]]:
    if len(child_results) != len(normalized_retrieved):
        raise RuntimeError(
            "Child retrieval and normalized retrieval lengths differ."
        )

    expanded_results: list[tuple[Document, float]] = []
    audit_rows: list[dict] = []

    for rank, ((child, score), normalized) in enumerate(
        zip(child_results, normalized_retrieved),
        start=1,
    ):
        candidates = parent_by_page.get(source_page_key(child), [])
        if not candidates:
            raise RuntimeError(
                "No 1000/200 parent chunks exist for selected child page "
                f"{source_page_key(child)}."
            )

        scored = [
            (
                token_coverage(child, parent),
                -int(parent.metadata.get("chunk", -1)),
                parent,
            )
            for parent in candidates
        ]
        coverage, _tie_break, parent = max(
            scored,
            key=lambda item: (item[0], item[1]),
        )
        if coverage < MIN_PARENT_TOKEN_COVERAGE:
            raise RuntimeError(
                "Could not map a selected 500/100 child to a reliable "
                f"1000/200 parent on {source_page_key(child)}. "
                f"Best token coverage was {coverage:.3f}."
            )

        expanded_metadata = dict(parent.metadata)
        expanded_metadata["retrieval_child_chunk"] = int(
            child.metadata["chunk"]
        )
        expanded_metadata["generation_parent_chunk"] = int(
            parent.metadata["chunk"]
        )
        expanded_metadata["context_expanded"] = True
        expanded_document = Document(
            page_content=parent.page_content,
            metadata=expanded_metadata,
        )
        expanded_results.append((expanded_document, score))

        audit_rows.append(
            {
                "rank": rank,
                "doc_id": normalized["doc_id"],
                "pages": list(normalized["pages"]),
                "retrieval_child_chunk": int(child.metadata["chunk"]),
                "generation_parent_chunk": int(parent.metadata["chunk"]),
                "child_characters": len(child.page_content),
                "parent_characters": len(parent.page_content),
                "child_token_coverage": round(coverage, 6),
                "text": parent.page_content.strip(),
            }
        )

    return expanded_results, audit_rows


def save_run_config(
    child_database: Path,
    parent_database: Path,
    child_manifest: dict,
    parent_manifest: dict,
) -> None:
    config = {
        "experiment_id": (
            "small_child_parent_context_decomposed_balanced_hybrid_top_3"
        ),
        "independent_variable": "post-retrieval generation context",
        "control_experiment": (
            "small_chunk_decomposed_balanced_hybrid_top_3"
        ),
        "hypothesis": (
            "500/100 children preserve retrieval precision while matched "
            "1000/200 parents restore enough surrounding context for complete "
            "answers"
        ),
        "child_index": str(child_database),
        "parent_index": str(parent_database),
        "collection_name": COLLECTION_NAME,
        "child_chunk_count": int(child_manifest["chunk_count"]),
        "parent_chunk_count": int(parent_manifest["chunk_count"]),
        "child_chunk_size_characters": (
            CHILD_CONFIG.chunk_size_characters
        ),
        "child_chunk_overlap_characters": (
            CHILD_CONFIG.chunk_overlap_characters
        ),
        "parent_chunk_size_characters": (
            PARENT_CONFIG.chunk_size_characters
        ),
        "parent_chunk_overlap_characters": (
            PARENT_CONFIG.chunk_overlap_characters
        ),
        "parent_mapping": {
            "constraint": "same source and page as selected child",
            "ranking": "maximum multiset token coverage of child",
            "minimum_child_token_coverage": MIN_PARENT_TOKEN_COVERAGE,
            "uses_gold_labels_or_evidence": False,
        },
        "embedding_model": EMBEDDING_MODEL,
        "vector_database": "Chroma",
        "distance": "cosine",
        "retrieval": {
            "dense": {
                "method": "Chroma cosine similarity over 500/100 children",
                "candidate_k": DENSE_CANDIDATE_K,
            },
            "lexical": {
                "method": "BM25 over 500/100 children",
                "candidate_k": BM25_CANDIDATE_K,
                "k1": BM25_K1,
                "b": BM25_B,
            },
            "fusion": {
                "candidate_scoring": "equal-weight Reciprocal Rank Fusion",
                "rrf_constant": RRF_CONSTANT,
                "decomposition_trigger": "explicit both X and Y syntax",
                "page_deduplication": True,
            },
            "final_top_k_pages": FINAL_TOP_K,
        },
        "generation_context": (
            "matched frozen 1000/200 parent chunk for each selected child"
        ),
        "generation_model": GENERATION_MODEL,
        "temperature": 0.2,
        "num_ctx": 4096,
        "num_predict": 256,
        "reasoning": False,
        "ocr": {
            "enabled": True,
            "strategy": (
                "reuse sweep indexes built from the shared native extraction "
                "and selective Tesseract page cache"
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


def run_evaluation() -> None:
    validate_inputs()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    child_manifest = load_index_manifest()
    child_database = validate_small_index(child_manifest)
    parent_manifest, parent_database = load_parent_index_manifest(
        child_manifest
    )

    gold_rows = read_jsonl(GOLD_FILE)
    gold_ids = {str(row["id"]) for row in gold_rows}
    completed = load_completed_predictions()
    extra_ids = sorted(set(completed) - gold_ids)
    if extra_ids:
        raise ValueError(
            f"Predictions contain IDs not present in the gold set: {extra_ids}"
        )
    screening_by_id = load_screening_retrieval(gold_ids)

    child_vector_store = open_index(CHILD_CONFIG)
    child_chunks = load_frozen_chunks(child_vector_store)
    if len(child_chunks) != int(child_manifest["chunk_count"]):
        raise RuntimeError(
            "The loaded child chunk count does not match its manifest."
        )
    bm25_index = BM25Index(child_chunks)

    parent_vector_store = open_index(PARENT_CONFIG)
    parent_chunks = load_frozen_chunks(parent_vector_store)
    if len(parent_chunks) != int(parent_manifest["chunk_count"]):
        raise RuntimeError(
            "The loaded parent chunk count does not match its manifest."
        )
    parent_by_page = build_parent_lookup(parent_chunks)

    save_run_config(
        child_database=child_database,
        parent_database=parent_database,
        child_manifest=child_manifest,
        parent_manifest=parent_manifest,
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

    print("Controlled experiment: small-child retrieval + parent context")
    print(f"Child retrieval index: {child_database.resolve()}")
    print(f"Parent context index: {parent_database.resolve()}")
    print(
        "Changed variable: generation context 500/100 child -> matched "
        "1000/200 parent"
    )
    print(
        "Frozen controls: selected pages, OCR, embeddings, Chroma cosine, "
        "balanced hybrid, decomposition, top-3, prompt, and Qwen settings"
    )
    print(
        f"Child corpus: {len(child_chunks)} chunks | "
        f"parent corpus: {len(parent_chunks)} chunks"
    )
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
                    child_results,
                    decomposed_selected,
                    balanced_selected,
                    subqueries,
                    query_runs,
                ) = retrieve_with_decomposition(
                    vector_store=child_vector_store,
                    bm25_index=bm25_index,
                    question=question,
                )

                if len(child_results) != FINAL_TOP_K:
                    raise RuntimeError(
                        f"Expected {FINAL_TOP_K} selected children, "
                        f"but received {len(child_results)}."
                    )
                if (
                    len(
                        {
                            page_key(document)
                            for document, _score in child_results
                        }
                    )
                    != FINAL_TOP_K
                ):
                    raise RuntimeError(
                        "Child retrieval produced duplicate source pages."
                    )

                if decomposed_selected is None:
                    normalized_retrieved = normalize_balanced_retrieved(
                        balanced_selected
                    )
                else:
                    normalized_retrieved = normalize_decomposed_retrieved(
                        decomposed_selected
                    )

                expected_signature = retrieval_signature(
                    screening_by_id[question_id]["retrieved"]
                )
                actual_signature = retrieval_signature(normalized_retrieved)
                screening_exact_match = (
                    actual_signature == expected_signature
                )
                if not screening_exact_match:
                    print(
                        "Retrieval drift warning: live child retrieval "
                        "differs from the frozen screening result."
                    )
                    print(f"  Expected: {expected_signature}")
                    print(f"  Actual:   {actual_signature}")

                (
                    generation_results,
                    generation_context,
                ) = expand_generation_context(
                    child_results=child_results,
                    parent_by_page=parent_by_page,
                    normalized_retrieved=normalized_retrieved,
                )
                retrieval_ms = (
                    time.perf_counter() - retrieval_start
                ) * 1000

                stop_ollama_model(EMBEDDING_MODEL)
                generation_start = time.perf_counter()
                answer = generate_answer(
                    llm=llm,
                    question=question,
                    results=generation_results,
                )
                generation_ms = (
                    time.perf_counter() - generation_start
                ) * 1000
                total_ms = (time.perf_counter() - total_start) * 1000

                prediction = {
                    "id": question_id,
                    "answer": answer,
                    "abstained": is_abstention(answer),
                    "child_chunking": {
                        "name": CHILD_CONFIG.name,
                        "size_characters": (
                            CHILD_CONFIG.chunk_size_characters
                        ),
                        "overlap_characters": (
                            CHILD_CONFIG.chunk_overlap_characters
                        ),
                    },
                    "parent_context": {
                        "name": PARENT_CONFIG.name,
                        "size_characters": (
                            PARENT_CONFIG.chunk_size_characters
                        ),
                        "overlap_characters": (
                            PARENT_CONFIG.chunk_overlap_characters
                        ),
                        "mapping": (
                            "same-page maximum child-token coverage"
                        ),
                    },
                    "decomposition": {
                        "applied": len(subqueries) > 1,
                        "subqueries": (
                            subqueries if len(subqueries) > 1 else []
                        ),
                    },
                    "retrieved": normalized_retrieved,
                    "generation_context": generation_context,
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
                print(
                    "Context expansion: "
                    + ", ".join(
                        (
                            f"rank {item['rank']} "
                            f"child {item['retrieval_child_chunk']} -> "
                            f"parent {item['generation_parent_chunk']} "
                            f"({item['child_token_coverage']:.1%} coverage)"
                        )
                        for item in generation_context
                    )
                )
                print(f"Answer: {answer}")
                print(
                    f"Latency: retrieval+expansion={retrieval_ms:.0f} ms, "
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
    print("\nCalculate parent-context metrics with:")
    print(
        r"python rag-evaluation-starter\evaluation\metrics.py "
        r"--gold rag-evaluation-starter\evaluation\gold_questions.jsonl "
        r"--predictions "
        r"rag-evaluation-starter\results"
        r"\parent_context_decomposed_hybrid_predictions.jsonl "
        r"--output "
        r"rag-evaluation-starter\results"
        r"\parent_context_decomposed_hybrid_metrics.json"
    )


def main() -> None:
    run_evaluation()


if __name__ == "__main__":
    main()
