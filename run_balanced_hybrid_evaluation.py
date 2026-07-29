from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass

from langchain_core.documents import Document
from langchain_ollama import ChatOllama

from build_index import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
)
from rag_query import GENERATION_MODEL, generate_answer, stop_ollama_model
from run_hybrid_evaluation import (
    BM25_B,
    BM25_CANDIDATE_K,
    BM25_K1,
    DENSE_CANDIDATE_K,
    RRF_CONSTANT,
    BM25Index,
    FusedCandidate,
    chunk_key,
    load_frozen_chunks,
)
from run_ocr_evaluation import (
    DATABASE_DIR,
    GOLD_FILE,
    RESULTS_DIR,
    SOURCE_TO_DOC_ID,
    extract_citations,
    is_abstention,
    open_evaluation_vector_store,
    read_jsonl,
    validate_inputs,
)


# Controlled experiment: only the final hybrid selection policy changes.
# OCR, chunks, embeddings, candidate retrieval, RRF, prompt, generator, and
# final context size remain frozen.
FINAL_TOP_K = 3

PREDICTIONS_FILE = RESULTS_DIR / "balanced_hybrid_predictions.jsonl"
CONFIG_FILE = RESULTS_DIR / "balanced_hybrid_run_config.json"


def page_key(document: Document) -> tuple[str, int]:
    """Identify a physical source page so only one chunk per page survives."""
    return (
        str(document.metadata.get("source", "")),
        int(document.metadata.get("page", -1)),
    )


def reciprocal_rank_fusion_all(
    dense_results: list[tuple[Document, float]],
    bm25_results: list[tuple[Document, float]],
) -> list[FusedCandidate]:
    """Fuse all candidates without truncating before balanced selection."""
    candidates: dict[tuple[str, int, int], FusedCandidate] = {}

    for rank, (document, distance) in enumerate(dense_results, start=1):
        key = chunk_key(document)
        candidate = candidates.setdefault(
            key,
            FusedCandidate(document=document),
        )
        candidate.dense_rank = rank
        candidate.dense_distance = float(distance)
        candidate.rrf_score += 1.0 / (RRF_CONSTANT + rank)

    for rank, (document, score) in enumerate(bm25_results, start=1):
        key = chunk_key(document)
        candidate = candidates.setdefault(
            key,
            FusedCandidate(document=document),
        )
        candidate.bm25_rank = rank
        candidate.bm25_score = float(score)
        candidate.rrf_score += 1.0 / (RRF_CONSTANT + rank)

    return sorted(
        candidates.values(),
        key=lambda candidate: (
            -candidate.rrf_score,
            candidate.dense_rank
            if candidate.dense_rank is not None
            else math.inf,
            candidate.bm25_rank
            if candidate.bm25_rank is not None
            else math.inf,
            chunk_key(candidate.document),
        ),
    )


@dataclass(frozen=True)
class SelectedCandidate:
    candidate: FusedCandidate
    selection_role: str
    global_rrf_rank: int


def select_balanced_top_3(
    dense_results: list[tuple[Document, float]],
    bm25_results: list[tuple[Document, float]],
    fused_ranked: list[FusedCandidate],
) -> list[SelectedCandidate]:
    """
    Select one dense page, one different BM25 page, and one different RRF page.

    The page-level uniqueness guard prevents scarce top-3 context from being
    wasted on multiple chunks from the same physical page.
    """
    by_chunk = {
        chunk_key(candidate.document): candidate
        for candidate in fused_ranked
    }
    rrf_ranks = {
        chunk_key(candidate.document): rank
        for rank, candidate in enumerate(fused_ranked, start=1)
    }
    selected: list[SelectedCandidate] = []
    selected_pages: set[tuple[str, int]] = set()
    selected_chunks: set[tuple[str, int, int]] = set()

    def add(document: Document, role: str) -> bool:
        chunk = chunk_key(document)
        page = page_key(document)
        if chunk in selected_chunks or page in selected_pages:
            return False

        candidate = by_chunk[chunk]
        selected.append(
            SelectedCandidate(
                candidate=candidate,
                selection_role=role,
                global_rrf_rank=rrf_ranks[chunk],
            )
        )
        selected_chunks.add(chunk)
        selected_pages.add(page)
        return True

    # Slot 1: preserve dense retrieval's strongest semantic result.
    for document, _distance in dense_results:
        if add(document, "dense_anchor"):
            break

    # Slot 2: guarantee BM25's strongest lexical result from a different page.
    for document, _score in bm25_results:
        if add(document, "bm25_anchor"):
            break

    # Slot 3: use consensus ranking, still enforcing a third unique page.
    for candidate in fused_ranked:
        if add(candidate.document, "rrf_fill"):
            break

    if len(selected) != FINAL_TOP_K:
        raise RuntimeError(
            f"Balanced fusion selected {len(selected)} chunks; "
            f"expected {FINAL_TOP_K} unique pages."
        )

    return selected


def retrieve_balanced_hybrid(
    vector_store,
    bm25_index: BM25Index,
    question: str,
) -> tuple[
    list[tuple[Document, float]],
    list[SelectedCandidate],
    int,
    int,
]:
    dense_results = vector_store.similarity_search_with_score(
        query=question,
        k=DENSE_CANDIDATE_K,
    )
    bm25_results = bm25_index.search(
        query=question,
        k=BM25_CANDIDATE_K,
    )
    fused_ranked = reciprocal_rank_fusion_all(
        dense_results=dense_results,
        bm25_results=bm25_results,
    )
    selected = select_balanced_top_3(
        dense_results=dense_results,
        bm25_results=bm25_results,
        fused_ranked=fused_ranked,
    )
    generation_results = [
        (item.candidate.document, -item.candidate.rrf_score)
        for item in selected
    ]
    return (
        generation_results,
        selected,
        len(dense_results),
        len(bm25_results),
    )


def normalize_balanced_retrieved(
    selected: list[SelectedCandidate],
) -> list[dict]:
    normalized: list[dict] = []

    for rank, item in enumerate(selected, start=1):
        candidate = item.candidate
        document = candidate.document
        source = str(document.metadata.get("source", ""))
        if source not in SOURCE_TO_DOC_ID:
            raise ValueError(f"Unknown source filename in Chroma: {source}")

        normalized.append(
            {
                "rank": rank,
                "doc_id": SOURCE_TO_DOC_ID[source],
                "pages": [int(document.metadata["page"])],
                "chunk": int(document.metadata["chunk"]),
                "distance": (
                    float(candidate.dense_distance)
                    if candidate.dense_distance is not None
                    else None
                ),
                "hybrid_rrf_score": float(candidate.rrf_score),
                "dense_rank": candidate.dense_rank,
                "bm25_rank": candidate.bm25_rank,
                "bm25_score": candidate.bm25_score,
                "selection_role": item.selection_role,
                "global_rrf_rank": item.global_rrf_rank,
                "text": document.page_content.strip(),
            }
        )

    return normalized


def save_run_config(corpus_size: int) -> None:
    config = {
        "experiment_id": "ocr_balanced_hybrid_top_3",
        "independent_variable": "final hybrid selection policy",
        "control_experiment": "ocr_hybrid_rrf_top_3",
        "reused_index": str(DATABASE_DIR),
        "collection_name": COLLECTION_NAME,
        "corpus_chunks": corpus_size,
        "chunking": "RecursiveCharacterTextSplitter",
        "chunk_size_characters": CHUNK_SIZE,
        "chunk_overlap_characters": CHUNK_OVERLAP,
        "embedding_model": EMBEDDING_MODEL,
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
                "storage": "in memory from frozen OCR chunks",
            },
            "fusion": {
                "candidate_scoring": "equal-weight Reciprocal Rank Fusion",
                "rrf_constant": RRF_CONSTANT,
                "final_selection": [
                    "best dense candidate",
                    "best BM25 candidate from a different page",
                    "best remaining RRF candidate from a third page",
                ],
                "page_deduplication": True,
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
            "strategy": "reuse frozen selective-OCR index",
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

    rows = read_jsonl(PREDICTIONS_FILE)
    completed: dict[str, dict] = {}
    for row in rows:
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
    if not DATABASE_DIR.exists():
        raise FileNotFoundError(
            "The frozen OCR index was not found at "
            f"{DATABASE_DIR}. Do not rebuild it for this experiment."
        )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    gold_rows = read_jsonl(GOLD_FILE)
    completed = load_completed_predictions()
    gold_ids = {str(row["id"]) for row in gold_rows}
    extra_ids = sorted(set(completed) - gold_ids)
    if extra_ids:
        raise ValueError(
            f"Predictions contain IDs not present in the gold set: {extra_ids}"
        )

    vector_store = open_evaluation_vector_store()
    frozen_chunks = load_frozen_chunks(vector_store)
    bm25_index = BM25Index(frozen_chunks)
    save_run_config(corpus_size=len(frozen_chunks))

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
    print("Controlled experiment: balanced hybrid top-3 retrieval")
    print(f"Reusing frozen OCR index: {DATABASE_DIR.resolve()}")
    print(
        f"Candidates: dense top {DENSE_CANDIDATE_K} + "
        f"BM25 top {BM25_CANDIDATE_K}"
    )
    print(
        "Final selection: dense anchor + different BM25 page + "
        "different RRF page"
    )
    print(f"BM25 corpus: {len(frozen_chunks)} frozen OCR chunks")
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
                    selected,
                    dense_count,
                    bm25_count,
                ) = retrieve_balanced_hybrid(
                    vector_store=vector_store,
                    bm25_index=bm25_index,
                    question=question,
                )
                retrieval_ms = (
                    time.perf_counter() - retrieval_start
                ) * 1000

                if dense_count != DENSE_CANDIDATE_K:
                    raise RuntimeError(
                        f"Expected {DENSE_CANDIDATE_K} dense candidates, "
                        f"but received {dense_count}."
                    )
                if len(results) != FINAL_TOP_K:
                    raise RuntimeError(
                        f"Expected {FINAL_TOP_K} selected chunks, "
                        f"but received {len(results)}."
                    )
                if len({page_key(document) for document, _ in results}) != 3:
                    raise RuntimeError(
                        "Balanced fusion produced duplicate source pages."
                    )

                normalized_retrieved = normalize_balanced_retrieved(selected)

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
                    "retrieved": normalized_retrieved,
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
                    f"Candidates: dense={dense_count}, bm25={bm25_count}; "
                    f"selected chunks={len(results)}"
                )
                print(
                    "Selected: "
                    + ", ".join(
                        (
                            f"{rank}:{item.selection_role},"
                            f"dense={item.candidate.dense_rank or '-'},"
                            f"bm25={item.candidate.bm25_rank or '-'},"
                            f"rrf={item.global_rrf_rank}"
                        )
                        for rank, item in enumerate(selected, start=1)
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
    print("\nCalculate balanced-hybrid metrics with:")
    print(
        r"python rag-evaluation-starter\evaluation\metrics.py "
        r"--gold rag-evaluation-starter\evaluation\gold_questions.jsonl "
        r"--predictions "
        r"rag-evaluation-starter\results\balanced_hybrid_predictions.jsonl "
        r"--output "
        r"rag-evaluation-starter\results\balanced_hybrid_metrics.json"
    )


def main() -> None:
    run_evaluation()


if __name__ == "__main__":
    main()
