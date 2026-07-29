from __future__ import annotations

import json
import re
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
from run_balanced_hybrid_evaluation import (
    FINAL_TOP_K,
    normalize_balanced_retrieved,
    page_key,
    reciprocal_rank_fusion_all,
    retrieve_balanced_hybrid,
)
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


# Controlled experiment: only explicit "both X and Y" questions receive
# deterministic query decomposition. The frozen OCR index, candidate
# retrievers, RRF formula, final top-k, prompt, and generator remain unchanged.
PREDICTIONS_FILE = RESULTS_DIR / "decomposed_hybrid_predictions.jsonl"
CONFIG_FILE = RESULTS_DIR / "decomposed_hybrid_run_config.json"

BOTH_PATTERN = re.compile(
    r"^(?P<stem>.+?)\bboth\s+"
    r"(?P<left>.+?)\s+and\s+"
    r"(?P<right>.+?)[?.!]*$",
    flags=re.IGNORECASE,
)


def clean_clause(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip(" \t\r\n,;:.?!")


def decompose_question(question: str) -> list[str]:
    """
    Split an explicit "both X and Y" question into two standalone queries.

    This is deliberately deterministic and label-blind: it does not inspect
    question IDs, benchmark types, reference answers, or gold evidence.
    Questions without the explicit coordinator are returned unchanged.
    """
    normalized = re.sub(r"\s+", " ", question).strip()
    match = BOTH_PATTERN.match(normalized)
    if match is None:
        return [normalized]

    stem = match.group("stem").strip()
    left = clean_clause(match.group("left"))
    right = clean_clause(match.group("right"))
    if not stem or len(left.split()) < 2 or len(right.split()) < 2:
        return [normalized]

    return [
        f"{stem} {left}?",
        f"{stem} {right}?",
    ]


@dataclass(frozen=True)
class QueryRun:
    query_index: int
    query: str
    dense_count: int
    bm25_count: int
    fused_ranked: list[FusedCandidate]


@dataclass(frozen=True)
class DecomposedSelection:
    candidate: FusedCandidate
    selection_role: str
    query_index: int
    query: str
    query_rrf_rank: int


def retrieve_query(
    vector_store,
    bm25_index: BM25Index,
    query: str,
    query_index: int,
) -> QueryRun:
    dense_results = vector_store.similarity_search_with_score(
        query=query,
        k=DENSE_CANDIDATE_K,
    )
    bm25_results = bm25_index.search(
        query=query,
        k=BM25_CANDIDATE_K,
    )
    if len(dense_results) != DENSE_CANDIDATE_K:
        raise RuntimeError(
            f"Expected {DENSE_CANDIDATE_K} dense candidates for query "
            f"{query_index}, but received {len(dense_results)}."
        )

    return QueryRun(
        query_index=query_index,
        query=query,
        dense_count=len(dense_results),
        bm25_count=len(bm25_results),
        fused_ranked=reciprocal_rank_fusion_all(
            dense_results=dense_results,
            bm25_results=bm25_results,
        ),
    )


def select_decomposed_top_3(
    subquery_runs: list[QueryRun],
    full_question_run: QueryRun,
) -> list[DecomposedSelection]:
    """
    Reserve one unique page for each subquery, then fill from the full query.

    The fallback over all subquery rankings is only used if the full-question
    candidate pool cannot supply three unique physical pages.
    """
    selected: list[DecomposedSelection] = []
    selected_pages: set[tuple[str, int]] = set()
    selected_chunks: set[tuple[str, int, int]] = set()

    def add(
        candidate: FusedCandidate,
        role: str,
        query_run: QueryRun,
        rank: int,
    ) -> bool:
        chunk = chunk_key(candidate.document)
        page = page_key(candidate.document)
        if chunk in selected_chunks or page in selected_pages:
            return False

        selected.append(
            DecomposedSelection(
                candidate=candidate,
                selection_role=role,
                query_index=query_run.query_index,
                query=query_run.query,
                query_rrf_rank=rank,
            )
        )
        selected_chunks.add(chunk)
        selected_pages.add(page)
        return True

    for run in subquery_runs:
        for rank, candidate in enumerate(run.fused_ranked, start=1):
            if add(
                candidate=candidate,
                role=f"subquery_{run.query_index}",
                query_run=run,
                rank=rank,
            ):
                break

    for rank, candidate in enumerate(
        full_question_run.fused_ranked,
        start=1,
    ):
        if len(selected) == FINAL_TOP_K:
            break
        add(
            candidate=candidate,
            role="full_question_fill",
            query_run=full_question_run,
            rank=rank,
        )

    if len(selected) < FINAL_TOP_K:
        for run in subquery_runs:
            for rank, candidate in enumerate(run.fused_ranked, start=1):
                if len(selected) == FINAL_TOP_K:
                    break
                add(
                    candidate=candidate,
                    role="subquery_fallback",
                    query_run=run,
                    rank=rank,
                )

    if len(selected) != FINAL_TOP_K:
        raise RuntimeError(
            f"Decomposed retrieval selected {len(selected)} chunks; "
            f"expected {FINAL_TOP_K} unique pages."
        )

    return selected


def retrieve_with_decomposition(
    vector_store,
    bm25_index: BM25Index,
    question: str,
) -> tuple[
    list[tuple[Document, float]],
    list[DecomposedSelection] | None,
    list,
    list[str],
    list[QueryRun],
]:
    subqueries = decompose_question(question)

    # Preserve the balanced-hybrid control exactly for non-decomposed queries.
    if len(subqueries) == 1:
        (
            results,
            balanced_selected,
            dense_count,
            bm25_count,
        ) = retrieve_balanced_hybrid(
            vector_store=vector_store,
            bm25_index=bm25_index,
            question=question,
        )
        query_run = QueryRun(
            query_index=0,
            query=question,
            dense_count=dense_count,
            bm25_count=bm25_count,
            fused_ranked=[],
        )
        return (
            results,
            None,
            balanced_selected,
            subqueries,
            [query_run],
        )

    subquery_runs = [
        retrieve_query(
            vector_store=vector_store,
            bm25_index=bm25_index,
            query=subquery,
            query_index=index,
        )
        for index, subquery in enumerate(subqueries, start=1)
    ]
    full_question_run = retrieve_query(
        vector_store=vector_store,
        bm25_index=bm25_index,
        query=question,
        query_index=0,
    )
    selected = select_decomposed_top_3(
        subquery_runs=subquery_runs,
        full_question_run=full_question_run,
    )
    results = [
        (item.candidate.document, -item.candidate.rrf_score)
        for item in selected
    ]
    return (
        results,
        selected,
        [],
        subqueries,
        [*subquery_runs, full_question_run],
    )


def normalize_decomposed_retrieved(
    selected: list[DecomposedSelection],
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
                "retrieval_query_index": item.query_index,
                "retrieval_query": item.query,
                "query_rrf_rank": item.query_rrf_rank,
                "text": document.page_content.strip(),
            }
        )

    return normalized


def save_run_config(corpus_size: int) -> None:
    config = {
        "experiment_id": "ocr_decomposed_balanced_hybrid_top_3",
        "independent_variable": (
            "deterministic query decomposition for explicit both-and questions"
        ),
        "control_experiment": "ocr_balanced_hybrid_top_3",
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
                "decomposition_trigger": "explicit both X and Y syntax",
                "decomposed_selection": [
                    "best RRF page for first subquery",
                    "best RRF page for second subquery",
                    "best remaining full-question RRF page",
                ],
                "non_decomposed_selection": (
                    "unchanged balanced-hybrid top-3 policy"
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
    print("Controlled experiment: query-decomposed balanced hybrid top-3")
    print(f"Reusing frozen OCR index: {DATABASE_DIR.resolve()}")
    print(
        "Decomposition trigger: explicit 'both X and Y' questions only"
    )
    print(
        "Decomposed selection: one unique page per subquery + "
        "full-question RRF fill"
    )
    print(
        "All other questions: unchanged balanced-hybrid retrieval"
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
                if len({page_key(document) for document, _ in results}) != 3:
                    raise RuntimeError(
                        "Query-decomposed fusion produced duplicate pages."
                    )

                if decomposed_selected is None:
                    normalized_retrieved = normalize_balanced_retrieved(
                        balanced_selected
                    )
                else:
                    normalized_retrieved = normalize_decomposed_retrieved(
                        decomposed_selected
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
                    "decomposition": {
                        "applied": len(subqueries) > 1,
                        "subqueries": (
                            subqueries if len(subqueries) > 1 else []
                        ),
                    },
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
                if decomposed_selected is not None:
                    print(
                        "Selected: "
                        + ", ".join(
                            (
                                f"{rank}:{item.selection_role},"
                                f"query={item.query_index},"
                                f"rrf={item.query_rrf_rank}"
                            )
                            for rank, item in enumerate(
                                decomposed_selected,
                                start=1,
                            )
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
    print("\nCalculate query-decomposition metrics with:")
    print(
        r"python rag-evaluation-starter\evaluation\metrics.py "
        r"--gold rag-evaluation-starter\evaluation\gold_questions.jsonl "
        r"--predictions "
        r"rag-evaluation-starter\results\decomposed_hybrid_predictions.jsonl "
        r"--output "
        r"rag-evaluation-starter\results\decomposed_hybrid_metrics.json"
    )


def main() -> None:
    run_evaluation()


if __name__ == "__main__":
    main()
