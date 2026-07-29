from __future__ import annotations

import json
import math
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from langchain_core.documents import Document

from build_index import CHUNK_OVERLAP, CHUNK_SIZE, EMBEDDING_MODEL
from rag_query import GENERATION_MODEL, stop_ollama_model
from run_balanced_hybrid_evaluation import (
    FINAL_TOP_K,
    page_key,
    reciprocal_rank_fusion_all,
    select_balanced_top_3,
)
from run_decomposed_hybrid_evaluation import (
    DecomposedSelection,
    decompose_question,
    retrieve_query,
    select_decomposed_top_3,
)
from run_hybrid_evaluation import (
    BM25_CANDIDATE_K,
    DENSE_CANDIDATE_K,
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
    open_evaluation_vector_store,
    read_jsonl,
    validate_inputs,
)


# Controlled Part 2 experiment:
# - Keep the frozen 1000/200 OCR index, mxbai embeddings, dense/BM25 candidate
#   retrieval, equal-weight RRF, deterministic both-and decomposition, and
#   three-page context budget fixed.
# - Preserve the first two selections made by the preferred pipeline.
# - Change only how the third unique page is selected: pure RRF relevance
#   (lambda=1.0 control) versus relevance/redundancy trade-offs.
#
# This is retrieval-only screening. It does not call the generation model.
LAMBDAS = (1.00, 0.85, 0.70, 0.50)
SWEEP_DIR = RESULTS_DIR / "diversity_reranking_sweep"
SUMMARY_FILE = SWEEP_DIR / "diversity_reranking_sweep_results.json"


@dataclass(frozen=True)
class ControlState:
    selected: list[FusedCandidate]
    selection_roles: list[str]
    candidate_pool: list[FusedCandidate]
    decomposed: bool
    subqueries: list[str]


@dataclass(frozen=True)
class RerankedThird:
    candidate: FusedCandidate
    pool_rank: int
    mmr_score: float
    normalized_relevance: float
    maximum_anchor_similarity: float


def lambda_name(value: float) -> str:
    return f"lambda_{value:.2f}".replace(".", "_")


def paths_for(value: float) -> dict[str, Path]:
    name = lambda_name(value)
    return {
        "predictions": SWEEP_DIR / f"{name}_retrieval.jsonl",
        "metrics": SWEEP_DIR / f"{name}_retrieval_metrics.json",
    }


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def validate_frozen_configuration() -> None:
    if CHUNK_SIZE != 1000 or CHUNK_OVERLAP != 200:
        raise RuntimeError(
            "This sweep requires the frozen 1000/200-character control, but "
            f"build_index.py reports {CHUNK_SIZE}/{CHUNK_OVERLAP}."
        )
    if "mxbai-embed-large" not in EMBEDDING_MODEL:
        raise RuntimeError(
            "This sweep requires the preferred mxbai-embed-large index, but "
            f"build_index.py reports {EMBEDDING_MODEL!r}."
        )
    if FINAL_TOP_K != 3:
        raise RuntimeError(
            f"This sweep requires a three-page context budget, not {FINAL_TOP_K}."
        )


def load_embedding_map(vector_store) -> dict[tuple[str, int, int], list[float]]:
    collection = vector_store.get(include=["metadatas", "embeddings"])
    metadatas = collection.get("metadatas")
    embeddings = collection.get("embeddings")
    if metadatas is None or embeddings is None:
        raise RuntimeError("The frozen Chroma index did not return embeddings.")
    if len(metadatas) != len(embeddings) or not metadatas:
        raise RuntimeError(
            "The frozen Chroma metadata and embedding counts do not match."
        )

    result: dict[tuple[str, int, int], list[float]] = {}
    for metadata, embedding in zip(metadatas, embeddings):
        document = Document(page_content="", metadata=dict(metadata or {}))
        key = chunk_key(document)
        if key in result:
            raise RuntimeError(f"Duplicate embedding key in Chroma: {key}")
        result[key] = [float(value) for value in embedding]
    return result


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right) or not left:
        raise ValueError("Embedding vectors must have equal non-zero length.")
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        raise ValueError("Cannot compare a zero-length embedding vector.")
    return max(-1.0, min(1.0, dot / (left_norm * right_norm)))


def retrieve_control_state(
    vector_store,
    bm25_index: BM25Index,
    question: str,
) -> ControlState:
    subqueries = decompose_question(question)

    if len(subqueries) == 1:
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
        selected_items = select_balanced_top_3(
            dense_results=dense_results,
            bm25_results=bm25_results,
            fused_ranked=fused_ranked,
        )
        return ControlState(
            selected=[item.candidate for item in selected_items],
            selection_roles=[item.selection_role for item in selected_items],
            candidate_pool=fused_ranked,
            decomposed=False,
            subqueries=[],
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
    selected_items: list[DecomposedSelection] = select_decomposed_top_3(
        subquery_runs=subquery_runs,
        full_question_run=full_question_run,
    )
    return ControlState(
        selected=[item.candidate for item in selected_items],
        selection_roles=[item.selection_role for item in selected_items],
        candidate_pool=full_question_run.fused_ranked,
        decomposed=True,
        subqueries=subqueries,
    )


def rerank_third_slot(
    anchors: list[FusedCandidate],
    candidate_pool: list[FusedCandidate],
    embedding_map: dict[tuple[str, int, int], list[float]],
    lambda_value: float,
) -> RerankedThird:
    if len(anchors) != 2:
        raise ValueError("Exactly two frozen anchors are required.")

    selected_pages = {page_key(item.document) for item in anchors}
    anchor_vectors = [
        embedding_map[chunk_key(item.document)]
        for item in anchors
    ]
    maximum_rrf = max(
        (candidate.rrf_score for candidate in candidate_pool),
        default=0.0,
    )
    if maximum_rrf <= 0.0:
        raise RuntimeError("The fused candidate pool has no positive RRF score.")

    scored: list[tuple[float, int, tuple[str, int, int], RerankedThird]] = []
    seen_pages = set(selected_pages)
    for pool_rank, candidate in enumerate(candidate_pool, start=1):
        page = page_key(candidate.document)
        if page in seen_pages:
            continue
        seen_pages.add(page)

        vector = embedding_map[chunk_key(candidate.document)]
        maximum_similarity = max(
            cosine_similarity(vector, anchor_vector)
            for anchor_vector in anchor_vectors
        )
        # Negative similarity does not represent redundancy.
        redundancy = max(0.0, maximum_similarity)
        normalized_relevance = candidate.rrf_score / maximum_rrf
        mmr_score = (
            lambda_value * normalized_relevance
            - (1.0 - lambda_value) * redundancy
        )
        item = RerankedThird(
            candidate=candidate,
            pool_rank=pool_rank,
            mmr_score=mmr_score,
            normalized_relevance=normalized_relevance,
            maximum_anchor_similarity=maximum_similarity,
        )
        scored.append(
            (
                -mmr_score,
                pool_rank,
                chunk_key(candidate.document),
                item,
            )
        )

    if not scored:
        raise RuntimeError(
            "No third candidate from a unique physical page was available."
        )
    scored.sort(key=lambda row: row[:3])
    return scored[0][3]


def normalized_retrieval(
    candidates: list[FusedCandidate],
    roles: list[str],
) -> list[dict]:
    if len(candidates) != FINAL_TOP_K or len(roles) != FINAL_TOP_K:
        raise ValueError("Expected exactly three candidates and three roles.")

    normalized: list[dict] = []
    for rank, (candidate, role) in enumerate(
        zip(candidates, roles),
        start=1,
    ):
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
                "hybrid_rrf_score": float(candidate.rrf_score),
                "dense_rank": candidate.dense_rank,
                "bm25_rank": candidate.bm25_rank,
                "selection_role": role,
            }
        )
    return normalized


def pairwise_similarity(
    candidates: list[FusedCandidate],
    embedding_map: dict[tuple[str, int, int], list[float]],
) -> float:
    vectors = [
        embedding_map[chunk_key(candidate.document)]
        for candidate in candidates
    ]
    values = [
        cosine_similarity(vectors[left], vectors[right])
        for left in range(len(vectors))
        for right in range(left + 1, len(vectors))
    ]
    return statistics.fmean(values)


def evidence_units(items: Iterable[dict]) -> set[tuple[str, int]]:
    units: set[tuple[str, int]] = set()
    for item in items:
        doc_id = str(item["doc_id"])
        pages = item.get("pages")
        if pages is None and "page" in item:
            pages = [item["page"]]
        for page in pages or []:
            units.add((doc_id, int(page)))
    return units


def score_one(retrieved: list[dict], gold_evidence: list[dict]) -> dict[str, float]:
    gold_units = evidence_units(gold_evidence)
    seen_gold: set[tuple[str, int]] = set()
    first_relevant_rank: int | None = None
    for rank, result in enumerate(retrieved[:FINAL_TOP_K], start=1):
        relevant = evidence_units([result]) & gold_units
        if relevant and first_relevant_rank is None:
            first_relevant_rank = rank
        seen_gold.update(relevant)
    return {
        "hit_at_3": float(bool(seen_gold)),
        "recall_at_3": len(seen_gold) / len(gold_units),
        "mrr_at_3": (
            0.0 if first_relevant_rank is None else 1.0 / first_relevant_rank
        ),
    }


def aggregate_metrics(
    gold_rows: list[dict],
    predictions: list[dict],
    lambda_value: float,
) -> dict:
    by_id = {str(row["id"]): row for row in predictions}
    if set(by_id) != {str(row["id"]) for row in gold_rows}:
        raise ValueError("Retrieval prediction IDs do not match gold IDs.")

    values = {"hit_at_3": [], "recall_at_3": [], "mrr_at_3": []}
    by_type: dict[str, dict[str, list[float]]] = {}
    similarities: list[float] = []
    third_slot_changes = 0

    for gold in gold_rows:
        prediction = by_id[str(gold["id"])]
        similarities.append(float(prediction["mean_pairwise_similarity"]))
        third_slot_changes += int(bool(prediction["third_slot_changed"]))
        if not bool(gold["answerable"]):
            continue
        scores = score_one(
            prediction["retrieved"],
            gold["gold_evidence"],
        )
        question_type = str(gold["type"])
        type_values = by_type.setdefault(
            question_type,
            {"hit_at_3": [], "recall_at_3": [], "mrr_at_3": []},
        )
        for name, value in scores.items():
            values[name].append(value)
            type_values[name].append(value)

    return {
        "config": {
            "name": lambda_name(lambda_value),
            "lambda": lambda_value,
            "selection": (
                "frozen first two anchors plus diversity-reranked third page"
            ),
        },
        "retrieval": {
            name: statistics.fmean(metric_values)
            for name, metric_values in values.items()
        },
        "diversity": {
            "mean_pairwise_cosine_similarity": statistics.fmean(similarities),
            "third_slot_changes_vs_control": third_slot_changes,
        },
        "by_question_type": {
            question_type: {
                name: statistics.fmean(metric_values)
                for name, metric_values in type_values.items()
            }
            for question_type, type_values in sorted(by_type.items())
        },
        "answer_generation_performed": False,
    }


def run_sweep() -> None:
    validate_inputs()
    validate_frozen_configuration()
    if not DATABASE_DIR.exists():
        raise FileNotFoundError(
            f"Frozen OCR index not found at {DATABASE_DIR}."
        )

    SWEEP_DIR.mkdir(parents=True, exist_ok=True)
    gold_rows = read_jsonl(GOLD_FILE)
    vector_store = open_evaluation_vector_store()
    frozen_chunks = load_frozen_chunks(vector_store)
    bm25_index = BM25Index(frozen_chunks)
    embedding_map = load_embedding_map(vector_store)
    if len(embedding_map) != len(frozen_chunks):
        raise RuntimeError(
            "The frozen chunk and embedding counts do not match: "
            f"{len(frozen_chunks)} versus {len(embedding_map)}."
        )

    predictions_by_lambda: dict[float, list[dict]] = {
        value: [] for value in LAMBDAS
    }
    print("Controlled experiment: diversity-aware third-slot reranking")
    print(f"Reusing frozen OCR index: {DATABASE_DIR.resolve()}")
    print(
        f"Frozen pipeline: {CHUNK_SIZE}/{CHUNK_OVERLAP} characters, "
        f"{EMBEDDING_MODEL}, dense + BM25 RRF, decomposition, top 3"
    )
    print(f"Reranking lambdas: {', '.join(str(value) for value in LAMBDAS)}")
    print("Generation: disabled during retrieval screening")

    try:
        for position, gold in enumerate(gold_rows, start=1):
            question = str(gold["question"])
            question_start = time.perf_counter()
            state = retrieve_control_state(
                vector_store=vector_store,
                bm25_index=bm25_index,
                question=question,
            )
            retrieval_ms = (time.perf_counter() - question_start) * 1000
            anchors = state.selected[:2]
            control_third_key = chunk_key(state.selected[2].document)

            for lambda_value in LAMBDAS:
                reranking_start = time.perf_counter()
                third = rerank_third_slot(
                    anchors=anchors,
                    candidate_pool=state.candidate_pool,
                    embedding_map=embedding_map,
                    lambda_value=lambda_value,
                )
                selected = [*anchors, third.candidate]
                selected_pages = {page_key(item.document) for item in selected}
                if len(selected_pages) != FINAL_TOP_K:
                    raise RuntimeError("Reranking produced duplicate pages.")

                if lambda_value == 1.0:
                    actual_key = chunk_key(third.candidate.document)
                    if actual_key != control_third_key:
                        raise RuntimeError(
                            "Lambda 1.0 failed to reproduce the frozen "
                            f"control third slot: {actual_key} != "
                            f"{control_third_key}."
                        )

                predictions_by_lambda[lambda_value].append(
                    {
                        "id": str(gold["id"]),
                        "question": question,
                        "retrieved": normalized_retrieval(
                            candidates=selected,
                            roles=[
                                state.selection_roles[0],
                                state.selection_roles[1],
                                (
                                    "rrf_control_third"
                                    if lambda_value == 1.0
                                    else "diversity_reranked_third"
                                ),
                            ],
                        ),
                        "decomposition": {
                            "applied": state.decomposed,
                            "subqueries": state.subqueries,
                        },
                        "lambda": lambda_value,
                        "third_slot": {
                            "pool_rank": third.pool_rank,
                            "mmr_score": third.mmr_score,
                            "normalized_relevance": (
                                third.normalized_relevance
                            ),
                            "maximum_anchor_similarity": (
                                third.maximum_anchor_similarity
                            ),
                        },
                        "third_slot_changed": (
                            chunk_key(third.candidate.document)
                            != control_third_key
                        ),
                        "mean_pairwise_similarity": pairwise_similarity(
                            selected,
                            embedding_map,
                        ),
                        "retrieval_ms": round(retrieval_ms, 3),
                        "reranking_ms": round(
                            (time.perf_counter() - reranking_start) * 1000,
                            3,
                        ),
                    }
                )

            print(
                f"  [{position:02d}/{len(gold_rows)}] {gold['id']} "
                f"{(time.perf_counter() - question_start) * 1000:.1f} ms"
            )
    finally:
        stop_ollama_model(EMBEDDING_MODEL)
        stop_ollama_model(GENERATION_MODEL)

    summaries: list[dict] = []
    for lambda_value in LAMBDAS:
        rows = predictions_by_lambda[lambda_value]
        metrics = aggregate_metrics(
            gold_rows=gold_rows,
            predictions=rows,
            lambda_value=lambda_value,
        )
        write_jsonl(paths_for(lambda_value)["predictions"], rows)
        write_json(paths_for(lambda_value)["metrics"], metrics)
        summaries.append(metrics)

    control = next(
        item for item in summaries if item["config"]["lambda"] == 1.0
    )
    control_retrieval = control["retrieval"]
    eligible = [
        item
        for item in summaries
        if (
            item["retrieval"]["hit_at_3"]
            >= control_retrieval["hit_at_3"]
            and item["retrieval"]["recall_at_3"]
            >= control_retrieval["recall_at_3"]
        )
    ]
    ranked = sorted(
        eligible,
        key=lambda item: (
            -item["retrieval"]["recall_at_3"],
            -item["retrieval"]["hit_at_3"],
            -item["retrieval"]["mrr_at_3"],
            item["diversity"]["mean_pairwise_cosine_similarity"],
            -item["config"]["lambda"],
        ),
    )
    winner = ranked[0]
    winner_is_challenger = winner["config"]["lambda"] != 1.0

    result = {
        "experiment": "Part 2 diversity-aware third-slot reranking sweep",
        "independent_variable": "MMR relevance-diversity lambda",
        "controls": {
            "ocr": "frozen selective-OCR index",
            "chunk_size_characters": CHUNK_SIZE,
            "chunk_overlap_characters": CHUNK_OVERLAP,
            "embedding_model": EMBEDDING_MODEL,
            "retrieval": (
                f"dense top {DENSE_CANDIDATE_K} plus BM25 top "
                f"{BM25_CANDIDATE_K}, equal-weight RRF, "
                "deterministic both-and query decomposition"
            ),
            "frozen_slots": (
                "first two pages from the preferred decomposed balanced "
                "hybrid pipeline"
            ),
            "final_top_k": FINAL_TOP_K,
            "generation": "not run during screening",
        },
        "eligibility_rule": (
            "A challenger must not reduce control Hit@3 or Recall@3."
        ),
        "ranking_rule": (
            "Recall@3, Hit@3, MRR@3, then lower pairwise similarity."
        ),
        "results": summaries,
        "provisional_winner": winner["config"]["name"],
        "winner_is_diversity_challenger": winner_is_challenger,
        "next_step": (
            "Run one full 30-question generation confirmation for the "
            "winning challenger."
            if winner_is_challenger
            else "Reject diversity reranking and keep the frozen control."
        ),
    }
    write_json(SUMMARY_FILE, result)

    print("\nDiversity reranking sweep complete")
    for item in summaries:
        retrieval = item["retrieval"]
        diversity = item["diversity"]
        print(
            f"  {item['config']['name']}: "
            f"hit@3={retrieval['hit_at_3']:.4f}, "
            f"recall@3={retrieval['recall_at_3']:.4f}, "
            f"mrr@3={retrieval['mrr_at_3']:.4f}, "
            "mean_similarity="
            f"{diversity['mean_pairwise_cosine_similarity']:.4f}, "
            "third_slot_changes="
            f"{diversity['third_slot_changes_vs_control']}"
        )
    print(f"Provisional winner: {winner['config']['name']}")
    print(f"Summary saved to: {SUMMARY_FILE.resolve()}")


def main() -> None:
    run_sweep()


if __name__ == "__main__":
    main()
