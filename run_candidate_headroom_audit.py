from __future__ import annotations

import json
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from langchain_core.documents import Document

from build_index import CHUNK_OVERLAP, CHUNK_SIZE, EMBEDDING_MODEL
from rag_query import GENERATION_MODEL, stop_ollama_model
from run_balanced_hybrid_evaluation import FINAL_TOP_K, page_key
from run_decomposed_hybrid_evaluation import QueryRun, retrieve_query
from run_diversity_reranking_sweep import (
    ControlState,
    retrieve_control_state,
)
from run_hybrid_evaluation import (
    BM25_CANDIDATE_K,
    DENSE_CANDIDATE_K,
    BM25Index,
    FusedCandidate,
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


# Diagnostic experiment:
# - Keep the frozen selective-OCR index, 1000/200 chunks, mxbai embeddings,
#   dense/BM25 retrieval, equal-weight RRF, explicit both-and decomposition,
#   and final top-3 control unchanged.
# - Inspect the top unique physical pages already present in each frozen query
#   run at cutoffs 3, 5, 10, and 20.
# - Use gold evidence only after retrieval to measure candidate availability.
#   Gold pages never affect queries, retrieval, ranking, or selection.
AUDIT_DIR = RESULTS_DIR / "candidate_headroom_audit"
DETAIL_FILE = AUDIT_DIR / "candidate_headroom_by_question.jsonl"
SUMMARY_FILE = AUDIT_DIR / "candidate_headroom_summary.json"

CUTOFFS = (3, 5, 10, 20)
MAX_PAGE_CANDIDATES = max(CUTOFFS)


@dataclass(frozen=True)
class PageCandidate:
    page_rank: int
    fused_chunk_rank: int
    candidate: FusedCandidate


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
            "This audit requires the frozen 1000/200-character index, but "
            f"build_index.py reports {CHUNK_SIZE}/{CHUNK_OVERLAP}."
        )
    if "mxbai-embed-large" not in EMBEDDING_MODEL:
        raise RuntimeError(
            "This audit requires the preferred mxbai-embed-large index, but "
            f"build_index.py reports {EMBEDDING_MODEL!r}."
        )
    if FINAL_TOP_K != 3:
        raise RuntimeError(
            f"This audit requires the frozen top-3 control, not top "
            f"{FINAL_TOP_K}."
        )
    if DENSE_CANDIDATE_K < MAX_PAGE_CANDIDATES:
        raise RuntimeError(
            "Dense retrieval must expose at least 20 candidates for this audit."
        )
    if BM25_CANDIDATE_K < MAX_PAGE_CANDIDATES:
        raise RuntimeError(
            "BM25 retrieval must expose at least 20 candidates for this audit."
        )


def source_to_doc_id(document: Document) -> str:
    source = str(document.metadata.get("source", ""))
    if source not in SOURCE_TO_DOC_ID:
        raise ValueError(f"Unknown source filename in Chroma: {source}")
    return SOURCE_TO_DOC_ID[source]


def candidate_unit(candidate: FusedCandidate) -> tuple[str, int]:
    return (
        source_to_doc_id(candidate.document),
        int(candidate.document.metadata["page"]),
    )


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


def unique_page_pool(
    fused_ranked: list[FusedCandidate],
    limit: int = MAX_PAGE_CANDIDATES,
) -> list[PageCandidate]:
    """
    Collapse chunk-level RRF output to unique physical pages.

    The first chunk for a page is its highest-ranked RRF chunk, so this is a
    deterministic page-level view of the unchanged candidate ranking.
    """
    pool: list[PageCandidate] = []
    seen_pages: set[tuple[str, int]] = set()
    for fused_chunk_rank, candidate in enumerate(fused_ranked, start=1):
        physical_page = page_key(candidate.document)
        if physical_page in seen_pages:
            continue
        seen_pages.add(physical_page)
        pool.append(
            PageCandidate(
                page_rank=len(pool) + 1,
                fused_chunk_rank=fused_chunk_rank,
                candidate=candidate,
            )
        )
        if len(pool) == limit:
            break
    return pool


def normalize_page_candidate(item: PageCandidate) -> dict:
    candidate = item.candidate
    document = candidate.document
    return {
        "page_rank": item.page_rank,
        "fused_chunk_rank": item.fused_chunk_rank,
        "doc_id": source_to_doc_id(document),
        "page": int(document.metadata["page"]),
        "chunk": int(document.metadata["chunk"]),
        "hybrid_rrf_score": float(candidate.rrf_score),
        "dense_rank": candidate.dense_rank,
        "bm25_rank": candidate.bm25_rank,
    }


def normalize_control_top_3(state: ControlState) -> list[dict]:
    if len(state.selected) != FINAL_TOP_K:
        raise RuntimeError(
            f"Frozen control selected {len(state.selected)} pages; expected "
            f"{FINAL_TOP_K}."
        )
    rows: list[dict] = []
    seen: set[tuple[str, int]] = set()
    for rank, (candidate, role) in enumerate(
        zip(state.selected, state.selection_roles),
        start=1,
    ):
        unit = candidate_unit(candidate)
        if unit in seen:
            raise RuntimeError("Frozen control produced a duplicate page.")
        seen.add(unit)
        rows.append(
            {
                "rank": rank,
                "doc_id": unit[0],
                "pages": [unit[1]],
                "chunk": int(candidate.document.metadata["chunk"]),
                "selection_role": role,
                "hybrid_rrf_score": float(candidate.rrf_score),
                "dense_rank": candidate.dense_rank,
                "bm25_rank": candidate.bm25_rank,
            }
        )
    return rows


def frozen_query_runs(
    vector_store,
    bm25_index: BM25Index,
    question: str,
    control: ControlState,
) -> list[QueryRun]:
    """
    Return candidate-producing query runs from the frozen control.

    The control already provides its full-question pool. For the small subset
    using explicit both-and decomposition, reproduce only the existing
    subquery runs so their deeper candidates can also be audited.
    """
    full_question = QueryRun(
        query_index=0,
        query=question,
        dense_count=sum(
            candidate.dense_rank is not None
            for candidate in control.candidate_pool
        ),
        bm25_count=sum(
            candidate.bm25_rank is not None
            for candidate in control.candidate_pool
        ),
        fused_ranked=control.candidate_pool,
    )
    if not control.decomposed:
        return [full_question]

    subquery_runs = [
        retrieve_query(
            vector_store=vector_store,
            bm25_index=bm25_index,
            query=subquery,
            query_index=index,
        )
        for index, subquery in enumerate(control.subqueries, start=1)
    ]
    return [*subquery_runs, full_question]


def query_role(run: QueryRun) -> str:
    return "full_question" if run.query_index == 0 else "frozen_subquery"


def best_candidate_match(
    unit: tuple[str, int],
    pools: list[tuple[QueryRun, list[PageCandidate]]],
) -> dict | None:
    matches: list[tuple[int, int, int, QueryRun, PageCandidate]] = []
    for run_order, (run, pool) in enumerate(pools):
        for item in pool:
            if candidate_unit(item.candidate) == unit:
                query_priority = 1 if run.query_index == 0 else 0
                matches.append(
                    (
                        item.page_rank,
                        query_priority,
                        run_order,
                        run,
                        item,
                    )
                )
                break
    if not matches:
        return None

    _rank, _query_priority, _run_order, run, item = min(matches)
    return {
        "page_rank": item.page_rank,
        "fused_chunk_rank": item.fused_chunk_rank,
        "query_index": run.query_index,
        "query_role": query_role(run),
        "query": run.query,
        "chunk": int(item.candidate.document.metadata["chunk"]),
        "hybrid_rrf_score": float(item.candidate.rrf_score),
        "dense_rank": item.candidate.dense_rank,
        "bm25_rank": item.candidate.bm25_rank,
    }


def control_rank(
    unit: tuple[str, int],
    control_top_3: list[dict],
) -> int | None:
    for row in control_top_3:
        row_unit = (str(row["doc_id"]), int(row["pages"][0]))
        if row_unit == unit:
            return int(row["rank"])
    return None


def audit_gold_after_retrieval(
    gold_evidence: list[dict],
    control_top_3: list[dict],
    pools: list[tuple[QueryRun, list[PageCandidate]]],
) -> list[dict]:
    """
    Evaluate retrieval after all candidate generation is complete.

    Keeping this function separate makes the leakage boundary explicit: only
    this post-retrieval diagnostic receives gold evidence.
    """
    rows: list[dict] = []
    for doc_id, page in sorted(evidence_units(gold_evidence)):
        unit = (doc_id, page)
        selected_rank = control_rank(unit, control_top_3)
        match = best_candidate_match(unit, pools)
        rows.append(
            {
                "doc_id": doc_id,
                "page": page,
                "in_control_top_3": selected_rank is not None,
                "control_rank": selected_rank,
                "best_candidate_match": match,
                "available_by_cutoff": {
                    f"top_{cutoff}": (
                        match is not None
                        and int(match["page_rank"]) <= cutoff
                    )
                    for cutoff in CUTOFFS
                },
                "diagnosis": (
                    "already_selected"
                    if selected_rank is not None
                    else (
                        "retrieved_below_final_cutoff"
                        if match is not None
                        else "absent_from_all_top_20_query_pools"
                    )
                ),
            }
        )
    return rows


def score_control(
    control_top_3: list[dict],
    gold_evidence: list[dict],
) -> dict[str, float]:
    gold = evidence_units(gold_evidence)
    found: set[tuple[str, int]] = set()
    first_relevant_rank: int | None = None
    for rank, item in enumerate(control_top_3[:FINAL_TOP_K], start=1):
        relevant = evidence_units([item]) & gold
        if relevant and first_relevant_rank is None:
            first_relevant_rank = rank
        found.update(relevant)
    return {
        "hit_at_3": float(bool(found)),
        "recall_at_3": len(found) / len(gold),
        "mrr_at_3": (
            0.0 if first_relevant_rank is None else 1.0 / first_relevant_rank
        ),
    }


def score_candidate_availability(
    gold_audit: list[dict],
) -> dict[str, float]:
    if not gold_audit:
        raise ValueError("Candidate availability requires gold evidence.")
    return {
        f"recall_at_{cutoff}": (
            sum(
                int(bool(item["available_by_cutoff"][f"top_{cutoff}"]))
                for item in gold_audit
            )
            / len(gold_audit)
        )
        for cutoff in CUTOFFS
    }


def mean_metrics(rows: list[dict], field: str) -> dict[str, float]:
    names = sorted(rows[0][field]) if rows else []
    return {
        name: statistics.fmean(float(row[field][name]) for row in rows)
        for name in names
    }


def aggregate_summary(reference_rows: list[dict], audits: list[dict]) -> dict:
    by_id = {str(row["id"]): row for row in audits}
    if set(by_id) != {str(row["id"]) for row in reference_rows}:
        raise ValueError("Audit IDs do not match gold reference IDs.")

    answerable_rows = [
        by_id[str(reference["id"])]
        for reference in reference_rows
        if bool(reference["answerable"])
    ]
    if not answerable_rows:
        raise RuntimeError("The benchmark contains no answerable questions.")

    by_type: dict[str, list[dict]] = {}
    for reference in reference_rows:
        if not bool(reference["answerable"]):
            continue
        by_type.setdefault(str(reference["type"]), []).append(
            by_id[str(reference["id"])]
        )

    missing_units: list[dict] = []
    for row in answerable_rows:
        for item in row["gold_evidence_audit"]:
            if not bool(item["in_control_top_3"]):
                missing_units.append(
                    {
                        "id": row["id"],
                        "question_type": row["type"],
                        "doc_id": item["doc_id"],
                        "page": item["page"],
                        "diagnosis": item["diagnosis"],
                        "best_candidate_match": item["best_candidate_match"],
                    }
                )

    rerankable = [
        item
        for item in missing_units
        if item["diagnosis"] == "retrieved_below_final_cutoff"
    ]
    absent = [
        item
        for item in missing_units
        if item["diagnosis"] == "absent_from_all_top_20_query_pools"
    ]

    if not missing_units:
        recommendation = (
            "No retrieval repair is needed on this benchmark; investigate "
            "generation and citation quality instead."
        )
        next_experiment = "structured_evidence_checked_generation"
    elif len(rerankable) == len(missing_units):
        recommendation = (
            "All top-3 misses already exist in the frozen candidate pools; "
            "test evidence-aware reranking before query rewriting."
        )
        next_experiment = "evidence_aware_reranking"
    elif rerankable:
        recommendation = (
            "The failure is mixed: rerank candidate-present misses and use one "
            "controlled corpus-terminology rewrite only for candidate-absent "
            "needs."
        )
        next_experiment = "self_correcting_retrieval_rerank_then_rewrite"
    else:
        recommendation = (
            "None of the top-3 misses exists in the frozen top-20 pools; "
            "reranking cannot recover them. Test one controlled "
            "corpus-terminology query rewrite."
        )
        next_experiment = "corpus_terminology_query_rewrite"

    return {
        "control_retrieval": mean_metrics(
            answerable_rows,
            "control_scores",
        ),
        "candidate_availability": mean_metrics(
            answerable_rows,
            "candidate_availability",
        ),
        "by_question_type": {
            question_type: {
                "control_retrieval": mean_metrics(
                    rows,
                    "control_scores",
                ),
                "candidate_availability": mean_metrics(
                    rows,
                    "candidate_availability",
                ),
            }
            for question_type, rows in sorted(by_type.items())
        },
        "headroom": {
            "gold_units_missing_from_control_top_3": len(missing_units),
            "missing_units_found_in_top_20": len(rerankable),
            "missing_units_absent_from_top_20": len(absent),
            "recoverable_fraction": (
                len(rerankable) / len(missing_units)
                if missing_units
                else 1.0
            ),
            "rerankable_misses": rerankable,
            "candidate_absent_misses": absent,
        },
        "recommended_next_experiment": next_experiment,
        "recommendation": recommendation,
    }


def run_audit() -> None:
    validate_inputs()
    validate_frozen_configuration()
    if not DATABASE_DIR.exists():
        raise FileNotFoundError(
            f"Frozen OCR index not found at {DATABASE_DIR}."
        )

    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    reference_rows = read_jsonl(GOLD_FILE)
    vector_store = open_evaluation_vector_store()
    frozen_chunks = load_frozen_chunks(vector_store)
    bm25_index = BM25Index(frozen_chunks)
    audits: list[dict] = []

    print("Diagnostic experiment: frozen candidate-pool headroom audit")
    print(f"Reusing frozen OCR index: {DATABASE_DIR.resolve()}")
    print(
        f"Frozen pipeline: {CHUNK_SIZE}/{CHUNK_OVERLAP} characters, "
        f"{EMBEDDING_MODEL}, dense + BM25 RRF"
    )
    print(
        "Queries: full question plus existing explicit both-and subqueries only"
    )
    print("Candidate cutoffs: 3, 5, 10, 20 unique physical pages per query")
    print("Generation: disabled")
    print("Gold evidence: post-retrieval evaluation only")

    try:
        for position, reference in enumerate(reference_rows, start=1):
            question = str(reference["question"])
            start = time.perf_counter()

            # Retrieval is completed before gold evidence is passed to any
            # diagnostic function.
            control = retrieve_control_state(
                vector_store=vector_store,
                bm25_index=bm25_index,
                question=question,
            )
            query_runs = frozen_query_runs(
                vector_store=vector_store,
                bm25_index=bm25_index,
                question=question,
                control=control,
            )
            pools = [
                (run, unique_page_pool(run.fused_ranked))
                for run in query_runs
            ]
            control_top_3 = normalize_control_top_3(control)

            if bool(reference["answerable"]):
                gold_audit = audit_gold_after_retrieval(
                    gold_evidence=list(reference["gold_evidence"]),
                    control_top_3=control_top_3,
                    pools=pools,
                )
                control_scores = score_control(
                    control_top_3=control_top_3,
                    gold_evidence=list(reference["gold_evidence"]),
                )
                availability = score_candidate_availability(gold_audit)
            else:
                gold_audit = []
                control_scores = {}
                availability = {}

            elapsed_ms = (time.perf_counter() - start) * 1000
            missing = [
                item
                for item in gold_audit
                if not bool(item["in_control_top_3"])
            ]
            found_deeper = sum(
                item["diagnosis"] == "retrieved_below_final_cutoff"
                for item in missing
            )

            audits.append(
                {
                    "id": str(reference["id"]),
                    "question": question,
                    "type": str(reference["type"]),
                    "answerable": bool(reference["answerable"]),
                    "control_route": (
                        "explicit_both_and_decomposition"
                        if control.decomposed
                        else "single_full_question"
                    ),
                    "control_top_3": control_top_3,
                    "query_pools": [
                        {
                            "query_index": run.query_index,
                            "query_role": query_role(run),
                            "query": run.query,
                            "dense_candidate_count": run.dense_count,
                            "bm25_candidate_count": run.bm25_count,
                            "fused_chunk_count": len(run.fused_ranked),
                            "unique_page_count_saved": len(pool),
                            "top_unique_pages": [
                                normalize_page_candidate(item)
                                for item in pool
                            ],
                        }
                        for run, pool in pools
                    ],
                    "gold_evidence_audit": gold_audit,
                    "control_scores": control_scores,
                    "candidate_availability": availability,
                    "retrieval_ms": round(elapsed_ms, 3),
                }
            )

            print(
                f"  [{position:02d}/{len(reference_rows)}] "
                f"{reference['id']} queries={len(query_runs)} "
                f"top3_misses={len(missing)} "
                f"found_deeper={found_deeper} "
                f"{elapsed_ms:.1f} ms"
            )
    finally:
        stop_ollama_model(EMBEDDING_MODEL)
        stop_ollama_model(GENERATION_MODEL)

    summary_metrics = aggregate_summary(
        reference_rows=reference_rows,
        audits=audits,
    )
    summary = {
        "experiment": "Part 2 frozen candidate-pool headroom audit",
        "purpose": (
            "Determine whether gold evidence missed by the final top-3 "
            "selection is already present deeper in the unchanged candidate "
            "pools."
        ),
        "controls": {
            "ocr": "frozen selective-OCR index",
            "chunk_size_characters": CHUNK_SIZE,
            "chunk_overlap_characters": CHUNK_OVERLAP,
            "embedding_model": EMBEDDING_MODEL,
            "candidate_retrieval": (
                f"dense top {DENSE_CANDIDATE_K} plus BM25 top "
                f"{BM25_CANDIDATE_K}, equal-weight RRF"
            ),
            "query_policy": (
                "full question plus the frozen explicit both-and "
                "decomposition only"
            ),
            "control_final_top_k": FINAL_TOP_K,
            "generation": "not run",
            "uses_gold_for_query_retrieval_ranking_or_selection": False,
            "gold_usage": "post-retrieval diagnostic scoring only",
        },
        "candidate_pool_definition": (
            "Top unique physical pages per frozen query after chunk-level RRF; "
            "the highest-ranked chunk represents each page."
        ),
        "cutoffs": list(CUTOFFS),
        **summary_metrics,
        "files": {
            "question_level_audit": str(DETAIL_FILE),
            "summary": str(SUMMARY_FILE),
        },
    }
    write_jsonl(DETAIL_FILE, audits)
    write_json(SUMMARY_FILE, summary)

    control = summary["control_retrieval"]
    availability = summary["candidate_availability"]
    headroom = summary["headroom"]
    print("\nCandidate headroom audit complete")
    print(
        "  frozen control: "
        f"hit@3={control['hit_at_3']:.4f}, "
        f"recall@3={control['recall_at_3']:.4f}, "
        f"mrr@3={control['mrr_at_3']:.4f}"
    )
    print(
        "  candidate availability recall: "
        + ", ".join(
            f"@{cutoff}={availability[f'recall_at_{cutoff}']:.4f}"
            for cutoff in CUTOFFS
        )
    )
    print(
        "  missing gold units: "
        f"{headroom['gold_units_missing_from_control_top_3']}"
    )
    print(
        "  found deeper by top 20: "
        f"{headroom['missing_units_found_in_top_20']}"
    )
    print(
        "  absent from all top-20 query pools: "
        f"{headroom['missing_units_absent_from_top_20']}"
    )
    print(
        "Recommended next experiment: "
        f"{summary['recommended_next_experiment']}"
    )
    print(f"Summary saved to: {SUMMARY_FILE.resolve()}")


def main() -> None:
    run_audit()


if __name__ == "__main__":
    main()
