from __future__ import annotations

import itertools
import json
import re
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from langchain_core.documents import Document

from build_index import CHUNK_OVERLAP, CHUNK_SIZE, EMBEDDING_MODEL
from rag_query import GENERATION_MODEL, stop_ollama_model
from run_balanced_hybrid_evaluation import (
    FINAL_TOP_K,
    page_key,
    reciprocal_rank_fusion_all,
)
from run_decomposed_hybrid_evaluation import (
    QueryRun,
    decompose_question,
    retrieve_query,
)
from run_diversity_reranking_sweep import (
    ControlState,
    retrieve_control_state,
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
# - Keep the selective-OCR index, 1000/200 chunks, mxbai embeddings,
#   dense/BM25 candidate retrieval, equal-weight RRF, and final top-3 page
#   budget fixed.
# - Keep the preferred explicit both-and decomposed pipeline as the control.
# - Change only query routing and routed-page allocation:
#     1. detect compound requests beyond literal "both X and Y";
#     2. recognize explicit or generic multi-document requests;
#     3. reserve evidence for each atomic need or requested source;
#     4. use the full-question ranking for the remaining page.
#
# This is retrieval-only screening. Gold evidence is used only after retrieval
# to calculate metrics; it never influences routing, source choice, or page
# selection.
SWEEP_DIR = RESULTS_DIR / "adaptive_evidence_coverage_sweep"
CONTROL_FILE = SWEEP_DIR / "frozen_control_retrieval.jsonl"
CHALLENGER_FILE = SWEEP_DIR / "adaptive_coverage_retrieval.jsonl"
SUMMARY_FILE = SWEEP_DIR / "adaptive_evidence_coverage_results.json"

MAX_NEEDS = FINAL_TOP_K
ASSIGNMENT_CANDIDATES_PER_NEED = 8

WH_WORDS = r"what|which|how|why|where|when|who"
REPEATED_WH_PATTERN = re.compile(
    rf"\s+and\s+(?P<wh>{WH_WORDS})\b",
    flags=re.IGNORECASE,
)
FIRST_WH_PATTERN = re.compile(rf"\b(?P<wh>{WH_WORDS})\b", re.IGNORECASE)
GENERIC_MULTI_SOURCE_PATTERN = re.compile(
    r"\b(?:two|multiple|several)\s+"
    r"(?:documents|reports|profiles|publications|sources)\b",
    flags=re.IGNORECASE,
)

# These aliases describe the indexed corpus. They are ordinary production
# metadata, not benchmark labels or gold evidence.
SOURCE_ALIASES: dict[str, tuple[str, ...]] = {
    "nist_ai_rmf_1_0.pdf": (
        "ai rmf core",
        "general ai rmf",
        "ai risk management framework",
        "nist ai 100-1",
    ),
    "nist_ai_600_1_genai_profile.pdf": (
        "generative ai profile",
        "genai profile",
        "gai profile",
        "nist ai 600-1",
    ),
    "nist_sp_800_218a_scanned.pdf": (
        "ssdf ai community profile",
        "ssdf community profile",
        "sp 800-218a",
        "nist sp 800-218a",
    ),
}


@dataclass(frozen=True)
class RoutePlan:
    mode: str
    reason: str
    atomic_queries: list[str]
    explicit_sources: list[str]
    inferred_source_count: int

    @property
    def routed(self) -> bool:
        return self.mode != "direct"


@dataclass(frozen=True)
class CoverageRun:
    query_index: int
    query: str
    source_scope: str | None
    dense_count: int
    bm25_count: int
    fused_ranked: list[FusedCandidate]


@dataclass(frozen=True)
class CoverageSelection:
    candidate: FusedCandidate
    selection_role: str
    need_index: int | None
    query: str
    source_scope: str | None
    query_rrf_rank: int


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


def normalize_question(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def clean_question_fragment(text: str) -> str:
    return normalize_question(text).strip(" \t\r\n,;:.?!")


def explicit_sources_in_question(question: str) -> list[str]:
    folded = question.casefold()
    matches: list[tuple[int, str]] = []
    for source, aliases in SOURCE_ALIASES.items():
        positions = [
            folded.find(alias.casefold())
            for alias in aliases
            if folded.find(alias.casefold()) >= 0
        ]
        if positions:
            matches.append((min(positions), source))
    matches.sort()
    return [source for _position, source in matches]


def split_repeated_wh_question(question: str) -> list[str]:
    """
    Split requests such as "After X, what is recorded and what happens next?"

    Requiring a repeated question word avoids incorrectly splitting ordinary
    noun pairs such as "inputs and outputs".
    """
    normalized = normalize_question(question)
    conjunction = REPEATED_WH_PATTERN.search(normalized)
    first_wh = FIRST_WH_PATTERN.search(normalized)
    if conjunction is None or first_wh is None:
        return [normalized]
    if first_wh.start() >= conjunction.start():
        return [normalized]

    prefix = clean_question_fragment(normalized[: first_wh.start()])
    first_word = first_wh.group("wh")
    first_body = clean_question_fragment(
        normalized[first_wh.end() : conjunction.start()]
    )
    second_word = conjunction.group("wh")
    second_body = clean_question_fragment(
        normalized[conjunction.end() :]
    )
    if not first_body or not second_body:
        return [normalized]

    prefix_text = f"{prefix}, " if prefix else ""
    return [
        f"{prefix_text}{first_word} {first_body}?",
        f"{prefix_text}{second_word} {second_body}?",
    ]


def route_question(question: str) -> RoutePlan:
    """
    Choose a label-blind retrieval route from the question text alone.

    Priority:
    1. explicit references to at least two indexed documents;
    2. a generic request involving multiple documents;
    3. repeated-WH compound requests;
    4. explicit both-and compound requests;
    5. unchanged direct retrieval.
    """
    normalized = normalize_question(question)
    explicit_sources = explicit_sources_in_question(normalized)
    if len(explicit_sources) >= 2:
        return RoutePlan(
            mode="multi_source",
            reason="multiple_explicit_document_aliases",
            atomic_queries=[],
            explicit_sources=explicit_sources[:MAX_NEEDS],
            inferred_source_count=0,
        )

    if GENERIC_MULTI_SOURCE_PATTERN.search(normalized):
        return RoutePlan(
            mode="multi_source",
            reason="generic_multi_document_request",
            atomic_queries=[],
            explicit_sources=explicit_sources,
            inferred_source_count=max(2, len(explicit_sources)),
        )

    repeated_wh = split_repeated_wh_question(normalized)
    if len(repeated_wh) > 1:
        return RoutePlan(
            mode="atomic",
            reason="repeated_question_word",
            atomic_queries=repeated_wh[:MAX_NEEDS],
            explicit_sources=[],
            inferred_source_count=0,
        )

    both_and = decompose_question(normalized)
    if len(both_and) > 1:
        return RoutePlan(
            mode="atomic",
            reason="explicit_both_and",
            atomic_queries=both_and[:MAX_NEEDS],
            explicit_sources=[],
            inferred_source_count=0,
        )

    return RoutePlan(
        mode="direct",
        reason="single_information_need",
        atomic_queries=[normalized],
        explicit_sources=[],
        inferred_source_count=0,
    )


def validate_router_self_tests() -> None:
    repeated = route_question(
        "After a system is tested, what must be recorded and "
        "what action follows?"
    )
    if repeated.mode != "atomic" or len(repeated.atomic_queries) != 2:
        raise RuntimeError("Repeated-WH router self-test failed.")

    both = route_question(
        "How should a team protect both stored models and runtime records?"
    )
    if both.mode != "atomic" or len(both.atomic_queries) != 2:
        raise RuntimeError("Both-and router self-test failed.")

    generic_sources = route_question(
        "How do the two reports connect governance with monitoring?"
    )
    if (
        generic_sources.mode != "multi_source"
        or generic_sources.inferred_source_count != 2
    ):
        raise RuntimeError("Generic multi-source router self-test failed.")

    direct = route_question(
        "Which controls protect weights and configuration parameters?"
    )
    if direct.mode != "direct":
        raise RuntimeError("Direct-question router self-test failed.")


def validate_frozen_configuration() -> None:
    if CHUNK_SIZE != 1000 or CHUNK_OVERLAP != 200:
        raise RuntimeError(
            "This sweep requires the frozen 1000/200-character index, but "
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
    missing_aliases = sorted(set(SOURCE_TO_DOC_ID) - set(SOURCE_ALIASES))
    extra_aliases = sorted(set(SOURCE_ALIASES) - set(SOURCE_TO_DOC_ID))
    if missing_aliases or extra_aliases:
        raise RuntimeError(
            "Corpus source aliases do not match SOURCE_TO_DOC_ID. "
            f"Missing={missing_aliases}; extra={extra_aliases}."
        )


def as_coverage_run(run: QueryRun) -> CoverageRun:
    return CoverageRun(
        query_index=run.query_index,
        query=run.query,
        source_scope=None,
        dense_count=run.dense_count,
        bm25_count=run.bm25_count,
        fused_ranked=run.fused_ranked,
    )


def retrieve_source_scoped_query(
    vector_store,
    source_bm25_indexes: dict[str, BM25Index],
    query: str,
    query_index: int,
    source: str,
) -> CoverageRun:
    dense_results = vector_store.similarity_search_with_score(
        query=query,
        k=DENSE_CANDIDATE_K,
        filter={"source": source},
    )
    bm25_results = source_bm25_indexes[source].search(
        query=query,
        k=BM25_CANDIDATE_K,
    )
    if not dense_results:
        raise RuntimeError(
            f"Source-scoped dense retrieval returned no chunks for {source}."
        )
    if not bm25_results:
        raise RuntimeError(
            f"Source-scoped BM25 retrieval returned no chunks for {source}."
        )
    return CoverageRun(
        query_index=query_index,
        query=query,
        source_scope=source,
        dense_count=len(dense_results),
        bm25_count=len(bm25_results),
        fused_ranked=reciprocal_rank_fusion_all(
            dense_results=dense_results,
            bm25_results=bm25_results,
        ),
    )


def infer_relevant_sources(
    full_question_run: CoverageRun,
    source_count: int,
    already_selected: list[str],
) -> list[str]:
    """
    Infer source affinity from the full-question global ranking.

    Only live retrieval ranks are used. Gold evidence and benchmark types are
    unavailable to this function.
    """
    best_rank: dict[str, int] = {}
    for rank, candidate in enumerate(
        full_question_run.fused_ranked,
        start=1,
    ):
        source = str(candidate.document.metadata.get("source", ""))
        best_rank.setdefault(source, rank)

    ranked_sources = sorted(
        SOURCE_TO_DOC_ID,
        key=lambda source: (
            best_rank.get(source, 10**9),
            source,
        ),
    )
    selected = list(already_selected)
    for source in ranked_sources:
        if source not in selected:
            selected.append(source)
        if len(selected) == source_count:
            break
    if len(selected) != source_count:
        raise RuntimeError(
            f"Could infer only {len(selected)} of {source_count} sources."
        )
    return selected


def unique_page_candidates(
    run: CoverageRun,
    limit: int,
) -> list[tuple[int, FusedCandidate]]:
    result: list[tuple[int, FusedCandidate]] = []
    seen_pages: set[tuple[str, int]] = set()
    for rank, candidate in enumerate(run.fused_ranked, start=1):
        page = page_key(candidate.document)
        if page in seen_pages:
            continue
        seen_pages.add(page)
        result.append((rank, candidate))
        if len(result) == limit:
            break
    return result


def choose_need_assignments(
    need_runs: list[CoverageRun],
) -> list[CoverageSelection]:
    """
    Solve the small evidence-allocation problem jointly.

    Each need receives one unique physical page. The winning assignment
    maximizes reciprocal-rank utility across all needs, avoiding the order
    bias of greedily serving the first need before the second.
    """
    if not 1 < len(need_runs) <= MAX_NEEDS:
        raise ValueError(
            f"Expected 2..{MAX_NEEDS} evidence needs, got {len(need_runs)}."
        )

    choices = [
        unique_page_candidates(
            run,
            limit=ASSIGNMENT_CANDIDATES_PER_NEED,
        )
        for run in need_runs
    ]
    if any(not items for items in choices):
        raise RuntimeError("At least one evidence need has no candidates.")

    scored: list[
        tuple[
            float,
            int,
            int,
            tuple[tuple[str, int, int], ...],
            tuple[tuple[int, FusedCandidate], ...],
        ]
    ] = []
    for assignment in itertools.product(*choices):
        pages = [
            page_key(candidate.document)
            for _rank, candidate in assignment
        ]
        if len(set(pages)) != len(pages):
            continue
        ranks = [rank for rank, _candidate in assignment]
        utility = sum(1.0 / rank for rank in ranks)
        keys = tuple(
            chunk_key(candidate.document)
            for _rank, candidate in assignment
        )
        scored.append(
            (
                -utility,
                max(ranks),
                sum(ranks),
                keys,
                assignment,
            )
        )

    if not scored:
        raise RuntimeError(
            "Could not assign a unique physical page to every evidence need."
        )
    scored.sort(key=lambda item: item[:4])
    winning_assignment = scored[0][4]

    return [
        CoverageSelection(
            candidate=candidate,
            selection_role=f"evidence_need_{need_index}",
            need_index=need_index,
            query=run.query,
            source_scope=run.source_scope,
            query_rrf_rank=rank,
        )
        for need_index, (run, (rank, candidate)) in enumerate(
            zip(need_runs, winning_assignment),
            start=1,
        )
    ]


def fill_from_full_question(
    selected: list[CoverageSelection],
    full_question_run: CoverageRun,
) -> list[CoverageSelection]:
    selected_pages = {
        page_key(item.candidate.document)
        for item in selected
    }
    for rank, candidate in enumerate(
        full_question_run.fused_ranked,
        start=1,
    ):
        if len(selected) == FINAL_TOP_K:
            break
        page = page_key(candidate.document)
        if page in selected_pages:
            continue
        selected.append(
            CoverageSelection(
                candidate=candidate,
                selection_role="full_question_fill",
                need_index=None,
                query=full_question_run.query,
                source_scope=None,
                query_rrf_rank=rank,
            )
        )
        selected_pages.add(page)

    if len(selected) != FINAL_TOP_K:
        raise RuntimeError(
            f"Coverage retrieval selected {len(selected)} pages; "
            f"expected {FINAL_TOP_K}."
        )
    return selected


def retrieve_adaptive_coverage(
    vector_store,
    bm25_index: BM25Index,
    source_bm25_indexes: dict[str, BM25Index],
    question: str,
    control: ControlState,
) -> tuple[list[CoverageSelection], RoutePlan, list[CoverageRun]]:
    plan = route_question(question)
    if not plan.routed:
        selections = [
            CoverageSelection(
                candidate=candidate,
                selection_role=role,
                need_index=None,
                query=question,
                source_scope=None,
                query_rrf_rank=rank,
            )
            for rank, (candidate, role) in enumerate(
                zip(control.selected, control.selection_roles),
                start=1,
            )
        ]
        return selections, plan, []

    full_run = as_coverage_run(
        retrieve_query(
            vector_store=vector_store,
            bm25_index=bm25_index,
            query=question,
            query_index=0,
        )
    )

    if plan.mode == "atomic":
        need_runs = [
            as_coverage_run(
                retrieve_query(
                    vector_store=vector_store,
                    bm25_index=bm25_index,
                    query=query,
                    query_index=index,
                )
            )
            for index, query in enumerate(
                plan.atomic_queries,
                start=1,
            )
        ]
    elif plan.mode == "multi_source":
        desired_count = max(
            len(plan.explicit_sources),
            plan.inferred_source_count,
        )
        desired_count = min(desired_count, MAX_NEEDS)
        sources = infer_relevant_sources(
            full_question_run=full_run,
            source_count=desired_count,
            already_selected=plan.explicit_sources,
        )
        need_runs = [
            retrieve_source_scoped_query(
                vector_store=vector_store,
                source_bm25_indexes=source_bm25_indexes,
                query=question,
                query_index=index,
                source=source,
            )
            for index, source in enumerate(sources, start=1)
        ]
    else:
        raise RuntimeError(f"Unknown route mode: {plan.mode}")

    selected = choose_need_assignments(need_runs)
    selected = fill_from_full_question(selected, full_run)
    return selected, plan, [*need_runs, full_run]


def normalized_control_retrieval(state: ControlState) -> list[dict]:
    rows: list[dict] = []
    for rank, (candidate, role) in enumerate(
        zip(state.selected, state.selection_roles),
        start=1,
    ):
        document = candidate.document
        source = str(document.metadata.get("source", ""))
        rows.append(
            {
                "rank": rank,
                "doc_id": SOURCE_TO_DOC_ID[source],
                "pages": [int(document.metadata["page"])],
                "chunk": int(document.metadata["chunk"]),
                "selection_role": role,
            }
        )
    return rows


def normalized_coverage_retrieval(
    selected: list[CoverageSelection],
) -> list[dict]:
    rows: list[dict] = []
    for rank, item in enumerate(selected, start=1):
        candidate = item.candidate
        document = candidate.document
        source = str(document.metadata.get("source", ""))
        rows.append(
            {
                "rank": rank,
                "doc_id": SOURCE_TO_DOC_ID[source],
                "pages": [int(document.metadata["page"])],
                "chunk": int(document.metadata["chunk"]),
                "hybrid_rrf_score": float(candidate.rrf_score),
                "dense_rank": candidate.dense_rank,
                "bm25_rank": candidate.bm25_rank,
                "selection_role": item.selection_role,
                "need_index": item.need_index,
                "retrieval_query": item.query,
                "source_scope": (
                    SOURCE_TO_DOC_ID[item.source_scope]
                    if item.source_scope is not None
                    else None
                ),
                "query_rrf_rank": item.query_rrf_rank,
            }
        )
    return rows


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


def score_one(
    retrieved: list[dict],
    reference_evidence: list[dict],
) -> dict[str, float]:
    reference_units = evidence_units(reference_evidence)
    seen_reference: set[tuple[str, int]] = set()
    first_relevant_rank: int | None = None
    for rank, result in enumerate(retrieved[:FINAL_TOP_K], start=1):
        relevant = evidence_units([result]) & reference_units
        if relevant and first_relevant_rank is None:
            first_relevant_rank = rank
        seen_reference.update(relevant)
    return {
        "hit_at_3": float(bool(seen_reference)),
        "recall_at_3": len(seen_reference) / len(reference_units),
        "mrr_at_3": (
            0.0 if first_relevant_rank is None else 1.0 / first_relevant_rank
        ),
    }


def aggregate_metrics(
    reference_rows: list[dict],
    predictions: list[dict],
) -> dict:
    by_id = {str(row["id"]): row for row in predictions}
    expected_ids = {str(row["id"]) for row in reference_rows}
    if set(by_id) != expected_ids:
        raise ValueError("Retrieval prediction IDs do not match reference IDs.")

    values = {"hit_at_3": [], "recall_at_3": [], "mrr_at_3": []}
    routed_values = {
        "hit_at_3": [],
        "recall_at_3": [],
        "mrr_at_3": [],
    }
    by_type: dict[str, dict[str, list[float]]] = {}
    route_counts: dict[str, int] = {}
    selection_changes = 0

    for reference in reference_rows:
        prediction = by_id[str(reference["id"])]
        route_reason = str(prediction["route"]["reason"])
        route_counts[route_reason] = route_counts.get(route_reason, 0) + 1
        selection_changes += int(bool(prediction["selection_changed"]))
        if not bool(reference["answerable"]):
            continue

        scores = score_one(
            prediction["retrieved"],
            reference["gold_evidence"],
        )
        question_type = str(reference["type"])
        type_values = by_type.setdefault(
            question_type,
            {"hit_at_3": [], "recall_at_3": [], "mrr_at_3": []},
        )
        for name, value in scores.items():
            values[name].append(value)
            type_values[name].append(value)
            if bool(prediction["route"]["applied"]):
                routed_values[name].append(value)

    return {
        "retrieval": {
            name: statistics.fmean(metric_values)
            for name, metric_values in values.items()
        },
        "by_question_type": {
            question_type: {
                name: statistics.fmean(metric_values)
                for name, metric_values in type_values.items()
            }
            for question_type, type_values in sorted(by_type.items())
        },
        "routed_answerable_subset": {
            name: (
                statistics.fmean(metric_values)
                if metric_values
                else None
            )
            for name, metric_values in routed_values.items()
        },
        "routing": {
            "route_counts_all_questions": dict(sorted(route_counts.items())),
            "selection_changes_vs_control": selection_changes,
        },
    }


def retrieval_signature(retrieved: list[dict]) -> list[tuple[str, int]]:
    return [
        (str(item["doc_id"]), int(item["pages"][0]))
        for item in retrieved
    ]


def build_source_bm25_indexes(
    frozen_chunks: list[Document],
) -> dict[str, BM25Index]:
    grouped: dict[str, list[Document]] = {
        source: [] for source in SOURCE_TO_DOC_ID
    }
    for document in frozen_chunks:
        source = str(document.metadata.get("source", ""))
        if source not in grouped:
            raise RuntimeError(f"Unexpected source in frozen index: {source}")
        grouped[source].append(document)
    empty = sorted(source for source, documents in grouped.items() if not documents)
    if empty:
        raise RuntimeError(f"Frozen index has no chunks for sources: {empty}")
    return {
        source: BM25Index(documents)
        for source, documents in grouped.items()
    }


def run_sweep() -> None:
    validate_inputs()
    validate_frozen_configuration()
    validate_router_self_tests()
    if not DATABASE_DIR.exists():
        raise FileNotFoundError(
            f"Frozen OCR index not found at {DATABASE_DIR}."
        )

    SWEEP_DIR.mkdir(parents=True, exist_ok=True)
    reference_rows = read_jsonl(GOLD_FILE)
    vector_store = open_evaluation_vector_store()
    frozen_chunks = load_frozen_chunks(vector_store)
    bm25_index = BM25Index(frozen_chunks)
    source_bm25_indexes = build_source_bm25_indexes(frozen_chunks)

    control_predictions: list[dict] = []
    challenger_predictions: list[dict] = []

    print("Controlled experiment: adaptive evidence-coverage routing")
    print(f"Reusing frozen OCR index: {DATABASE_DIR.resolve()}")
    print(
        f"Frozen pipeline: {CHUNK_SIZE}/{CHUNK_OVERLAP} characters, "
        f"{EMBEDDING_MODEL}, dense + BM25 RRF, top 3"
    )
    print(
        "Control: explicit both-and decomposition only"
    )
    print(
        "Challenger: adaptive atomic/multi-source routing + "
        "joint evidence allocation"
    )
    print("Generation: disabled during retrieval screening")

    try:
        for position, reference in enumerate(reference_rows, start=1):
            question = str(reference["question"])
            question_start = time.perf_counter()

            control = retrieve_control_state(
                vector_store=vector_store,
                bm25_index=bm25_index,
                question=question,
            )
            control_retrieved = normalized_control_retrieval(control)

            selected, route, runs = retrieve_adaptive_coverage(
                vector_store=vector_store,
                bm25_index=bm25_index,
                source_bm25_indexes=source_bm25_indexes,
                question=question,
                control=control,
            )
            challenger_retrieved = normalized_coverage_retrieval(selected)
            elapsed_ms = (time.perf_counter() - question_start) * 1000

            if len({tuple(item["pages"]) + (item["doc_id"],) for item in challenger_retrieved}) != FINAL_TOP_K:
                raise RuntimeError(
                    "Adaptive retrieval produced duplicate physical pages."
                )

            changed = (
                retrieval_signature(challenger_retrieved)
                != retrieval_signature(control_retrieved)
            )
            control_predictions.append(
                {
                    "id": str(reference["id"]),
                    "question": question,
                    "retrieved": control_retrieved,
                    "route": {
                        "applied": bool(control.decomposed),
                        "reason": (
                            "explicit_both_and"
                            if control.decomposed
                            else "single_information_need"
                        ),
                    },
                    "selection_changed": False,
                    "retrieval_ms": round(elapsed_ms, 3),
                }
            )
            challenger_predictions.append(
                {
                    "id": str(reference["id"]),
                    "question": question,
                    "retrieved": challenger_retrieved,
                    "control_retrieved": control_retrieved,
                    "route": {
                        "applied": route.routed,
                        "mode": route.mode,
                        "reason": route.reason,
                        "atomic_queries": route.atomic_queries,
                        "explicit_sources": [
                            SOURCE_TO_DOC_ID[source]
                            for source in route.explicit_sources
                        ],
                        "inferred_source_count": route.inferred_source_count,
                    },
                    "query_runs": [
                        {
                            "query_index": run.query_index,
                            "query": run.query,
                            "source_scope": (
                                SOURCE_TO_DOC_ID[run.source_scope]
                                if run.source_scope is not None
                                else None
                            ),
                            "dense_count": run.dense_count,
                            "bm25_count": run.bm25_count,
                        }
                        for run in runs
                    ],
                    "selection_changed": changed,
                    "retrieval_ms": round(elapsed_ms, 3),
                }
            )

            print(
                f"  [{position:02d}/{len(reference_rows)}] "
                f"{reference['id']} route={route.reason} "
                f"changed={'yes' if changed else 'no'} "
                f"{elapsed_ms:.1f} ms"
            )
    finally:
        stop_ollama_model(EMBEDDING_MODEL)
        stop_ollama_model(GENERATION_MODEL)

    write_jsonl(CONTROL_FILE, control_predictions)
    write_jsonl(CHALLENGER_FILE, challenger_predictions)

    control_metrics = aggregate_metrics(
        reference_rows=reference_rows,
        predictions=control_predictions,
    )
    challenger_metrics = aggregate_metrics(
        reference_rows=reference_rows,
        predictions=challenger_predictions,
    )

    control_retrieval = control_metrics["retrieval"]
    challenger_retrieval = challenger_metrics["retrieval"]
    challenger_types = challenger_metrics["by_question_type"]
    control_types = control_metrics["by_question_type"]

    criteria = {
        "hit_at_3_not_below_control": (
            challenger_retrieval["hit_at_3"]
            >= control_retrieval["hit_at_3"]
        ),
        "recall_at_3_above_control": (
            challenger_retrieval["recall_at_3"]
            > control_retrieval["recall_at_3"]
        ),
        "multi_chunk_recall_at_least_0_50": (
            challenger_types.get("multi_chunk", {}).get("recall_at_3", 0.0)
            >= 0.50
        ),
        "ocr_multi_chunk_not_below_control": (
            challenger_types.get(
                "ocr_multi_chunk",
                {},
            ).get("recall_at_3", 0.0)
            >= control_types.get(
                "ocr_multi_chunk",
                {},
            ).get("recall_at_3", 0.0)
        ),
    }
    promote = all(criteria.values())

    result = {
        "experiment": "Part 2 adaptive evidence-coverage routing screening",
        "independent_variable": (
            "query routing and coverage-aware allocation of the same "
            "three-page context budget"
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
            "control_router": "explicit both-and decomposition only",
            "final_top_k": FINAL_TOP_K,
            "generation": "not run during screening",
            "uses_gold_for_routing_or_selection": False,
        },
        "challenger": {
            "router": [
                "multiple explicit corpus document aliases",
                "generic multi-document requests",
                "repeated question words",
                "explicit both-and requests",
            ],
            "selection": (
                "joint maximum reciprocal-rank assignment of one unique "
                "page per evidence need, then full-question RRF fill"
            ),
            "source_routing": (
                "source-scoped dense and BM25 retrieval for multi-document "
                "requests"
            ),
        },
        "control_results": control_metrics,
        "challenger_results": challenger_metrics,
        "promotion_criteria": criteria,
        "decision": (
            "promote_to_full_generation_confirmation"
            if promote
            else "do_not_promote; audit routed questions before revising"
        ),
        "files": {
            "control_retrieval": str(CONTROL_FILE),
            "challenger_retrieval": str(CHALLENGER_FILE),
        },
    }
    write_json(SUMMARY_FILE, result)

    print("\nAdaptive evidence-coverage screening complete")
    print(
        "  control: "
        f"hit@3={control_retrieval['hit_at_3']:.4f}, "
        f"recall@3={control_retrieval['recall_at_3']:.4f}, "
        f"mrr@3={control_retrieval['mrr_at_3']:.4f}"
    )
    print(
        "  challenger: "
        f"hit@3={challenger_retrieval['hit_at_3']:.4f}, "
        f"recall@3={challenger_retrieval['recall_at_3']:.4f}, "
        f"mrr@3={challenger_retrieval['mrr_at_3']:.4f}"
    )
    for question_type in ("multi_chunk", "ocr_multi_chunk"):
        control_value = control_types.get(
            question_type,
            {},
        ).get("recall_at_3")
        challenger_value = challenger_types.get(
            question_type,
            {},
        ).get("recall_at_3")
        print(
            f"  {question_type} recall@3: "
            f"control={control_value:.4f}, "
            f"challenger={challenger_value:.4f}"
        )
    print(
        "  routed questions: "
        f"{sum(count for reason, count in challenger_metrics['routing']['route_counts_all_questions'].items() if reason != 'single_information_need')}"
    )
    print(
        "  selection changes: "
        f"{challenger_metrics['routing']['selection_changes_vs_control']}"
    )
    print(
        "Decision: "
        + (
            "PROMOTE to one full generation confirmation"
            if promote
            else "DO NOT PROMOTE; audit routed questions"
        )
    )
    print(f"Summary saved to: {SUMMARY_FILE.resolve()}")


def main() -> None:
    run_sweep()


if __name__ == "__main__":
    main()
