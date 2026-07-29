from __future__ import annotations

import json
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from langchain_ollama import ChatOllama

from build_index import CHUNK_OVERLAP, CHUNK_SIZE, EMBEDDING_MODEL
from rag_query import GENERATION_MODEL, stop_ollama_model
from run_balanced_hybrid_evaluation import FINAL_TOP_K, page_key
from run_candidate_headroom_audit import (
    PageCandidate,
    frozen_query_runs,
    unique_page_pool,
)
from run_diversity_reranking_sweep import (
    ControlState,
    normalized_retrieval,
    retrieve_control_state,
)
from run_hybrid_evaluation import BM25Index, FusedCandidate, load_frozen_chunks
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
# - Reuse the frozen selective-OCR index, 1000/200 chunks, mxbai embeddings,
#   dense + BM25 RRF retrieval, explicit both-and decomposition, and top-3
#   context budget.
# - Change only final page selection. The challenger lets the existing Qwen
#   model choose three collectively answer-bearing pages from a small frozen
#   candidate shortlist.
# - Gold evidence is never included in the prompt or candidate selection. It
#   is used only after every selection has been committed.
# - This script does not generate final answers.
EXPERIMENT_DIR = RESULTS_DIR / "evidence_aware_reranking_sweep"
DETAIL_FILE = EXPERIMENT_DIR / "evidence_aware_reranking_by_question.jsonl"
SUMMARY_FILE = EXPERIMENT_DIR / "evidence_aware_reranking_summary.json"

DIRECT_POOL_CUTOFF = 10
DECOMPOSED_POOL_CUTOFF = 3
VERIFIER_TEMPERATURE = 0.0
VERIFIER_NUM_CTX = 4096
VERIFIER_NUM_PREDICT = 96


@dataclass(frozen=True)
class CandidateView:
    label: str
    candidate: FusedCandidate
    control_rank: int | None
    query_sources: tuple[dict, ...]


@dataclass(frozen=True)
class RetrievalCase:
    question_id: str
    question: str
    question_type: str
    answerable: bool
    control: ControlState
    candidates: list[CandidateView]
    control_labels: list[str]
    retrieval_ms: float


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
            "This experiment requires the frozen 1000/200-character index, "
            f"but build_index.py reports {CHUNK_SIZE}/{CHUNK_OVERLAP}."
        )
    if "mxbai-embed-large" not in EMBEDDING_MODEL:
        raise RuntimeError(
            "This experiment requires the preferred mxbai-embed-large "
            f"index, but build_index.py reports {EMBEDDING_MODEL!r}."
        )
    if FINAL_TOP_K != 3:
        raise RuntimeError(
            f"This experiment requires a top-3 context budget, not "
            f"{FINAL_TOP_K}."
        )


def candidate_unit(candidate: FusedCandidate) -> tuple[str, int]:
    source = str(candidate.document.metadata.get("source", ""))
    if source not in SOURCE_TO_DOC_ID:
        raise ValueError(f"Unknown source filename in Chroma: {source}")
    return (
        SOURCE_TO_DOC_ID[source],
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


def build_candidate_shortlist(
    control: ControlState,
    query_pools: list[tuple[object, list[PageCandidate]]],
) -> tuple[list[CandidateView], list[str]]:
    """
    Build a deterministic, label-blind page shortlist.

    Control pages are always available as a safe fallback. A direct question
    receives the first ten unique pages from its unchanged full-question RRF
    pool. Existing decomposed questions receive the first three pages from
    every frozen subquery and from the full-question pool, which preserves
    coverage across the already-established information needs.
    """
    ordered_units: list[tuple[str, int]] = []
    candidates: dict[tuple[str, int], FusedCandidate] = {}
    control_ranks: dict[tuple[str, int], int] = {}
    sources: dict[tuple[str, int], list[dict]] = {}

    def add_candidate(
        candidate: FusedCandidate,
        source_record: dict,
        control_rank: int | None = None,
    ) -> None:
        unit = candidate_unit(candidate)
        if unit not in candidates:
            ordered_units.append(unit)
            candidates[unit] = candidate
            sources[unit] = []
        # Preserve the exact frozen control chunk for pages already selected.
        if control_rank is not None:
            candidates[unit] = candidate
            control_ranks[unit] = control_rank
        sources[unit].append(source_record)

    for rank, (candidate, role) in enumerate(
        zip(control.selected, control.selection_roles),
        start=1,
    ):
        add_candidate(
            candidate,
            {
                "role": "frozen_control",
                "control_rank": rank,
                "selection_role": role,
            },
            control_rank=rank,
        )

    cutoff = (
        DECOMPOSED_POOL_CUTOFF
        if control.decomposed
        else DIRECT_POOL_CUTOFF
    )
    for run, pool in query_pools:
        for item in pool[:cutoff]:
            add_candidate(
                item.candidate,
                {
                    "role": (
                        "full_question"
                        if int(run.query_index) == 0
                        else "frozen_subquery"
                    ),
                    "query_index": int(run.query_index),
                    "query": str(run.query),
                    "page_rank": int(item.page_rank),
                    "fused_chunk_rank": int(item.fused_chunk_rank),
                },
            )

    if len(ordered_units) < FINAL_TOP_K:
        raise RuntimeError(
            f"Candidate shortlist contains only {len(ordered_units)} unique "
            "pages."
        )

    views: list[CandidateView] = []
    for index, unit in enumerate(ordered_units, start=1):
        views.append(
            CandidateView(
                label=f"C{index:02d}",
                candidate=candidates[unit],
                control_rank=control_ranks.get(unit),
                query_sources=tuple(sources[unit]),
            )
        )

    labels_by_unit = {
        candidate_unit(view.candidate): view.label for view in views
    }
    control_labels = [
        labels_by_unit[candidate_unit(candidate)]
        for candidate in control.selected
    ]
    if len(control_labels) != FINAL_TOP_K:
        raise RuntimeError("Could not map all frozen control pages.")
    return views, control_labels


def candidate_signature(candidates: list[CandidateView]) -> list[dict]:
    return [
        {
            "label": view.label,
            "doc_id": candidate_unit(view.candidate)[0],
            "page": candidate_unit(view.candidate)[1],
            "chunk": int(view.candidate.document.metadata["chunk"]),
            "control_rank": view.control_rank,
        }
        for view in candidates
    ]


def candidate_prompt_block(view: CandidateView) -> str:
    doc_id, page = candidate_unit(view.candidate)
    control = (
        f"yes, rank {view.control_rank}"
        if view.control_rank is not None
        else "no"
    )
    text = view.candidate.document.page_content.strip()
    return (
        f"[{view.label}] document={doc_id}; page={page}; "
        f"current_control={control}\n{text}"
    )


def verifier_messages(case: RetrievalCase) -> list[tuple[str, str]]:
    candidates = "\n\n---\n\n".join(
        candidate_prompt_block(view) for view in case.candidates
    )
    return [
        (
            "system",
            (
                "You are an evidence-page selector for a RAG system. Use only "
                "the supplied candidate passages. Select exactly three unique "
                "candidate IDs whose passages collectively provide the most "
                "direct and complete factual support for every part of the "
                "question. A passage that merely mentions the topic is weaker "
                "than one that states the requested definition, requirement, "
                "relationship, action, or list. Prefer the current control "
                "pages when evidence quality is tied, but replace them when a "
                "deeper candidate clearly answers a missing part. Do not "
                "answer the question. Return only one JSON object in this "
                "exact shape: "
                '{"selected":["C01","C02","C03"],'
                '"coverage":"complete_or_partial"}. '
                "Order selected IDs from strongest direct evidence to weakest. "
                "Do not include prose, Markdown, or reasoning."
            ),
        ),
        (
            "human",
            (
                f"Question:\n{case.question}\n\n"
                "Current control IDs:\n"
                f"{json.dumps(case.control_labels)}\n\n"
                f"Candidate passages:\n{candidates}"
            ),
        ),
    ]


def find_json_object(raw: str) -> dict:
    decoder = json.JSONDecoder()
    for position, character in enumerate(raw):
        if character != "{":
            continue
        try:
            value, _end = decoder.raw_decode(raw[position:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise ValueError("Verifier response did not contain a JSON object.")


def parse_verifier_selection(
    raw: str,
    valid_labels: set[str],
) -> tuple[list[str], str]:
    value = find_json_object(raw)
    selected = value.get("selected")
    if not isinstance(selected, list):
        raise ValueError("Verifier JSON has no selected list.")
    labels = [str(item).strip().upper() for item in selected]
    if len(labels) != FINAL_TOP_K or len(set(labels)) != FINAL_TOP_K:
        raise ValueError("Verifier must select exactly three unique IDs.")
    invalid = [label for label in labels if label not in valid_labels]
    if invalid:
        raise ValueError(f"Verifier selected invalid IDs: {invalid}")
    coverage = str(value.get("coverage", "unspecified")).strip()
    return labels, coverage


def normalize_selected(
    selected: list[FusedCandidate],
    selected_labels: list[str],
    status: str,
) -> list[dict]:
    return normalized_retrieval(
        candidates=selected,
        roles=[
            (
                "frozen_control_fallback"
                if status != "ok"
                else f"qwen_evidence_verifier_{label}"
            )
            for label in selected_labels
        ],
    )


def score_one(
    retrieved: list[dict],
    gold_evidence: list[dict],
) -> dict[str, float]:
    """
    Post-selection evaluation boundary.

    This is the only stage that needs gold evidence. Neither retrieval, the
    prompt, Qwen, nor response parsing receives benchmark evidence labels.
    """
    gold = evidence_units(gold_evidence)
    found: set[tuple[str, int]] = set()
    first_relevant_rank: int | None = None
    for rank, item in enumerate(retrieved[:FINAL_TOP_K], start=1):
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


def mean_metrics(rows: list[dict], field: str) -> dict[str, float]:
    names = ("hit_at_3", "recall_at_3", "mrr_at_3")
    return {
        name: statistics.fmean(float(row[field][name]) for row in rows)
        for name in names
    }


def aggregate_results(
    gold_rows: list[dict],
    detail_rows: list[dict],
) -> dict:
    details_by_id = {str(row["id"]): row for row in detail_rows}
    expected_ids = {str(row["id"]) for row in gold_rows}
    if set(details_by_id) != expected_ids:
        raise ValueError("Detail IDs do not match gold question IDs.")

    answerable: list[dict] = []
    by_type: dict[str, list[dict]] = {}
    for gold in gold_rows:
        if not bool(gold["answerable"]):
            continue
        row = details_by_id[str(gold["id"])]
        row["control_scores"] = score_one(
            row["control_retrieved"],
            list(gold["gold_evidence"]),
        )
        row["challenger_scores"] = score_one(
            row["challenger_retrieved"],
            list(gold["gold_evidence"]),
        )
        answerable.append(row)
        by_type.setdefault(str(gold["type"]), []).append(row)

    control = mean_metrics(answerable, "control_scores")
    challenger = mean_metrics(answerable, "challenger_scores")
    by_type_metrics = {
        question_type: {
            "control": mean_metrics(rows, "control_scores"),
            "challenger": mean_metrics(rows, "challenger_scores"),
        }
        for question_type, rows in sorted(by_type.items())
    }

    multi_recall = by_type_metrics.get("multi_chunk", {}).get(
        "challenger", {}
    ).get("recall_at_3", 0.0)
    ocr_multi_control = by_type_metrics.get("ocr_multi_chunk", {}).get(
        "control", {}
    ).get("recall_at_3", 0.0)
    ocr_multi_challenger = by_type_metrics.get(
        "ocr_multi_chunk", {}
    ).get("challenger", {}).get("recall_at_3", 0.0)

    gates = {
        "hit_not_below_control": (
            challenger["hit_at_3"] >= control["hit_at_3"]
        ),
        "overall_recall_above_control": (
            challenger["recall_at_3"] > control["recall_at_3"]
        ),
        "mrr_not_below_control": (
            challenger["mrr_at_3"] >= control["mrr_at_3"]
        ),
        "multi_chunk_recall_at_least_0_50": multi_recall >= 0.50,
        "ocr_multi_chunk_recall_above_control": (
            ocr_multi_challenger > ocr_multi_control
        ),
    }
    promoted = all(gates.values())

    return {
        "control_retrieval": control,
        "challenger_retrieval": challenger,
        "by_question_type": by_type_metrics,
        "selection_changes": sum(
            int(bool(row["selection_changed"])) for row in detail_rows
        ),
        "verifier_fallbacks": sum(
            int(str(row["verifier"]["status"]) != "ok")
            for row in detail_rows
        ),
        "mean_verifier_ms": statistics.fmean(
            float(row["verifier_ms"]) for row in detail_rows
        ),
        "promotion_gates": gates,
        "promoted": promoted,
    }


def load_completed() -> dict[str, dict]:
    if not DETAIL_FILE.exists():
        return {}
    completed: dict[str, dict] = {}
    for row in read_jsonl(DETAIL_FILE):
        question_id = str(row["id"])
        if question_id in completed:
            raise ValueError(f"Duplicate detail ID: {question_id}")
        completed[question_id] = row
    return completed


def run_verifier(
    llm: ChatOllama,
    case: RetrievalCase,
) -> dict:
    views_by_label = {view.label: view for view in case.candidates}
    valid_labels = set(views_by_label)
    start = time.perf_counter()
    raw = ""
    error: str | None = None
    try:
        response = llm.invoke(verifier_messages(case))
        raw = str(response.content).strip()
        selected_labels, coverage = parse_verifier_selection(
            raw=raw,
            valid_labels=valid_labels,
        )
        status = "ok"
    except Exception as exc:
        status = "fallback_to_control"
        selected_labels = list(case.control_labels)
        coverage = "unverified"
        error = f"{type(exc).__name__}: {exc}"

    verifier_ms = (time.perf_counter() - start) * 1000
    selected_candidates = [
        views_by_label[label].candidate for label in selected_labels
    ]
    if len({page_key(item.document) for item in selected_candidates}) != 3:
        raise RuntimeError("Verifier selection contains duplicate pages.")

    control_retrieved = normalized_retrieval(
        candidates=case.control.selected,
        roles=case.control.selection_roles,
    )
    challenger_retrieved = normalize_selected(
        selected=selected_candidates,
        selected_labels=selected_labels,
        status=status,
    )
    control_units = [
        (str(item["doc_id"]), int(item["pages"][0]))
        for item in control_retrieved
    ]
    challenger_units = [
        (str(item["doc_id"]), int(item["pages"][0]))
        for item in challenger_retrieved
    ]

    return {
        "id": case.question_id,
        "question": case.question,
        "type": case.question_type,
        "answerable": case.answerable,
        "decomposition": {
            "applied": case.control.decomposed,
            "subqueries": case.control.subqueries,
        },
        "candidate_signature": candidate_signature(case.candidates),
        "candidate_sources": {
            view.label: list(view.query_sources) for view in case.candidates
        },
        "control_labels": case.control_labels,
        "selected_labels": selected_labels,
        "control_retrieved": control_retrieved,
        "challenger_retrieved": challenger_retrieved,
        "selection_changed": challenger_units != control_units,
        "verifier": {
            "status": status,
            "coverage": coverage,
            "raw_response": raw,
            "error": error,
        },
        "retrieval_ms": round(case.retrieval_ms, 3),
        "verifier_ms": round(verifier_ms, 3),
        "final_answer_generation_performed": False,
    }


def run_sweep() -> None:
    validate_inputs()
    validate_frozen_configuration()
    if not DATABASE_DIR.exists():
        raise FileNotFoundError(
            f"Frozen OCR index not found at {DATABASE_DIR}."
        )

    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
    gold_rows = read_jsonl(GOLD_FILE)
    completed = load_completed()
    expected_ids = {str(row["id"]) for row in gold_rows}
    extra_ids = sorted(set(completed) - expected_ids)
    if extra_ids:
        raise ValueError(f"Detail file contains unknown IDs: {extra_ids}")

    print("Controlled experiment: Qwen evidence-aware listwise reranking")
    print(f"Reusing frozen OCR index: {DATABASE_DIR.resolve()}")
    print(
        f"Frozen pipeline: {CHUNK_SIZE}/{CHUNK_OVERLAP} characters, "
        f"{EMBEDDING_MODEL}, dense + BM25 RRF, decomposition, top 3"
    )
    print(
        "Challenger: Qwen selects three collectively answer-bearing pages "
        "from frozen candidates"
    )
    print(
        f"Candidate cutoffs: direct top {DIRECT_POOL_CUTOFF}; decomposed "
        f"top {DECOMPOSED_POOL_CUTOFF} per frozen query"
    )
    print("Final answer generation: disabled")
    print("Gold evidence: post-selection evaluation only")

    vector_store = open_evaluation_vector_store()
    frozen_chunks = load_frozen_chunks(vector_store)
    bm25_index = BM25Index(frozen_chunks)
    cases: list[RetrievalCase] = []

    # Retrieve all questions together so the embedding model can be unloaded
    # before the verifier model is loaded.
    stop_ollama_model(GENERATION_MODEL)
    try:
        for position, gold in enumerate(gold_rows, start=1):
            start = time.perf_counter()
            question = str(gold["question"])
            control = retrieve_control_state(
                vector_store=vector_store,
                bm25_index=bm25_index,
                question=question,
            )
            runs = frozen_query_runs(
                vector_store=vector_store,
                bm25_index=bm25_index,
                question=question,
                control=control,
            )
            pool_limit = (
                DECOMPOSED_POOL_CUTOFF
                if control.decomposed
                else DIRECT_POOL_CUTOFF
            )
            pools = [
                (
                    run,
                    unique_page_pool(
                        fused_ranked=run.fused_ranked,
                        limit=pool_limit,
                    ),
                )
                for run in runs
            ]
            candidates, control_labels = build_candidate_shortlist(
                control=control,
                query_pools=pools,
            )
            elapsed_ms = (time.perf_counter() - start) * 1000
            case = RetrievalCase(
                question_id=str(gold["id"]),
                question=question,
                question_type=str(gold["type"]),
                answerable=bool(gold["answerable"]),
                control=control,
                candidates=candidates,
                control_labels=control_labels,
                retrieval_ms=elapsed_ms,
            )

            existing = completed.get(case.question_id)
            if (
                existing is not None
                and existing.get("candidate_signature")
                != candidate_signature(case.candidates)
            ):
                raise RuntimeError(
                    "A resumed candidate signature changed for "
                    f"{case.question_id}. Move or delete {DETAIL_FILE} before "
                    "starting a fresh experiment."
                )
            cases.append(case)
            print(
                f"  retrieve [{position:02d}/{len(gold_rows)}] "
                f"{case.question_id} candidates={len(candidates)} "
                f"{elapsed_ms:.1f} ms"
            )
    finally:
        stop_ollama_model(EMBEDDING_MODEL)

    llm = ChatOllama(
        model=GENERATION_MODEL,
        temperature=VERIFIER_TEMPERATURE,
        num_ctx=VERIFIER_NUM_CTX,
        num_predict=VERIFIER_NUM_PREDICT,
        reasoning=False,
    )

    ordered_rows: list[dict] = []
    try:
        for position, case in enumerate(cases, start=1):
            if case.question_id in completed:
                row = completed[case.question_id]
                action = "resumed"
            else:
                row = run_verifier(llm=llm, case=case)
                completed[case.question_id] = row
                write_jsonl(
                    DETAIL_FILE,
                    (
                        completed[str(gold["id"])]
                        for gold in gold_rows
                        if str(gold["id"]) in completed
                    ),
                )
                action = str(row["verifier"]["status"])
            ordered_rows.append(row)
            print(
                f"  verify  [{position:02d}/{len(cases)}] "
                f"{case.question_id} status={action} "
                f"changed={'yes' if row['selection_changed'] else 'no'} "
                f"{float(row['verifier_ms']):.1f} ms"
            )
    finally:
        stop_ollama_model(GENERATION_MODEL)

    results = aggregate_results(
        gold_rows=gold_rows,
        detail_rows=ordered_rows,
    )
    # Re-save details with post-selection metrics included.
    write_jsonl(DETAIL_FILE, ordered_rows)

    summary = {
        "experiment": "Part 2 Qwen evidence-aware listwise reranking",
        "independent_variable": (
            "frozen top-3 page policy versus Qwen listwise evidence selection"
        ),
        "controls": {
            "ocr": "frozen selective-OCR index",
            "chunk_size_characters": CHUNK_SIZE,
            "chunk_overlap_characters": CHUNK_OVERLAP,
            "embedding_model": EMBEDDING_MODEL,
            "retrieval": "dense top 20 plus BM25 top 20, equal-weight RRF",
            "query_policy": (
                "full question plus existing explicit both-and "
                "decomposition only"
            ),
            "final_top_k": FINAL_TOP_K,
            "final_answer_generation": "not run",
            "uses_gold_for_retrieval_prompt_or_selection": False,
            "gold_usage": "post-selection retrieval scoring only",
        },
        "challenger": {
            "model": GENERATION_MODEL,
            "method": (
                "single listwise verifier call per question with conservative "
                "control fallback"
            ),
            "temperature": VERIFIER_TEMPERATURE,
            "num_ctx": VERIFIER_NUM_CTX,
            "num_predict": VERIFIER_NUM_PREDICT,
            "direct_candidate_cutoff": DIRECT_POOL_CUTOFF,
            "decomposed_candidate_cutoff_per_query": (
                DECOMPOSED_POOL_CUTOFF
            ),
        },
        "eligibility_rule": (
            "Improve overall Recall@3; preserve Hit@3 and MRR@3; raise "
            "multi-chunk Recall@3 to at least 0.50; improve OCR multi-chunk "
            "Recall@3."
        ),
        **results,
        "decision": (
            "PROMOTE_TO_FULL_GENERATION_CONFIRMATION"
            if results["promoted"]
            else "DO_NOT_PROMOTE; AUDIT VERIFIER SELECTIONS"
        ),
        "files": {
            "question_level": str(DETAIL_FILE),
            "summary": str(SUMMARY_FILE),
        },
    }
    write_json(SUMMARY_FILE, summary)

    control = results["control_retrieval"]
    challenger = results["challenger_retrieval"]
    print("\nEvidence-aware reranking screening complete")
    print(
        "  control: "
        f"hit@3={control['hit_at_3']:.4f}, "
        f"recall@3={control['recall_at_3']:.4f}, "
        f"mrr@3={control['mrr_at_3']:.4f}"
    )
    print(
        "  challenger: "
        f"hit@3={challenger['hit_at_3']:.4f}, "
        f"recall@3={challenger['recall_at_3']:.4f}, "
        f"mrr@3={challenger['mrr_at_3']:.4f}"
    )
    print(f"  selection changes: {results['selection_changes']}")
    print(f"  verifier fallbacks: {results['verifier_fallbacks']}")
    print(
        "Decision: "
        + (
            "PROMOTE to one full generation confirmation"
            if results["promoted"]
            else "DO NOT PROMOTE; audit verifier selections"
        )
    )
    print(f"Summary saved to: {SUMMARY_FILE.resolve()}")


def main() -> None:
    run_sweep()


if __name__ == "__main__":
    main()
