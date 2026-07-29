from __future__ import annotations

import itertools
import json
import re
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from langchain_ollama import ChatOllama

from build_index import CHUNK_OVERLAP, CHUNK_SIZE, EMBEDDING_MODEL
from rag_query import GENERATION_MODEL, stop_ollama_model
from run_adaptive_evidence_coverage_sweep import (
    RoutePlan,
    route_question,
    validate_router_self_tests,
)
from run_balanced_hybrid_evaluation import FINAL_TOP_K, page_key
from run_candidate_headroom_audit import (
    frozen_query_runs,
    unique_page_pool,
)
from run_decomposed_hybrid_evaluation import decompose_question
from run_diversity_reranking_sweep import (
    ControlState,
    normalized_retrieval,
    retrieve_control_state,
)
from run_evidence_aware_reranking_sweep import (
    CandidateView,
    build_candidate_shortlist,
    candidate_prompt_block,
    candidate_signature,
    candidate_unit,
    find_json_object,
    validate_frozen_configuration,
)
from run_hybrid_evaluation import BM25Index, load_frozen_chunks
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
# - Keep the frozen selective-OCR index, 1000/200 chunks, mxbai embeddings,
#   dense + BM25 RRF retrieval, explicit both-and decomposition, and top-3
#   context budget.
# - Keep every direct question exactly on the frozen control.
# - Use the existing label-blind compound-question gate only to decide whether
#   a support-matrix call is eligible. Do not reuse the rejected adaptive
#   router's forced page-allocation policy.
# - For an eligible question, Qwen reports sparse candidate-to-need support
#   scores and short literal excerpts. Python validates every excerpt, builds
#   the complete matrix, and performs deterministic constrained selection.
# - A non-control page may enter only when the best three-page set covers more
#   validated atomic needs than the frozen control.
# - Gold evidence is used only after selection for retrieval evaluation.
# - This script does not generate final answers.
EXPERIMENT_DIR = RESULTS_DIR / "compound_support_matrix_sweep"
DETAIL_FILE = EXPERIMENT_DIR / "compound_support_matrix_by_question.jsonl"
SUMMARY_FILE = EXPERIMENT_DIR / "compound_support_matrix_summary.json"

DIRECT_POOL_CUTOFF = 10
DECOMPOSED_POOL_CUTOFF = 3
VERIFIER_TEMPERATURE = 0.0
VERIFIER_NUM_CTX = 4096
VERIFIER_NUM_PREDICT = 512
MIN_QUOTE_CHARACTERS = 12
MIN_QUOTE_TOKENS = 3


@dataclass(frozen=True)
class EvidenceNeed:
    label: str
    text: str
    source_file: str | None


@dataclass(frozen=True)
class MatrixCase:
    question_id: str
    question: str
    question_type: str
    answerable: bool
    route: RoutePlan
    needs: list[EvidenceNeed]
    control: ControlState
    candidates: list[CandidateView]
    control_labels: list[str]
    retrieval_ms: float


@dataclass(frozen=True)
class ParsedSupport:
    candidate_label: str
    need_label: str
    score: int
    quote: str


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


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def normalize_label(value: object, prefix: str) -> str | None:
    match = re.fullmatch(
        rf"{re.escape(prefix)}0*(\d+)",
        str(value).strip().upper(),
    )
    if match is None:
        return None
    number = int(match.group(1))
    if number <= 0:
        return None
    return f"{prefix}{number:02d}"


def infer_source_files(
    control: ControlState,
    requested_count: int,
    fixed_sources: list[str],
) -> list[str]:
    """
    Infer document affinity from the unchanged full-question RRF ranking.

    This uses only live retrieval ranks and ordinary corpus metadata. It does
    not inspect benchmark IDs, question types, reference answers, or gold
    evidence.
    """
    selected = [
        source for source in fixed_sources if source in SOURCE_TO_DOC_ID
    ]
    best_rank: dict[str, int] = {}
    for rank, candidate in enumerate(control.candidate_pool, start=1):
        source = str(candidate.document.metadata.get("source", ""))
        if source in SOURCE_TO_DOC_ID:
            best_rank.setdefault(source, rank)

    ranked_sources = sorted(
        SOURCE_TO_DOC_ID,
        key=lambda source: (best_rank.get(source, 10**9), source),
    )
    for source in ranked_sources:
        if source not in selected:
            selected.append(source)
        if len(selected) >= requested_count:
            break
    if len(selected) < requested_count:
        raise RuntimeError(
            f"Could infer only {len(selected)} of {requested_count} sources."
        )
    return selected[:requested_count]


def build_evidence_needs(
    question: str,
    route: RoutePlan,
    control: ControlState,
) -> list[EvidenceNeed]:
    if not route.routed:
        return []

    if route.mode == "atomic":
        both_and_queries = decompose_question(question)
        raw_queries = (
            both_and_queries
            if len(both_and_queries) > 1
            else route.atomic_queries
        )
        queries = [normalize_space(item) for item in raw_queries]
        if not 1 < len(queries) <= FINAL_TOP_K:
            raise RuntimeError(
                "Atomic compound route must expose two or three needs."
            )
        return [
            EvidenceNeed(
                label=f"N{index:02d}",
                text=query,
                source_file=None,
            )
            for index, query in enumerate(queries, start=1)
        ]

    if route.mode == "multi_source":
        requested_count = max(
            2,
            len(route.explicit_sources),
            route.inferred_source_count,
        )
        requested_count = min(requested_count, FINAL_TOP_K)
        sources = infer_source_files(
            control=control,
            requested_count=requested_count,
            fixed_sources=list(route.explicit_sources),
        )
        return [
            EvidenceNeed(
                label=f"N{index:02d}",
                text=(
                    f"Find the direct evidence in document "
                    f"{SOURCE_TO_DOC_ID[source]} that answers its part of: "
                    f"{question}"
                ),
                source_file=source,
            )
            for index, source in enumerate(sources, start=1)
        ]

    raise RuntimeError(f"Unsupported routed mode: {route.mode}")


def need_signature(needs: list[EvidenceNeed]) -> list[dict]:
    return [
        {
            "label": need.label,
            "text": need.text,
            "source_file": need.source_file,
        }
        for need in needs
    ]


def matrix_messages(case: MatrixCase) -> list[tuple[str, str]]:
    need_lines = []
    for need in case.needs:
        scope = (
            "any supplied document"
            if need.source_file is None
            else SOURCE_TO_DOC_ID[need.source_file]
        )
        need_lines.append(
            f"[{need.label}] source_scope={scope}\n{need.text}"
        )
    candidate_blocks = "\n\n---\n\n".join(
        candidate_prompt_block(view) for view in case.candidates
    )
    return [
        (
            "system",
            (
                "You are an evidence-support classifier for a RAG system. "
                "Do not select final pages and do not answer the question. "
                "Assess each supplied candidate passage independently against "
                "each atomic evidence need. Use score 2 only when the passage "
                "directly states facts that answer the need. Use score 1 when "
                "it is relevant but incomplete. Omit score-0 pairs. Every "
                "reported pair must include a short exact excerpt copied "
                "verbatim from that candidate passage; never paraphrase or "
                "invent an excerpt. Respect source_scope: a candidate from a "
                "different document has score 0. Return only one JSON object "
                "with this shape: "
                '{"support":[{"candidate":"C01","need":"N01",'
                '"score":2,"quote":"exact copied words"}]}. '
                "Do not include prose, Markdown, coverage claims, rankings, "
                "or recommendations."
            ),
        ),
        (
            "human",
            (
                f"Question:\n{case.question}\n\n"
                "Atomic evidence needs:\n"
                f"{chr(10).join(need_lines)}\n\n"
                "Candidate passages:\n"
                f"{candidate_blocks}"
            ),
        ),
    ]


def validate_literal_quote(quote: object, passage: str) -> tuple[bool, str]:
    cleaned = normalize_space(str(quote)).strip(
        " \t\r\n\"'`“”‘’"
    )
    if len(cleaned) < MIN_QUOTE_CHARACTERS:
        return False, cleaned
    if len(cleaned.split()) < MIN_QUOTE_TOKENS:
        return False, cleaned
    normalized_passage = normalize_space(passage).casefold()
    return cleaned.casefold() in normalized_passage, cleaned


def parse_sparse_support(
    raw: str,
    case: MatrixCase,
) -> tuple[list[ParsedSupport], list[dict]]:
    value = find_json_object(raw)
    supplied = value.get("support")
    if not isinstance(supplied, list):
        raise ValueError("Verifier JSON has no support list.")

    views_by_label = {view.label: view for view in case.candidates}
    needs_by_label = {need.label: need for need in case.needs}
    accepted: dict[tuple[str, str], ParsedSupport] = {}
    rejected: list[dict] = []

    for position, item in enumerate(supplied, start=1):
        if not isinstance(item, dict):
            rejected.append(
                {"position": position, "reason": "entry_not_object"}
            )
            continue
        candidate_label = normalize_label(item.get("candidate"), "C")
        need_label = normalize_label(item.get("need"), "N")
        if candidate_label not in views_by_label:
            rejected.append(
                {
                    "position": position,
                    "reason": "invalid_candidate",
                    "value": item.get("candidate"),
                }
            )
            continue
        if need_label not in needs_by_label:
            rejected.append(
                {
                    "position": position,
                    "reason": "invalid_need",
                    "value": item.get("need"),
                }
            )
            continue
        try:
            score = int(item.get("score"))
        except (TypeError, ValueError):
            score = -1
        if score not in (1, 2):
            rejected.append(
                {
                    "position": position,
                    "reason": "score_not_1_or_2",
                    "value": item.get("score"),
                }
            )
            continue

        view = views_by_label[candidate_label]
        need = needs_by_label[need_label]
        candidate_source = str(
            view.candidate.document.metadata.get("source", "")
        )
        if (
            need.source_file is not None
            and candidate_source != need.source_file
        ):
            rejected.append(
                {
                    "position": position,
                    "reason": "source_scope_mismatch",
                    "candidate": candidate_label,
                    "need": need_label,
                }
            )
            continue

        quote_valid, cleaned_quote = validate_literal_quote(
            item.get("quote", ""),
            view.candidate.document.page_content,
        )
        if not quote_valid:
            rejected.append(
                {
                    "position": position,
                    "reason": "quote_not_literal_or_too_short",
                    "candidate": candidate_label,
                    "need": need_label,
                    "quote": cleaned_quote,
                }
            )
            continue

        parsed = ParsedSupport(
            candidate_label=candidate_label,
            need_label=need_label,
            score=score,
            quote=cleaned_quote,
        )
        key = (candidate_label, need_label)
        previous = accepted.get(key)
        if previous is None or (parsed.score, len(parsed.quote)) > (
            previous.score,
            len(previous.quote),
        ):
            accepted[key] = parsed

    return list(accepted.values()), rejected


def complete_support_matrix(
    case: MatrixCase,
    accepted: list[ParsedSupport],
) -> list[dict]:
    accepted_by_pair = {
        (item.candidate_label, item.need_label): item
        for item in accepted
    }
    rows: list[dict] = []
    for view in case.candidates:
        support: list[dict] = []
        for need in case.needs:
            item = accepted_by_pair.get((view.label, need.label))
            support.append(
                {
                    "need": need.label,
                    "score": 0 if item is None else item.score,
                    "quote": "" if item is None else item.quote,
                    "literal_quote_validated": item is not None,
                }
            )
        unit = candidate_unit(view.candidate)
        rows.append(
            {
                "candidate": view.label,
                "doc_id": unit[0],
                "page": unit[1],
                "control_rank": view.control_rank,
                "support": support,
            }
        )
    return rows


def support_maps(
    matrix: list[dict],
) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    direct: dict[str, set[str]] = {}
    partial: dict[str, set[str]] = {}
    for row in matrix:
        label = str(row["candidate"])
        direct[label] = {
            str(item["need"])
            for item in row["support"]
            if int(item["score"]) == 2
        }
        partial[label] = {
            str(item["need"])
            for item in row["support"]
            if int(item["score"]) >= 1
        }
    return direct, partial


def covered_needs(
    labels: Iterable[str],
    support: dict[str, set[str]],
) -> set[str]:
    result: set[str] = set()
    for label in labels:
        result.update(support.get(label, set()))
    return result


def guarded_select(
    case: MatrixCase,
    matrix: list[dict],
) -> tuple[list[str], dict]:
    """
    Choose exactly three pages using a conservative lexicographic objective.

    First maximize validated direct-need coverage. Among equal-coverage sets,
    retain as many frozen control pages as possible. Further ties prefer more
    direct support pairs, more partial coverage, and earlier frozen-candidate
    order. The challenger changes only for a strict coverage gain.
    """
    candidate_labels = [view.label for view in case.candidates]
    candidate_order = {
        label: index for index, label in enumerate(candidate_labels)
    }
    control_set = set(case.control_labels)
    direct, partial = support_maps(matrix)
    control_coverage = covered_needs(case.control_labels, direct)

    ranked: list[tuple[tuple, tuple[str, ...], set[str]]] = []
    for labels in itertools.combinations(candidate_labels, FINAL_TOP_K):
        direct_coverage = covered_needs(labels, direct)
        direct_pairs = sum(len(direct[label]) for label in labels)
        partial_coverage = covered_needs(labels, partial)
        outside_count = sum(label not in control_set for label in labels)
        order_sum = sum(candidate_order[label] for label in labels)
        key = (
            -len(direct_coverage),
            outside_count,
            -direct_pairs,
            -len(partial_coverage),
            order_sum,
            tuple(candidate_order[label] for label in labels),
        )
        ranked.append((key, labels, direct_coverage))
    if not ranked:
        raise RuntimeError("No valid three-page candidate combinations.")
    ranked.sort(key=lambda item: item[0])
    _key, best_labels_tuple, best_coverage = ranked[0]

    if len(best_coverage) <= len(control_coverage):
        return list(case.control_labels), {
            "changed": False,
            "reason": "no_strict_validated_coverage_gain",
            "control_direct_coverage": sorted(control_coverage),
            "best_direct_coverage": sorted(best_coverage),
        }

    best_set = set(best_labels_tuple)
    retained_control = [
        label for label in case.control_labels if label in best_set
    ]
    outsiders = sorted(
        (label for label in best_labels_tuple if label not in control_set),
        key=lambda label: candidate_order[label],
    )
    selected = [*retained_control, *outsiders]
    if len(selected) != FINAL_TOP_K:
        raise RuntimeError("Guarded selector did not produce exactly 3 pages.")
    return selected, {
        "changed": selected != case.control_labels,
        "reason": "strict_validated_coverage_gain",
        "control_direct_coverage": sorted(control_coverage),
        "best_direct_coverage": sorted(best_coverage),
    }


def normalized_selected(
    case: MatrixCase,
    selected_labels: list[str],
    status: str,
) -> list[dict]:
    views_by_label = {view.label: view for view in case.candidates}
    candidates = [views_by_label[label].candidate for label in selected_labels]
    if len({page_key(item.document) for item in candidates}) != FINAL_TOP_K:
        raise RuntimeError("Final selection contains duplicate pages.")
    roles = [
        (
            f"support_matrix_{label}"
            if status == "ok"
            else f"frozen_control_{status}"
        )
        for label in selected_labels
    ]
    return normalized_retrieval(candidates=candidates, roles=roles)


def retrieval_units(retrieved: list[dict]) -> list[tuple[str, int]]:
    return [
        (str(item["doc_id"]), int(item["pages"][0]))
        for item in retrieved
    ]


def direct_control_row(case: MatrixCase) -> dict:
    control_retrieved = normalized_retrieval(
        candidates=case.control.selected,
        roles=case.control.selection_roles,
    )
    return {
        "id": case.question_id,
        "question": case.question,
        "type": case.question_type,
        "answerable": case.answerable,
        "route": {
            "mode": case.route.mode,
            "reason": case.route.reason,
            "eligible": False,
        },
        "needs": [],
        "candidate_signature": candidate_signature(case.candidates),
        "control_labels": case.control_labels,
        "selected_labels": case.control_labels,
        "support_matrix": [],
        "control_retrieved": control_retrieved,
        "challenger_retrieved": control_retrieved,
        "selection_changed": False,
        "selection": {
            "changed": False,
            "reason": "direct_question_frozen",
            "control_direct_coverage": [],
            "best_direct_coverage": [],
        },
        "verifier": {
            "status": "not_applicable_direct_question",
            "raw_response": "",
            "error": None,
            "accepted_support_pairs": 0,
            "rejected_entries": [],
        },
        "retrieval_ms": round(case.retrieval_ms, 3),
        "verifier_ms": 0.0,
        "final_answer_generation_performed": False,
    }


def run_matrix_verifier(llm: ChatOllama, case: MatrixCase) -> dict:
    start = time.perf_counter()
    raw = ""
    error: str | None = None
    accepted: list[ParsedSupport] = []
    rejected: list[dict] = []
    try:
        response = llm.invoke(matrix_messages(case))
        raw = str(response.content).strip()
        accepted, rejected = parse_sparse_support(raw=raw, case=case)
        matrix = complete_support_matrix(case=case, accepted=accepted)
        selected_labels, selection = guarded_select(case=case, matrix=matrix)
        status = "ok"
    except Exception as exc:
        status = "fallback_to_control"
        selected_labels = list(case.control_labels)
        selection = {
            "changed": False,
            "reason": "verifier_or_parser_failure",
            "control_direct_coverage": [],
            "best_direct_coverage": [],
        }
        matrix = complete_support_matrix(case=case, accepted=[])
        error = f"{type(exc).__name__}: {exc}"
    verifier_ms = (time.perf_counter() - start) * 1000

    control_retrieved = normalized_retrieval(
        candidates=case.control.selected,
        roles=case.control.selection_roles,
    )
    challenger_retrieved = normalized_selected(
        case=case,
        selected_labels=selected_labels,
        status=status,
    )
    changed = (
        retrieval_units(challenger_retrieved)
        != retrieval_units(control_retrieved)
    )
    if changed != bool(selection["changed"]):
        raise RuntimeError("Selection-change bookkeeping is inconsistent.")

    return {
        "id": case.question_id,
        "question": case.question,
        "type": case.question_type,
        "answerable": case.answerable,
        "route": {
            "mode": case.route.mode,
            "reason": case.route.reason,
            "eligible": True,
        },
        "needs": need_signature(case.needs),
        "candidate_signature": candidate_signature(case.candidates),
        "control_labels": case.control_labels,
        "selected_labels": selected_labels,
        "support_matrix": matrix,
        "control_retrieved": control_retrieved,
        "challenger_retrieved": challenger_retrieved,
        "selection_changed": changed,
        "selection": selection,
        "verifier": {
            "status": status,
            "raw_response": raw,
            "error": error,
            "accepted_support_pairs": len(accepted),
            "rejected_entries": rejected,
        },
        "retrieval_ms": round(case.retrieval_ms, 3),
        "verifier_ms": round(verifier_ms, 3),
        "final_answer_generation_performed": False,
    }


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
    gold_evidence: list[dict],
) -> dict[str, float]:
    """
    Post-selection evaluation boundary.

    This is the only function that receives gold evidence. Retrieval, routing,
    need construction, Qwen, parsing, excerpt validation, and selection never
    receive benchmark evidence labels.
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
    routed_rows: list[dict] = []
    for gold in gold_rows:
        row = details_by_id[str(gold["id"])]
        if not bool(row["route"]["eligible"]):
            if retrieval_units(row["control_retrieved"]) != retrieval_units(
                row["challenger_retrieved"]
            ):
                raise RuntimeError(
                    "A direct question changed despite the frozen guard."
                )
        else:
            routed_rows.append(row)

        if not bool(gold["answerable"]):
            continue
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
        "routed_questions": len(routed_rows),
        "selection_changes": sum(
            int(bool(row["selection_changed"])) for row in detail_rows
        ),
        "verifier_fallbacks": sum(
            int(str(row["verifier"]["status"]) == "fallback_to_control")
            for row in routed_rows
        ),
        "accepted_support_pairs": sum(
            int(row["verifier"]["accepted_support_pairs"])
            for row in routed_rows
        ),
        "rejected_support_entries": sum(
            len(row["verifier"]["rejected_entries"])
            for row in routed_rows
        ),
        "mean_routed_verifier_ms": (
            statistics.fmean(
                float(row["verifier_ms"]) for row in routed_rows
            )
            if routed_rows
            else 0.0
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


def run_sweep() -> None:
    validate_inputs()
    validate_frozen_configuration()
    validate_router_self_tests()
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

    print("Controlled experiment: compound-only evidence-support matrix")
    print(f"Reusing frozen OCR index: {DATABASE_DIR.resolve()}")
    print(
        f"Frozen pipeline: {CHUNK_SIZE}/{CHUNK_OVERLAP} characters, "
        f"{EMBEDDING_MODEL}, dense + BM25 RRF, decomposition, top 3"
    )
    print(
        "Direct questions: frozen control only; no verifier call or "
        "selection change allowed"
    )
    print(
        "Compound questions: Qwen supplies sparse support scores plus "
        "literal excerpts; Python validates and selects"
    )
    print(
        f"Candidate cutoffs: direct-route pool top {DIRECT_POOL_CUTOFF}; "
        f"decomposed pool top {DECOMPOSED_POOL_CUTOFF} per frozen query"
    )
    print("Final answer generation: disabled")
    print("Gold evidence: post-selection evaluation only")

    vector_store = open_evaluation_vector_store()
    frozen_chunks = load_frozen_chunks(vector_store)
    bm25_index = BM25Index(frozen_chunks)
    cases: list[MatrixCase] = []

    stop_ollama_model(GENERATION_MODEL)
    try:
        for position, gold in enumerate(gold_rows, start=1):
            start = time.perf_counter()
            question = str(gold["question"])
            route = route_question(question)
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
            needs = build_evidence_needs(
                question=question,
                route=route,
                control=control,
            )
            elapsed_ms = (time.perf_counter() - start) * 1000
            case = MatrixCase(
                question_id=str(gold["id"]),
                question=question,
                question_type=str(gold["type"]),
                answerable=bool(gold["answerable"]),
                route=route,
                needs=needs,
                control=control,
                candidates=candidates,
                control_labels=control_labels,
                retrieval_ms=elapsed_ms,
            )

            existing = completed.get(case.question_id)
            if existing is not None:
                if existing.get("candidate_signature") != candidate_signature(
                    case.candidates
                ):
                    raise RuntimeError(
                        "A resumed candidate signature changed for "
                        f"{case.question_id}. Move or delete {DETAIL_FILE} "
                        "before starting a fresh experiment."
                    )
                if existing.get("needs") != need_signature(case.needs):
                    raise RuntimeError(
                        "A resumed evidence-need signature changed for "
                        f"{case.question_id}. Move or delete {DETAIL_FILE} "
                        "before starting a fresh experiment."
                    )
            cases.append(case)
            print(
                f"  retrieve [{position:02d}/{len(gold_rows)}] "
                f"{case.question_id} route={route.mode} "
                f"needs={len(needs)} candidates={len(candidates)} "
                f"{elapsed_ms:.1f} ms"
            )
    finally:
        stop_ollama_model(EMBEDDING_MODEL)

    pending_routed = [
        case
        for case in cases
        if case.route.routed and case.question_id not in completed
    ]
    llm = (
        ChatOllama(
            model=GENERATION_MODEL,
            temperature=VERIFIER_TEMPERATURE,
            num_ctx=VERIFIER_NUM_CTX,
            num_predict=VERIFIER_NUM_PREDICT,
            reasoning=False,
        )
        if pending_routed
        else None
    )

    ordered_rows: list[dict] = []
    try:
        for position, case in enumerate(cases, start=1):
            if case.question_id in completed:
                row = completed[case.question_id]
                action = "resumed"
            elif not case.route.routed:
                row = direct_control_row(case)
                completed[case.question_id] = row
                action = "direct_frozen"
            else:
                if llm is None:
                    raise RuntimeError("Verifier model was not initialized.")
                row = run_matrix_verifier(llm=llm, case=case)
                completed[case.question_id] = row
                action = str(row["verifier"]["status"])

            write_jsonl(
                DETAIL_FILE,
                (
                    completed[str(gold["id"])]
                    for gold in gold_rows
                    if str(gold["id"]) in completed
                ),
            )
            ordered_rows.append(row)
            print(
                f"  verify  [{position:02d}/{len(cases)}] "
                f"{case.question_id} status={action} "
                f"changed={'yes' if row['selection_changed'] else 'no'} "
                f"{float(row['verifier_ms']):.1f} ms"
            )
    finally:
        if llm is not None:
            stop_ollama_model(GENERATION_MODEL)

    results = aggregate_results(
        gold_rows=gold_rows,
        detail_rows=ordered_rows,
    )
    write_jsonl(DETAIL_FILE, ordered_rows)

    summary = {
        "experiment": "Part 2 compound-only evidence-support matrix",
        "independent_variable": (
            "frozen top-3 page policy versus guarded deterministic selection "
            "from a Qwen candidate-to-need support matrix on routed compound "
            "questions only"
        ),
        "controls": {
            "ocr": "frozen selective-OCR index",
            "chunk_size_characters": CHUNK_SIZE,
            "chunk_overlap_characters": CHUNK_OVERLAP,
            "embedding_model": EMBEDDING_MODEL,
            "retrieval": "dense top 20 plus BM25 top 20, equal-weight RRF",
            "frozen_query_policy": (
                "full question plus existing explicit both-and "
                "decomposition only"
            ),
            "direct_question_policy": (
                "frozen control; support verifier not called"
            ),
            "final_top_k": FINAL_TOP_K,
            "final_answer_generation": "not run",
            "uses_gold_for_routing_prompt_matrix_or_selection": False,
            "gold_usage": "post-selection retrieval scoring only",
        },
        "challenger": {
            "eligibility_gate": (
                "existing label-blind compound router; allocation logic from "
                "the rejected adaptive experiment is not reused"
            ),
            "model": GENERATION_MODEL,
            "method": (
                "one sparse support-matrix call per routed question; literal "
                "excerpt validation; deterministic coverage selection"
            ),
            "temperature": VERIFIER_TEMPERATURE,
            "num_ctx": VERIFIER_NUM_CTX,
            "num_predict": VERIFIER_NUM_PREDICT,
            "direct_candidate_cutoff": DIRECT_POOL_CUTOFF,
            "decomposed_candidate_cutoff_per_query": (
                DECOMPOSED_POOL_CUTOFF
            ),
            "replacement_guard": (
                "change frozen pages only for a strict increase in validated "
                "score-2 atomic-need coverage"
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
            else "DO_NOT_PROMOTE; AUDIT SUPPORT MATRICES"
        ),
        "files": {
            "question_level": str(DETAIL_FILE),
            "summary": str(SUMMARY_FILE),
        },
    }
    write_json(SUMMARY_FILE, summary)

    control = results["control_retrieval"]
    challenger = results["challenger_retrieval"]
    print("\nCompound support-matrix screening complete")
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
    print(f"  routed questions: {results['routed_questions']}")
    print(f"  selection changes: {results['selection_changes']}")
    print(f"  verifier fallbacks: {results['verifier_fallbacks']}")
    print(
        f"  accepted support pairs: "
        f"{results['accepted_support_pairs']}"
    )
    print(
        f"  rejected support entries: "
        f"{results['rejected_support_entries']}"
    )
    print(
        "Decision: "
        + (
            "PROMOTE TO FULL GENERATION CONFIRMATION"
            if results["promoted"]
            else "DO NOT PROMOTE; audit support matrices"
        )
    )
    print(f"Summary saved to: {SUMMARY_FILE.resolve()}")


def main() -> None:
    try:
        run_sweep()
    except KeyboardInterrupt:
        print("\nInterrupted. Completed rows were saved; rerun to resume.")
        raise SystemExit(130)


if __name__ == "__main__":
    main()
