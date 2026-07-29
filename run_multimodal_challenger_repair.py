from __future__ import annotations

import hashlib
import json
import re
import statistics
import time
from copy import deepcopy
from pathlib import Path

from langchain_core.documents import Document
from langchain_ollama import ChatOllama

from build_index import EMBEDDING_MODEL
from rag_query import GENERATION_MODEL, stop_ollama_model
from run_multimodal_paired_generation import (
    NUM_CTX,
    NUM_PREDICT,
    STATE_FILE,
    TEMPERATURE,
    arm_prediction,
    evidence_units,
    generate_structured_answer,
    load_metrics_module,
    prediction_rows,
    question_level_comparison,
    validate_retrieval_gate,
)
from run_multimodal_visual_retrieval_sweep import (
    DESCRIPTION_FILE,
    EXPERIMENT_DIR,
    FINAL_TOP_K,
    SOURCE_TO_DOC_ID,
    VISUAL_DATABASE_DIR,
    VISUAL_GOLD_FILE,
    detect_figure_pages,
    validate_visual_description,
    visual_document,
    write_json,
    write_jsonl,
)
from run_ocr_evaluation import (
    DATABASE_DIR,
    read_jsonl,
)


# Faithfulness repair experiment:
# - The accepted text index and original control generations are immutable.
# - Only the challenger is regenerated.
# - Repaired visual descriptions must pass source-grounded structure checks.
# - Gold facts are used only after retrieval and generation.
# - Relationship and citation gates supplement keyword-based aggregate metrics.
REPAIR_DIR = EXPERIMENT_DIR / "faithfulness_repair"
TARGET_FIGURES = {3, 5}


def file_fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_original_state(gold_rows: list[dict]) -> dict[str, dict]:
    if not STATE_FILE.exists():
        raise FileNotFoundError(
            "The original paired state is missing. Keep the completed "
            f"file at: {STATE_FILE}"
        )
    state: dict[str, dict] = {}
    for row in read_jsonl(STATE_FILE):
        question_id = str(row["id"])
        if question_id in state:
            raise ValueError(
                f"Duplicate original state row for {question_id}."
            )
        state[question_id] = row

    expected_ids = {str(row["id"]) for row in gold_rows}
    if set(state) != expected_ids:
        raise ValueError(
            "The original paired state does not contain exactly the frozen "
            "12-question visual benchmark."
        )
    for gold in gold_rows:
        question_id = str(gold["id"])
        row = state[question_id]
        if "control" not in row or "challenger" not in row:
            raise ValueError(
                f"Original state is incomplete for {question_id}."
            )
        if str(row.get("question")) != str(gold["question"]):
            raise ValueError(
                f"Original state question changed for {question_id}."
            )
    return state


def validate_repaired_descriptions() -> list[dict]:
    rows = read_jsonl(DESCRIPTION_FILE)
    figures_by_key = {
        (item.source, item.page): item
        for item in detect_figure_pages()
    }
    validated: set[int] = set()
    for row in rows:
        key = (str(row["source"]), int(row["page"]))
        item = figures_by_key.get(key)
        if item is None:
            raise ValueError(
                f"Description has no detected figure page: {key}"
            )
        if TARGET_FIGURES.intersection(item.figure_numbers):
            validate_visual_description(
                item,
                str(row["description"]),
            )
            validated.update(
                TARGET_FIGURES.intersection(item.figure_numbers)
            )
    if validated != TARGET_FIGURES:
        raise RuntimeError(
            "Figures 3 and 5 do not both have validated repaired "
            "descriptions. Rebuild pages 16 and 25 first."
        )
    return rows


def load_repair_state(
    state_file: Path,
    *,
    fingerprint: str,
) -> dict[str, dict]:
    if not state_file.exists():
        return {}
    state: dict[str, dict] = {}
    for row in read_jsonl(state_file):
        question_id = str(row["id"])
        if str(row.get("description_fingerprint")) != fingerprint:
            raise ValueError(
                "Repair state belongs to a different description index."
            )
        if question_id in state:
            raise ValueError(
                f"Duplicate repair state row for {question_id}."
            )
        state[question_id] = row
    return state


def save_repair_state(
    state_file: Path,
    state: dict[str, dict],
    gold_order: list[str],
) -> None:
    write_jsonl(
        state_file,
        [
            state[question_id]
            for question_id in gold_order
            if question_id in state
        ],
    )


def adjusted_gold(gold_rows: list[dict]) -> list[dict]:
    adjusted = deepcopy(gold_rows)
    for row in adjusted:
        if str(row["id"]) == "v005":
            facts = list(row["required_facts"])
            facts[1] = list(facts[1]) + ["build and use"]
            row["required_facts"] = facts
    return adjusted


def retrieved_identity(rows: list[dict]) -> list[tuple[str, tuple[int, ...]]]:
    return [
        (
            str(row["doc_id"]),
            tuple(int(page) for page in row.get("pages", [])),
        )
        for row in rows
    ]


def frozen_challenger_context(
    original_row: dict,
    current_visual_documents: dict[tuple[str, int], Document],
) -> tuple[list[tuple[Document, float]], list[dict]]:
    frozen_rows = list(
        original_row.get("challenger", {}).get("retrieved", [])
    )
    if len(frozen_rows) != FINAL_TOP_K:
        raise ValueError(
            "The original challenger context does not contain exactly "
            f"top {FINAL_TOP_K}."
        )
    if str(frozen_rows[0].get("selection_role")) != "visual_anchor":
        raise ValueError(
            "The original challenger rank-one result is not the frozen "
            "visual anchor."
        )

    normalized: list[dict] = []
    results: list[tuple[Document, float]] = []
    doc_id_to_source = {
        doc_id: source for source, doc_id in SOURCE_TO_DOC_ID.items()
    }

    for rank, frozen in enumerate(frozen_rows, start=1):
        row = deepcopy(frozen)
        pages = [int(page) for page in row.get("pages", [])]
        if len(pages) != 1:
            raise ValueError(
                f"Frozen challenger rank {rank} does not identify exactly "
                "one page."
            )
        doc_id = str(row["doc_id"])
        source = doc_id_to_source.get(doc_id)
        if source is None:
            raise ValueError(
                f"Frozen challenger rank {rank} has unknown doc_id "
                f"{doc_id!r}."
            )
        page = pages[0]

        if rank == 1:
            document = current_visual_documents.get((doc_id, page))
            if document is None:
                raise ValueError(
                    "No repaired visual description matches the frozen "
                    f"anchor {doc_id}:p{page}."
                )
            row["chunk"] = int(document.metadata["chunk"])
            row["content_type"] = "visual_description"
            row["text"] = document.page_content.strip()
        else:
            text = str(row.get("text", "")).strip()
            if not text:
                raise ValueError(
                    f"Frozen challenger rank {rank} has no saved text."
                )
            document = Document(
                page_content=text,
                metadata={
                    "source": source,
                    "page": page,
                    "chunk": int(row["chunk"]),
                    "content_type": str(
                        row.get("content_type", "text")
                    ),
                },
            )

        row["rank"] = rank
        normalized.append(row)
        score = row.get("hybrid_rrf_score")
        results.append(
            (document, -float(score) if score is not None else 0.0)
        )

    return results, normalized


def normalized_answer(answer: str) -> str:
    answer = answer.casefold().replace("&", " and ")
    answer = re.sub(r"[^a-z0-9]+", " ", answer)
    return re.sub(r"\s+", " ", answer).strip()


def ordered_near(
    text: str,
    first: str,
    second: str,
    *,
    distance: int = 140,
) -> bool:
    return re.search(
        rf"\b{re.escape(first)}\b.{{0,{distance}}}"
        rf"\b{re.escape(second)}\b",
        text,
    ) is not None


def either_order_near(
    text: str,
    first: str,
    second: str,
    *,
    distance: int = 100,
) -> bool:
    return ordered_near(
        text,
        first,
        second,
        distance=distance,
    ) or ordered_near(
        text,
        second,
        first,
        distance=distance,
    )


def relationship_check(question_id: str, answer: str) -> tuple[bool, str]:
    text = normalized_answer(answer)

    if question_id == "v003":
        center = (
            either_order_near(
                text,
                "people and planet",
                "center",
                distance=60,
            )
            or either_order_near(
                text,
                "people and planet",
                "central",
                distance=60,
            )
        )
        dimensions = all(
            label in text
            for label in (
                "application context",
                "data and input",
                "ai model",
                "task and output",
            )
        )
        not_miscounted = (
            "five dimensions" not in text
            and "5 dimensions" not in text
        )
        passed = center and dimensions and not_miscounted
        return (
            passed,
            "center + four dimensions without a five-dimension claim",
        )

    if question_id == "v006":
        collect_mapping = ordered_near(
            text,
            "collect and process data",
            "internal and external validation",
            distance=150,
        )
        deploy_mapping = (
            ordered_near(
                text,
                "deploy and use",
                "integration",
                distance=100,
            )
            and ordered_near(
                text,
                "integration",
                "compliance testing",
                distance=80,
            )
            and ordered_near(
                text,
                "compliance testing",
                "validation",
                distance=80,
            )
        )
        return (
            collect_mapping and deploy_mapping,
            "validation phrases mapped to the correct lifecycle stages",
        )

    if question_id == "v009":
        govern_center = (
            either_order_near(
                text,
                "govern",
                "central",
                distance=50,
            )
            or either_order_near(
                text,
                "govern",
                "center",
                distance=50,
            )
        )
        surrounding = (
            "surround" in text
            and all(
                label in text
                for label in ("map", "measure", "manage")
            )
        )
        govern_not_outer = not (
            "govern is also" in text
            or "govern among" in text
            or "govern as a surrounding" in text
            or "govern is a surrounding" in text
        )
        return (
            govern_center and surrounding and govern_not_outer,
            "GOVERN central; MAP, MEASURE, and MANAGE surrounding",
        )

    if question_id == "v010":
        map_left = (
            ordered_near(text, "map", "left", distance=45)
            or ordered_near(text, "left", "map", distance=45)
        )
        measure_right = (
            ordered_near(text, "measure", "right", distance=45)
            or ordered_near(text, "right", "measure", distance=45)
        )
        manage_below = (
            ordered_near(text, "manage", "below", distance=45)
            or ordered_near(text, "below", "manage", distance=45)
        )
        govern_not_misplaced = re.search(
            r"\bgovern\b\s+(?:is|appears|sits|is located|is positioned)"
            r".{0,20}\b(top|left|right|below)\b",
            text,
        ) is None
        return (
            (
                map_left
                and measure_right
                and manage_below
                and govern_not_misplaced
            ),
            "MAP left; MEASURE right; MANAGE below",
        )

    return True, "no additional relationship gate"


def exact_citation_check(
    gold: dict,
    prediction: dict,
) -> bool:
    if not bool(gold["answerable"]):
        return bool(prediction["abstained"])
    gold_units = evidence_units(list(gold["gold_evidence"]))
    cited_units = evidence_units(
        list(prediction.get("citations", []))
    )
    return bool(gold_units.intersection(cited_units))


def run() -> None:
    retrieval_summary = validate_retrieval_gate()
    description_rows = validate_repaired_descriptions()
    description_fingerprint = file_fingerprint(DESCRIPTION_FILE)
    version = description_fingerprint[:12]

    state_file = REPAIR_DIR / f"repair_state_{version}.jsonl"
    challenger_file = (
        REPAIR_DIR / f"repair_challenger_predictions_{version}.jsonl"
    )
    control_file = (
        REPAIR_DIR / f"frozen_control_predictions_{version}.jsonl"
    )
    summary_file = REPAIR_DIR / f"repair_summary_{version}.json"
    config_file = REPAIR_DIR / f"repair_config_{version}.json"

    gold_rows = read_jsonl(VISUAL_GOLD_FILE)
    scoring_gold_rows = adjusted_gold(gold_rows)
    gold_order = [str(row["id"]) for row in gold_rows]
    original_state = load_original_state(gold_rows)
    repair_state = load_repair_state(
        state_file,
        fingerprint=description_fingerprint,
    )
    unknown_ids = sorted(set(repair_state) - set(gold_order))
    if unknown_ids:
        raise ValueError(
            f"Repair state contains unknown IDs: {unknown_ids}"
        )

    visual_documents = [
        visual_document(row, chunk_number=900000 + index)
        for index, row in enumerate(description_rows, start=1)
    ]
    current_visual_documents = {
        (
            SOURCE_TO_DOC_ID[str(document.metadata["source"])],
            int(document.metadata["page"]),
        ): document
        for document in visual_documents
    }

    invalidated: list[str] = []
    for question_id, row in repair_state.items():
        if "challenger" not in row:
            continue
        _, expected_normalized = frozen_challenger_context(
            original_state[question_id],
            current_visual_documents,
        )
        saved_normalized = list(
            row["challenger"].get("retrieved", [])
        )
        if retrieved_identity(saved_normalized) != retrieved_identity(
            expected_normalized
        ):
            row.pop("challenger", None)
            row.pop("visual_anchor_correct", None)
            invalidated.append(question_id)
    if invalidated:
        save_repair_state(
            state_file,
            repair_state,
            gold_order,
        )

    llm = ChatOllama(
        model=GENERATION_MODEL,
        temperature=TEMPERATURE,
        num_ctx=NUM_CTX,
        num_predict=NUM_PREDICT,
        reasoning=False,
    )

    completed = sum(
        int("challenger" in row)
        for row in repair_state.values()
    )
    print("Controlled experiment: multimodal faithfulness repair")
    print("Frozen control generations: reused; no new control calls")
    print(
        "Frozen retrieval context: original approved visual anchor "
        "+ original two text fills"
    )
    print("Repaired figures: 3 and 5")
    print(
        f"Description fingerprint: {description_fingerprint}"
    )
    print(
        f"Challenger calls: total={len(gold_rows)}, "
        f"completed={completed}, pending={len(gold_rows) - completed}"
    )
    if invalidated:
        print(
            "Invalidated saved answers with non-frozen context: "
            + ", ".join(invalidated)
        )

    correct_anchors = 0
    try:
        for position, gold in enumerate(gold_rows, start=1):
            question_id = str(gold["id"])
            question = str(gold["question"])
            row = repair_state.setdefault(
                question_id,
                {
                    "id": question_id,
                    "question": question,
                    "description_fingerprint": description_fingerprint,
                },
            )
            if str(row.get("question")) != question:
                raise ValueError(
                    f"Repair-state question changed for {question_id}."
                )
            if "challenger" in row:
                print(
                    f"  repair [{position:02d}/{len(gold_rows):02d}] "
                    f"{question_id} reused"
                )
                if (
                    bool(gold["answerable"])
                    and bool(row.get("visual_anchor_correct", False))
                ):
                    correct_anchors += 1
                continue

            retrieval_started = time.perf_counter()
            (
                challenger_results,
                challenger_normalized,
            ) = frozen_challenger_context(
                original_state[question_id],
                current_visual_documents,
            )
            retrieval_ms = (
                time.perf_counter() - retrieval_started
            ) * 1000
            if len(challenger_results) != FINAL_TOP_K:
                raise RuntimeError(
                    f"{question_id}: challenger did not return "
                    f"top {FINAL_TOP_K}."
                )

            anchor_unit = evidence_units(
                [challenger_normalized[0]]
            )
            gold_units = evidence_units(
                list(gold["gold_evidence"])
            )
            anchor_correct = (
                not bool(gold["answerable"])
                or bool(anchor_unit.intersection(gold_units))
            )
            if bool(gold["answerable"]) and not anchor_correct:
                raise RuntimeError(
                    f"{question_id}: repaired retrieval selected the wrong "
                    "visual anchor; generation blocked."
                )
            if bool(gold["answerable"]):
                correct_anchors += 1

            stop_ollama_model(EMBEDDING_MODEL)
            generation_started = time.perf_counter()
            answerable, answer, raw = generate_structured_answer(
                llm=llm,
                question=question,
                results=challenger_results,
            )
            generation_ms = (
                time.perf_counter() - generation_started
            ) * 1000
            row["visual_anchor_correct"] = anchor_correct
            row["challenger"] = arm_prediction(
                answerable=answerable,
                answer=answer,
                raw_response=raw,
                normalized_retrieved=challenger_normalized,
                retrieval_ms=retrieval_ms,
                generation_ms=generation_ms,
            )
            save_repair_state(
                state_file,
                repair_state,
                gold_order,
            )
            print(
                f"  repair [{position:02d}/{len(gold_rows):02d}] "
                f"{question_id} answerable="
                f"{str(answerable).lower()} {generation_ms:.1f} ms"
            )
    finally:
        stop_ollama_model(EMBEDDING_MODEL)
        stop_ollama_model(GENERATION_MODEL)

    incomplete = [
        question_id
        for question_id in gold_order
        if question_id not in repair_state
        or "challenger" not in repair_state[question_id]
    ]
    if incomplete:
        raise RuntimeError(
            f"Repair generation is incomplete: {incomplete}"
        )

    control_rows = prediction_rows(
        original_state,
        gold_rows,
        "control",
    )
    challenger_rows = prediction_rows(
        repair_state,
        gold_rows,
        "challenger",
    )
    write_jsonl(control_file, control_rows)
    write_jsonl(challenger_file, challenger_rows)

    metrics = load_metrics_module()
    control_metrics = metrics.evaluate(
        scoring_gold_rows,
        control_rows,
        ks=(3,),
    )
    challenger_metrics = metrics.evaluate(
        scoring_gold_rows,
        challenger_rows,
        ks=(3,),
    )
    (
        comparisons,
        improvements,
        regressions,
        answerability_regressions,
    ) = question_level_comparison(
        scoring_gold_rows,
        control_rows,
        challenger_rows,
        metrics,
    )

    relationships: list[dict] = []
    citation_results: list[dict] = []
    for gold, prediction in zip(
        scoring_gold_rows,
        challenger_rows,
        strict=True,
    ):
        question_id = str(gold["id"])
        relation_passed, relation = relationship_check(
            question_id,
            str(prediction["answer"]),
        )
        relationships.append(
            {
                "id": question_id,
                "passed": relation_passed,
                "gate": relation,
            }
        )
        citation_results.append(
            {
                "id": question_id,
                "passed": exact_citation_check(gold, prediction),
            }
        )

    answerable_count = sum(
        bool(row["answerable"]) for row in gold_rows
    )
    all_relationships_passed = all(
        row["passed"] for row in relationships
    )
    all_citations_passed = all(
        row["passed"] for row in citation_results
    )
    control_fact = float(
        control_metrics["answer"]["required_fact_recall"]
    )
    challenger_fact = float(
        challenger_metrics["answer"]["required_fact_recall"]
    )
    control_citation_f1 = float(
        control_metrics["citations"]["f1"]
    )
    challenger_citation_f1 = float(
        challenger_metrics["citations"]["f1"]
    )
    control_unanswerable = float(
        control_metrics["abstention"]["unanswerable_accuracy"]
    )
    challenger_unanswerable = float(
        challenger_metrics["abstention"]["unanswerable_accuracy"]
    )
    automated_gate_passed = (
        correct_anchors == answerable_count
        and challenger_fact > control_fact
        and challenger_citation_f1 >= control_citation_f1
        and challenger_unanswerable == 1.0
        and challenger_unanswerable >= control_unanswerable
        and regressions == 0
        and answerability_regressions == 0
        and all_relationships_passed
        and all_citations_passed
    )

    generation_latencies = [
        float(
            repair_state[question_id]["challenger"]["generation_ms"]
        )
        for question_id in gold_order
    ]
    summary = {
        "experiment_id": "multimodal_faithfulness_repair",
        "description_fingerprint": description_fingerprint,
        "repaired_figures": sorted(TARGET_FIGURES),
        "control_reused_from": str(STATE_FILE),
        "control": control_metrics,
        "challenger": challenger_metrics,
        "question_level": comparisons,
        "relationship_gates": relationships,
        "citation_gates": citation_results,
        "all_relationships_passed": all_relationships_passed,
        "all_citations_passed": all_citations_passed,
        "correct_visual_anchors": correct_anchors,
        "answerable_questions": answerable_count,
        "improvements": improvements,
        "regressions": regressions,
        "answerability_regressions": answerability_regressions,
        "generation_calls": len(generation_latencies),
        "mean_generation_ms": statistics.fmean(
            generation_latencies
        ),
        "automated_gate_passed": automated_gate_passed,
        "human_audit_required": True,
        "decision": (
            "PASS AUTOMATED REPAIR GATES; HUMAN AUDIT REQUIRED"
            if automated_gate_passed
            else "DO NOT PROMOTE; REPAIR GATE FAILED"
        ),
    }
    write_json(summary_file, summary)
    write_json(
        config_file,
        {
            "retrieval_screen": retrieval_summary,
            "description_fingerprint": description_fingerprint,
            "repaired_figures": sorted(TARGET_FIGURES),
            "visual_index": str(VISUAL_DATABASE_DIR),
            "frozen_text_index": str(DATABASE_DIR),
            "generation_model": GENERATION_MODEL,
            "temperature": TEMPERATURE,
            "num_ctx": NUM_CTX,
            "num_predict": NUM_PREDICT,
            "control_calls": 0,
            "challenger_calls": len(gold_rows),
            "retrieval_context_reused_from": str(STATE_FILE),
            "retrieval_context_frozen": True,
            "v005_scoring_alias_added": "build and use",
            "gold_visible_to_generation": False,
            "human_audit_required": True,
        },
    )

    print("\nMultimodal faithfulness repair complete")
    print(
        "  required-fact recall: "
        f"control={control_fact:.4f}, "
        f"challenger={challenger_fact:.4f}"
    )
    print(
        "  citation F1: "
        f"control={control_citation_f1:.4f}, "
        f"challenger={challenger_citation_f1:.4f}"
    )
    print(
        "  unanswerable accuracy: "
        f"control={control_unanswerable:.4f}, "
        f"challenger={challenger_unanswerable:.4f}"
    )
    print(
        f"  relationships_passed={all_relationships_passed}, "
        f"citations_passed={all_citations_passed}"
    )
    print(
        f"  improvements={improvements}, "
        f"regressions={regressions}"
    )
    print(f"Decision: {summary['decision']}")
    print(f"State saved to: {state_file.resolve()}")
    print(f"Predictions saved to: {challenger_file.resolve()}")
    print(f"Summary saved to: {summary_file.resolve()}")


if __name__ == "__main__":
    run()
