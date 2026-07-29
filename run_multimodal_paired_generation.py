from __future__ import annotations

import json
import re
import statistics
import sys
import time
from pathlib import Path
from typing import Iterable

from langchain_core.documents import Document
from langchain_ollama import ChatOllama

from build_index import EMBEDDING_MODEL
from rag_query import GENERATION_MODEL, stop_ollama_model
from run_decomposed_hybrid_evaluation import retrieve_with_decomposition
from run_hybrid_evaluation import BM25Index, load_frozen_chunks
from run_multimodal_visual_retrieval_sweep import (
    DESCRIPTION_FILE,
    DETAIL_FILE,
    EXPERIMENT_DIR,
    FINAL_TOP_K,
    SUMMARY_FILE,
    VISUAL_DATABASE_DIR,
    VISUAL_GOLD_FILE,
    challenger_selection,
    normalize_control,
    open_visual_vector_store,
    retrieve_visual_anchor,
    visual_document,
    write_json,
    write_jsonl,
)
from run_ocr_evaluation import (
    DATABASE_DIR,
    extract_citations,
    open_evaluation_vector_store,
    read_jsonl,
    validate_inputs,
)


# Controlled paired answer-generation experiment:
# - Requires the completed retrieval screen to have passed.
# - Reuses the frozen text index and the audited five-description visual index.
# - Retrieves once, then gives the control and challenger to the same model,
#   prompt, temperature, context window, and output budget.
# - Alternates arm order by question ID to reduce order/warm-up bias.
# - Never exposes reference answers, required facts, or gold evidence to Qwen.
# - Uses an explicit model-produced Boolean instead of prose substring matching.
STATE_FILE = EXPERIMENT_DIR / "multimodal_paired_generation_state.jsonl"
CONTROL_PREDICTIONS_FILE = (
    EXPERIMENT_DIR / "multimodal_control_predictions.jsonl"
)
CHALLENGER_PREDICTIONS_FILE = (
    EXPERIMENT_DIR / "multimodal_challenger_predictions.jsonl"
)
PAIRED_SUMMARY_FILE = (
    EXPERIMENT_DIR / "multimodal_paired_generation_summary.json"
)
PAIRED_CONFIG_FILE = (
    EXPERIMENT_DIR / "multimodal_paired_generation_config.json"
)

TEMPERATURE = 0.2
NUM_CTX = 4096
NUM_PREDICT = 256

ANSWERABLE_PATTERN = re.compile(
    r"(?mi)^\s*ANSWERABLE\s*:\s*(true|false)\s*$"
)
ANSWER_PATTERN = re.compile(
    r"(?ims)^\s*ANSWER\s*:\s*(.*?)\s*\Z"
)


def load_metrics_module():
    evaluation_module_dir = (
        Path("rag-evaluation-starter") / "evaluation"
    ).resolve()
    if str(evaluation_module_dir) not in sys.path:
        sys.path.insert(0, str(evaluation_module_dir))
    import metrics

    return metrics


def validate_retrieval_gate() -> dict:
    validate_inputs()
    required_paths = [
        DATABASE_DIR,
        VISUAL_DATABASE_DIR,
        DESCRIPTION_FILE,
        DETAIL_FILE,
        SUMMARY_FILE,
        VISUAL_GOLD_FILE,
    ]
    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Required retrieval-screen artifacts are missing:\n- "
            + "\n- ".join(missing)
        )

    summary = json.loads(SUMMARY_FILE.read_text(encoding="utf-8"))
    counts = summary.get("counts", {})
    answerable = int(counts.get("answerable", 0))
    if not bool(summary.get("promotion_gate_passed", False)):
        raise RuntimeError(
            "The multimodal retrieval screen did not pass. "
            "Paired answer generation is blocked."
        )
    if int(summary.get("correct_visual_anchors", -1)) != answerable:
        raise RuntimeError(
            "The retrieval screen did not produce a correct visual anchor "
            "for every answerable visual question."
        )
    if int(counts.get("original_questions_routed", -1)) != 0:
        raise RuntimeError(
            "The original benchmark was not mechanically frozen."
        )
    return summary


def load_state() -> dict[str, dict]:
    if not STATE_FILE.exists():
        return {}
    rows = read_jsonl(STATE_FILE)
    state: dict[str, dict] = {}
    for row in rows:
        question_id = str(row["id"])
        if question_id in state:
            raise ValueError(
                f"Duplicate state row for {question_id}: {STATE_FILE}"
            )
        state[question_id] = row
    return state


def save_state(
    state: dict[str, dict],
    gold_order: list[str],
) -> None:
    rows = [
        state[question_id]
        for question_id in gold_order
        if question_id in state
    ]
    write_jsonl(STATE_FILE, rows)


def create_context(
    results: list[tuple[Document, float]],
) -> str:
    parts: list[str] = []
    for rank, (document, _) in enumerate(results, start=1):
        source = str(document.metadata.get("source", "Unknown source"))
        page = document.metadata.get("page", "Unknown page")
        content_type = str(
            document.metadata.get("content_type", "text")
        )
        parts.append(
            f"[Source {rank}: {source}, page {page}, "
            f"type={content_type}]\n{document.page_content.strip()}"
        )
    return "\n\n---\n\n".join(parts)


def parse_structured_answer(raw: str) -> tuple[bool, str]:
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        first_newline = cleaned.find("\n")
        if first_newline != -1:
            cleaned = cleaned[first_newline + 1 :]
        if cleaned.rstrip().endswith("```"):
            cleaned = cleaned.rstrip()[:-3]
        cleaned = cleaned.strip()

    answerable_matches = ANSWERABLE_PATTERN.findall(cleaned)
    answer_match = ANSWER_PATTERN.search(cleaned)
    if len(answerable_matches) != 1:
        raise RuntimeError(
            "Generation did not return exactly one required ANSWERABLE "
            "Boolean. "
            f"structure. Raw response: {raw!r}"
        )

    answerable = answerable_matches[0].casefold() == "true"
    if not answerable:
        return (
            False,
            "I do not have enough information in the retrieved documents."
        )

    if answer_match is None:
        raise RuntimeError(
            "Generation marked the question answerable but did not return "
            f"the required ANSWER field. Raw response: {raw!r}"
        )
    answer = answer_match.group(1).strip()
    if not answer:
        raise RuntimeError("Generation returned an empty ANSWER field.")
    return True, answer


def generate_structured_answer(
    llm: ChatOllama,
    question: str,
    results: list[tuple[Document, float]],
) -> tuple[bool, str, str]:
    context = create_context(results)
    messages = [
        (
            "system",
            (
                "You answer questions using only the supplied document "
                "context. Do not use outside knowledge, copy assumptions "
                "from the question, or invent missing visual relationships. "
                "A question is answerable only when the context explicitly "
                "states or visually describes the requested facts. "
                "Return exactly two fields:\n"
                "ANSWERABLE: true or false\n"
                "ANSWER: one concise, complete answer\n"
                "Always write the ANSWER label, including when ANSWERABLE "
                "is false. When ANSWERABLE is false, the ANSWER must be "
                "exactly: I do not have enough information in the retrieved "
                "documents.\n"
                "When ANSWERABLE is true, cite supporting sources using "
                "[Source N]. Do not reveal a thinking process."
            ),
        ),
        (
            "human",
            (
                f"Context:\n{context}\n\n"
                f"Question: {question}\n\n"
                "Return only the required two-field response."
            ),
        ),
    ]
    response = llm.invoke(messages)
    raw = str(response.content).strip()
    answerable, answer = parse_structured_answer(raw)
    return answerable, answer, raw


def alternating_arm_order(question_id: str) -> tuple[str, str]:
    digits = re.findall(r"\d+", question_id)
    if not digits:
        raise ValueError(
            f"Question ID has no numeric component: {question_id}"
        )
    if int(digits[-1]) % 2:
        return ("control", "challenger")
    return ("challenger", "control")


def arm_prediction(
    *,
    answerable: bool,
    answer: str,
    raw_response: str,
    normalized_retrieved: list[dict],
    retrieval_ms: float,
    generation_ms: float,
) -> dict:
    return {
        "answerable": answerable,
        "answer": answer,
        "raw_response": raw_response,
        "abstained": not answerable,
        "retrieved": normalized_retrieved,
        "citations": extract_citations(
            answer,
            normalized_retrieved,
        ),
        "retrieval_ms": round(retrieval_ms, 3),
        "generation_ms": round(generation_ms, 3),
        "total_ms": round(retrieval_ms + generation_ms, 3),
    }


def prediction_rows(
    state: dict[str, dict],
    gold_rows: list[dict],
    arm: str,
) -> list[dict]:
    predictions: list[dict] = []
    for gold in gold_rows:
        question_id = str(gold["id"])
        arm_row = state[question_id][arm]
        predictions.append(
            {
                "id": question_id,
                **arm_row,
            }
        )
    return predictions


def evidence_units(items: Iterable[dict]) -> set[tuple[str, int]]:
    units: set[tuple[str, int]] = set()
    for item in items:
        pages = item.get("pages")
        if pages is None and "page" in item:
            pages = [item["page"]]
        for page in pages or []:
            units.add((str(item["doc_id"]), int(page)))
    return units


def question_level_comparison(
    gold_rows: list[dict],
    control_rows: list[dict],
    challenger_rows: list[dict],
    metrics,
) -> tuple[list[dict], int, int, int]:
    control_by_id = {str(row["id"]): row for row in control_rows}
    challenger_by_id = {
        str(row["id"]): row for row in challenger_rows
    }
    comparisons: list[dict] = []
    improvements = 0
    regressions = 0
    answerability_regressions = 0

    for gold in gold_rows:
        question_id = str(gold["id"])
        control = control_by_id[question_id]
        challenger = challenger_by_id[question_id]
        gold_answerable = bool(gold["answerable"])

        if gold_answerable:
            control_fact = metrics.required_fact_recall(
                str(control["answer"]),
                list(gold["required_facts"]),
            )
            challenger_fact = metrics.required_fact_recall(
                str(challenger["answer"]),
                list(gold["required_facts"]),
            )
            if challenger_fact > control_fact:
                result = "improved"
                improvements += 1
            elif challenger_fact < control_fact:
                result = "regressed"
                regressions += 1
            else:
                result = "same"
        else:
            control_fact = None
            challenger_fact = None
            control_correct = bool(control["abstained"])
            challenger_correct = bool(challenger["abstained"])
            if challenger_correct and not control_correct:
                result = "improved"
                improvements += 1
            elif control_correct and not challenger_correct:
                result = "regressed"
                regressions += 1
                answerability_regressions += 1
            else:
                result = "same"

        gold_units = evidence_units(list(gold["gold_evidence"]))
        control_citations = evidence_units(
            list(control.get("citations", []))
        )
        challenger_citations = evidence_units(
            list(challenger.get("citations", []))
        )
        comparisons.append(
            {
                "id": question_id,
                "answerable": gold_answerable,
                "result": result,
                "control_fact_recall": control_fact,
                "challenger_fact_recall": challenger_fact,
                "control_answerable": bool(control["answerable"]),
                "challenger_answerable": bool(
                    challenger["answerable"]
                ),
                "control_citation_correct": bool(
                    not gold_answerable
                    or (control_citations & gold_units)
                ),
                "challenger_citation_correct": bool(
                    not gold_answerable
                    or (challenger_citations & gold_units)
                ),
            }
        )

    return (
        comparisons,
        improvements,
        regressions,
        answerability_regressions,
    )


def save_config(retrieval_summary: dict) -> None:
    write_json(
        PAIRED_CONFIG_FILE,
        {
            "experiment_id": "selective_multimodal_paired_generation",
            "retrieval_screen": retrieval_summary,
            "frozen_text_index": str(DATABASE_DIR),
            "visual_index": str(VISUAL_DATABASE_DIR),
            "visual_descriptions": str(DESCRIPTION_FILE),
            "generation_model": GENERATION_MODEL,
            "temperature": TEMPERATURE,
            "num_ctx": NUM_CTX,
            "num_predict": NUM_PREDICT,
            "reasoning": False,
            "control": "frozen decomposed balanced-hybrid top 3",
            "challenger": (
                "one visual-description anchor plus two unique frozen "
                "control pages"
            ),
            "arm_order": "alternated by odd/even question ID",
            "answerability": "explicit model-produced Boolean",
            "gold_visible_to_generation": False,
        },
    )


def run() -> None:
    retrieval_summary = validate_retrieval_gate()
    gold_rows = read_jsonl(VISUAL_GOLD_FILE)
    gold_order = [str(row["id"]) for row in gold_rows]
    state = load_state()

    unknown_ids = sorted(set(state) - set(gold_order))
    if unknown_ids:
        raise ValueError(
            f"State contains IDs outside the visual benchmark: {unknown_ids}"
        )

    text_store = open_evaluation_vector_store()
    frozen_chunks = load_frozen_chunks(text_store)
    text_bm25 = BM25Index(frozen_chunks)
    visual_store = open_visual_vector_store()
    visual_documents = [
        visual_document(row, chunk_number=900000 + index)
        for index, row in enumerate(
            read_jsonl(DESCRIPTION_FILE),
            start=1,
        )
    ]
    visual_bm25 = BM25Index(visual_documents)
    llm = ChatOllama(
        model=GENERATION_MODEL,
        temperature=TEMPERATURE,
        num_ctx=NUM_CTX,
        num_predict=NUM_PREDICT,
        reasoning=False,
    )
    save_config(retrieval_summary)

    completed_arms = sum(
        int("control" in row) + int("challenger" in row)
        for row in state.values()
    )
    total_arms = len(gold_rows) * 2
    print("Controlled experiment: paired multimodal answer generation")
    print(f"Retrieval gate: PASS ({SUMMARY_FILE.resolve()})")
    print(f"Frozen OCR index: {DATABASE_DIR.resolve()}")
    print(f"Visual index: {VISUAL_DATABASE_DIR.resolve()}")
    print(
        "Control: frozen decomposed balanced-hybrid top 3 | "
        "Challenger: one visual anchor + two frozen control pages"
    )
    print(
        f"Generator: {GENERATION_MODEL}; temperature={TEMPERATURE}; "
        f"num_ctx={NUM_CTX}; num_predict={NUM_PREDICT}; reasoning=false"
    )
    print("Answerability: explicit Boolean; no prose marker matching")
    print("Gold answers/evidence: scoring only after both arms complete")
    print(
        f"Generation arms: total={total_arms}, "
        f"completed={completed_arms}, pending={total_arms - completed_arms}"
    )

    try:
        for position, gold in enumerate(gold_rows, start=1):
            question_id = str(gold["id"])
            question = str(gold["question"])
            row = state.setdefault(
                question_id,
                {
                    "id": question_id,
                    "question": question,
                    "arm_order": list(
                        alternating_arm_order(question_id)
                    ),
                },
            )
            if str(row.get("question")) != question:
                raise ValueError(
                    f"State question changed for {question_id}."
                )
            if "control" in row and "challenger" in row:
                print(
                    f"  pair [{position:02d}/{len(gold_rows):02d}] "
                    f"{question_id} reused"
                )
                continue

            stop_ollama_model(GENERATION_MODEL)
            retrieval_started = time.perf_counter()
            (
                control_results,
                decomposed_selected,
                balanced_selected,
                _subqueries,
                _query_runs,
            ) = retrieve_with_decomposition(
                vector_store=text_store,
                bm25_index=text_bm25,
                question=question,
            )
            control_normalized = normalize_control(
                control_results,
                decomposed_selected,
                balanced_selected,
            )
            visual_anchor, visual_diagnostics = retrieve_visual_anchor(
                visual_store=visual_store,
                visual_bm25=visual_bm25,
                question=question,
            )
            (
                challenger_results,
                challenger_normalized,
            ) = challenger_selection(
                control_results=control_results,
                control_normalized=control_normalized,
                visual_document=visual_anchor,
                visual_diagnostics=visual_diagnostics,
            )
            retrieval_ms = (
                time.perf_counter() - retrieval_started
            ) * 1000

            if len(control_results) != FINAL_TOP_K:
                raise RuntimeError(
                    f"{question_id}: control did not return top {FINAL_TOP_K}."
                )
            if len(challenger_results) != FINAL_TOP_K:
                raise RuntimeError(
                    f"{question_id}: challenger did not return "
                    f"top {FINAL_TOP_K}."
                )

            stop_ollama_model(EMBEDDING_MODEL)
            arm_inputs = {
                "control": (control_results, control_normalized),
                "challenger": (
                    challenger_results,
                    challenger_normalized,
                ),
            }
            for arm in alternating_arm_order(question_id):
                if arm in row:
                    continue
                results, normalized = arm_inputs[arm]
                generation_started = time.perf_counter()
                answerable, answer, raw = generate_structured_answer(
                    llm=llm,
                    question=question,
                    results=results,
                )
                generation_ms = (
                    time.perf_counter() - generation_started
                ) * 1000
                row[arm] = arm_prediction(
                    answerable=answerable,
                    answer=answer,
                    raw_response=raw,
                    normalized_retrieved=normalized,
                    retrieval_ms=retrieval_ms,
                    generation_ms=generation_ms,
                )
                save_state(state, gold_order)
                print(
                    f"  generate [{position:02d}/{len(gold_rows):02d}] "
                    f"{question_id} arm={arm} "
                    f"answerable={str(answerable).lower()} "
                    f"{generation_ms:.1f} ms"
                )
    finally:
        stop_ollama_model(EMBEDDING_MODEL)
        stop_ollama_model(GENERATION_MODEL)

    incomplete = [
        question_id
        for question_id in gold_order
        if question_id not in state
        or "control" not in state[question_id]
        or "challenger" not in state[question_id]
    ]
    if incomplete:
        raise RuntimeError(
            f"Paired generation is incomplete: {incomplete}"
        )

    control_rows = prediction_rows(state, gold_rows, "control")
    challenger_rows = prediction_rows(
        state,
        gold_rows,
        "challenger",
    )
    write_jsonl(CONTROL_PREDICTIONS_FILE, control_rows)
    write_jsonl(CHALLENGER_PREDICTIONS_FILE, challenger_rows)

    metrics = load_metrics_module()
    control_metrics = metrics.evaluate(
        gold_rows,
        control_rows,
        ks=(3,),
    )
    challenger_metrics = metrics.evaluate(
        gold_rows,
        challenger_rows,
        ks=(3,),
    )
    (
        comparisons,
        improvements,
        regressions,
        answerability_regressions,
    ) = question_level_comparison(
        gold_rows,
        control_rows,
        challenger_rows,
        metrics,
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
    promotion = (
        challenger_fact > control_fact
        and challenger_citation_f1 >= control_citation_f1
        and challenger_unanswerable == 1.0
        and challenger_unanswerable >= control_unanswerable
        and regressions == 0
        and answerability_regressions == 0
    )

    generation_latencies = [
        float(state[question_id][arm]["generation_ms"])
        for question_id in gold_order
        for arm in ("control", "challenger")
    ]
    summary = {
        "experiment_id": "selective_multimodal_paired_generation",
        "control": control_metrics,
        "challenger": challenger_metrics,
        "question_level": comparisons,
        "improvements": improvements,
        "regressions": regressions,
        "answerability_regressions": answerability_regressions,
        "generation_calls": len(generation_latencies),
        "mean_generation_ms": statistics.fmean(
            generation_latencies
        ),
        "promotion_gate_passed": promotion,
        "decision": (
            "PASS PAIRED GENERATION; eligible for selective integration"
            if promotion
            else "DO NOT PROMOTE; audit paired answers"
        ),
    }
    write_json(PAIRED_SUMMARY_FILE, summary)

    print("\nPaired multimodal generation complete")
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
        f"  improvements={improvements}, "
        f"regressions={regressions}"
    )
    print(f"Decision: {summary['decision']}")
    print(f"Summary saved to: {PAIRED_SUMMARY_FILE.resolve()}")


if __name__ == "__main__":
    run()
