from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


ARTICLES = {"a", "an", "the"}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
    return rows


def normalize_text(value: str, *, drop_articles: bool = True) -> str:
    value = unicodedata.normalize("NFKC", value).lower()
    value = re.sub(r"[^\w\s]", " ", value, flags=re.UNICODE)
    tokens = value.split()
    if drop_articles:
        tokens = [token for token in tokens if token not in ARTICLES]
    return " ".join(tokens)


def exact_match(prediction: str, reference: str) -> float:
    return float(normalize_text(prediction) == normalize_text(reference))


def token_f1(prediction: str, reference: str) -> float:
    prediction_tokens = normalize_text(prediction).split()
    reference_tokens = normalize_text(reference).split()
    if not prediction_tokens or not reference_tokens:
        return float(prediction_tokens == reference_tokens)
    overlap = Counter(prediction_tokens) & Counter(reference_tokens)
    common = sum(overlap.values())
    if common == 0:
        return 0.0
    precision = common / len(prediction_tokens)
    recall = common / len(reference_tokens)
    return 2 * precision * recall / (precision + recall)


def required_fact_recall(answer: str, required_facts: list[list[str]]) -> float:
    if not required_facts:
        return 1.0
    normalized_answer = f" {normalize_text(answer, drop_articles=False)} "
    hits = 0
    for aliases in required_facts:
        if any(
            f" {normalize_text(alias, drop_articles=False)} " in normalized_answer
            for alias in aliases
        ):
            hits += 1
    return hits / len(required_facts)


def evidence_units(items: Iterable[dict[str, Any]]) -> set[tuple[str, int]]:
    units: set[tuple[str, int]] = set()
    for item in items:
        doc_id = str(item["doc_id"])
        pages = item.get("pages")
        if pages is None and "page" in item:
            pages = [item["page"]]
        for page in pages or []:
            units.add((doc_id, int(page)))
    return units


def ranked_retrieval_metrics(
    retrieved: list[dict[str, Any]],
    gold_units: set[tuple[str, int]],
    k: int,
) -> dict[str, float]:
    top_results = retrieved[:k]
    seen_gold: set[tuple[str, int]] = set()
    first_relevant_rank: int | None = None

    for rank, result in enumerate(top_results, start=1):
        relevant_units = evidence_units([result]) & gold_units
        if relevant_units and first_relevant_rank is None:
            first_relevant_rank = rank
        new_units = relevant_units - seen_gold
        seen_gold.update(new_units)

    recall = len(seen_gold) / len(gold_units) if gold_units else 0.0

    return {
        f"hit@{k}": float(bool(seen_gold)),
        f"recall@{k}": recall,
        f"mrr@{k}": 0.0 if first_relevant_rank is None else 1 / first_relevant_rank,
    }


def precision_recall_f1(
    predicted: set[tuple[str, int]],
    gold: set[tuple[str, int]],
) -> tuple[float, float, float]:
    true_positive = len(predicted & gold)
    precision = true_positive / len(predicted) if predicted else 0.0
    recall = true_positive / len(gold) if gold else 0.0
    f1 = (
        0.0
        if precision + recall == 0
        else 2 * precision * recall / (precision + recall)
    )
    return precision, recall, f1


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def mean_or_none(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def evaluate(
    gold_rows: list[dict[str, Any]],
    prediction_rows: list[dict[str, Any]],
    ks: tuple[int, ...] = (3, 5),
) -> dict[str, Any]:
    predictions = {row["id"]: row for row in prediction_rows}
    missing = [row["id"] for row in gold_rows if row["id"] not in predictions]
    extra = sorted(set(predictions) - {row["id"] for row in gold_rows})
    if missing or extra:
        raise ValueError(f"Prediction IDs do not match gold IDs. missing={missing}, extra={extra}")

    retrieval_values: dict[str, list[float]] = {
        name: []
        for k in ks
        for name in (f"hit@{k}", f"recall@{k}", f"mrr@{k}")
    }
    exact_matches: list[float] = []
    token_f1s: list[float] = []
    fact_recalls: list[float] = []
    citation_precisions: list[float] = []
    citation_recalls: list[float] = []
    citation_f1s: list[float] = []
    abstention_correct: list[float] = []
    unanswerable_correct: list[float] = []
    retrieval_latencies: list[float] = []
    generation_latencies: list[float] = []
    total_latencies: list[float] = []
    human_correctness: list[float] = []
    human_faithfulness: list[float] = []
    by_type: dict[str, dict[str, list[float]]] = {}

    for gold in gold_rows:
        prediction = predictions[gold["id"]]
        question_type = gold["type"]
        by_type.setdefault(
            question_type,
            {"fact_recall": [], "abstention_accuracy": [], "hit@5": []},
        )

        answerable = bool(gold["answerable"])
        abstained = bool(prediction.get("abstained", False))
        abstention_score = float(abstained != answerable)
        abstention_correct.append(abstention_score)
        by_type[question_type]["abstention_accuracy"].append(abstention_score)

        if "retrieval_ms" in prediction:
            retrieval_latencies.append(float(prediction["retrieval_ms"]))
        if "generation_ms" in prediction:
            generation_latencies.append(float(prediction["generation_ms"]))
        if "total_ms" in prediction:
            total_latencies.append(float(prediction["total_ms"]))
        elif "retrieval_ms" in prediction and "generation_ms" in prediction:
            total_latencies.append(
                float(prediction["retrieval_ms"]) + float(prediction["generation_ms"])
            )

        if not answerable:
            unanswerable_correct.append(float(abstained))
            continue

        answer = str(prediction.get("answer", ""))
        exact_matches.append(exact_match(answer, gold["reference_answer"]))
        token_f1s.append(token_f1(answer, gold["reference_answer"]))
        fact_score = required_fact_recall(answer, gold["required_facts"])
        fact_recalls.append(fact_score)
        by_type[question_type]["fact_recall"].append(fact_score)

        gold_units = evidence_units(gold["gold_evidence"])
        retrieved = prediction.get("retrieved", [])
        for k in ks:
            result = ranked_retrieval_metrics(retrieved, gold_units, k)
            for name, value in result.items():
                retrieval_values[name].append(value)
            if k == 5:
                by_type[question_type]["hit@5"].append(result["hit@5"])

        cited_units = evidence_units(prediction.get("citations", []))
        precision, recall, f1 = precision_recall_f1(cited_units, gold_units)
        citation_precisions.append(precision)
        citation_recalls.append(recall)
        citation_f1s.append(f1)

        if "human_correctness" in prediction:
            human_correctness.append(float(prediction["human_correctness"]))
        if "human_faithfulness" in prediction:
            human_faithfulness.append(float(prediction["human_faithfulness"]))

    def latency_summary(values: list[float]) -> dict[str, float | None]:
        return {
            "median_ms": percentile(values, 0.5),
            "p95_ms": percentile(values, 0.95),
        }

    result: dict[str, Any] = {
        "counts": {
            "total": len(gold_rows),
            "answerable": sum(bool(row["answerable"]) for row in gold_rows),
            "unanswerable": sum(not bool(row["answerable"]) for row in gold_rows),
        },
        "retrieval": {
            name: mean_or_none(values) for name, values in retrieval_values.items()
        },
        "answer": {
            "exact_match": mean_or_none(exact_matches),
            "token_f1": mean_or_none(token_f1s),
            "required_fact_recall": mean_or_none(fact_recalls),
        },
        "citations": {
            "precision": mean_or_none(citation_precisions),
            "recall": mean_or_none(citation_recalls),
            "f1": mean_or_none(citation_f1s),
        },
        "abstention": {
            "accuracy_all_questions": mean_or_none(abstention_correct),
            "unanswerable_accuracy": mean_or_none(unanswerable_correct),
        },
        "latency": {
            "retrieval": latency_summary(retrieval_latencies),
            "generation": latency_summary(generation_latencies),
            "total": latency_summary(total_latencies),
        },
        "human_optional": {
            "correctness_mean_0_to_2": mean_or_none(human_correctness),
            "faithfulness_mean_0_to_2": mean_or_none(human_faithfulness),
        },
        "by_question_type": {
            question_type: {
                metric: mean_or_none(values)
                for metric, values in type_values.items()
                if values
            }
            for question_type, type_values in sorted(by_type.items())
        },
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate normalized RAG predictions.")
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    result = evaluate(read_jsonl(args.gold), read_jsonl(args.predictions))
    rendered = json.dumps(result, indent=2, ensure_ascii=False)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
