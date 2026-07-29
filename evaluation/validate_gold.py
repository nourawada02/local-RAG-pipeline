from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


REQUIRED_FIELDS = {
    "id",
    "question",
    "type",
    "answerable",
    "reference_answer",
    "required_facts",
    "gold_evidence",
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"line {line_number}: invalid JSON: {exc}") from exc
    return rows


def validate(rows: list[dict[str, Any]]) -> list[str]:
    errors: list[str] = []
    ids = [row.get("id") for row in rows]
    duplicates = [item for item, count in Counter(ids).items() if count > 1]
    if duplicates:
        errors.append(f"duplicate IDs: {duplicates}")

    for index, row in enumerate(rows, start=1):
        row_id = row.get("id", f"line-{index}")
        missing = REQUIRED_FIELDS - set(row)
        if missing:
            errors.append(f"{row_id}: missing fields {sorted(missing)}")
            continue
        if not isinstance(row["answerable"], bool):
            errors.append(f"{row_id}: answerable must be boolean")
        if row["answerable"]:
            if not row["reference_answer"].strip():
                errors.append(f"{row_id}: answerable question needs a reference answer")
            if not row["required_facts"]:
                errors.append(f"{row_id}: answerable question needs required facts")
            if not row["gold_evidence"]:
                errors.append(f"{row_id}: answerable question needs gold evidence")
        else:
            if row["gold_evidence"]:
                errors.append(f"{row_id}: unanswerable question must not have gold evidence")
        for evidence in row["gold_evidence"]:
            if "doc_id" not in evidence or not evidence.get("pages"):
                errors.append(f"{row_id}: each evidence item needs doc_id and pages")
            elif any(not isinstance(page, int) or page < 1 for page in evidence["pages"]):
                errors.append(f"{row_id}: evidence pages must be positive integers")
    return errors


def main() -> None:
    path = (
        Path(sys.argv[1])
        if len(sys.argv) > 1
        else Path(__file__).with_name("gold_questions_physical_pages.jsonl")
    )
    rows = load_jsonl(path)
    errors = validate(rows)
    if errors:
        print("\n".join(f"ERROR: {error}" for error in errors))
        raise SystemExit(1)
    counts = Counter(row["type"] for row in rows)
    print(f"Validated {len(rows)} questions.")
    for question_type, count in sorted(counts.items()):
        print(f"  {question_type}: {count}")


if __name__ == "__main__":
    main()
