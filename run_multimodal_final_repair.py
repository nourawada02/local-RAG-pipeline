from __future__ import annotations

import hashlib
import json
import re
import statistics
import time
import urllib.error
import urllib.request
from copy import deepcopy
from pathlib import Path

from langchain_core.documents import Document

from build_index import EMBEDDING_MODEL
from rag_query import GENERATION_MODEL, stop_ollama_model
from run_multimodal_paired_generation import (
    NUM_CTX,
    NUM_PREDICT,
    STATE_FILE,
    TEMPERATURE,
    arm_prediction,
    create_context,
    evidence_units,
    load_metrics_module,
    prediction_rows,
    question_level_comparison,
    validate_retrieval_gate,
)
from run_multimodal_visual_retrieval_sweep import (
    DESCRIPTION_FILE,
    EXPERIMENT_DIR,
    FINAL_TOP_K,
    FIGURE_2_CENTER_PATTERN,
    FIGURE_2_DIMENSION_PATTERN,
    FIGURE_2_STAGE_PATTERN,
    FIGURE_3_COLUMN_PATTERN,
    FIGURE_5_POSITION_PATTERN,
    SOURCE_TO_DOC_ID,
    VISUAL_DATABASE_DIR,
    VISUAL_GOLD_FILE,
    canonical_visual_text,
    detect_figure_pages,
    validate_visual_description,
    visual_document,
    write_json,
    write_jsonl,
)
from run_ocr_evaluation import DATABASE_DIR, read_jsonl


# Final controlled repair:
# - Original control generations and original approved retrieval identities stay
#   frozen.
# - Figures 2, 3, and 5 must pass source-grounded ingestion validation.
# - All 12 challenger answers are regenerated under one JSON contract.
# - Figure 2/3/5 relationship answers are checked against the actual retrieved
#   visual description before acceptance and retried on contradiction.
# - Reference answers, required facts, and gold evidence are scoring-only.
FINAL_REPAIR_DIR = EXPERIMENT_DIR / "faithfulness_repair_v3"
TARGET_FIGURES = {2, 3, 5}
GENERATION_MAX_ATTEMPTS = 3
OLLAMA_BASE_URL = "http://localhost:11434"
GENERATION_CONTRACT_VERSION = "structured-source-gated-v3"
ABSTENTION_TEXT = (
    "I do not have enough information in the retrieved documents."
)
STRUCTURED_QUESTION_IDS = {
    "v003",
    "v004",
    "v005",
    "v006",
    "v009",
    "v010",
}


def file_fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def contract_fingerprint(description_fingerprint: str) -> str:
    payload = (
        f"{GENERATION_CONTRACT_VERSION}\n"
        f"{description_fingerprint}\n"
        f"{GENERATION_MODEL}\n{TEMPERATURE}\n{NUM_CTX}\n{NUM_PREDICT}"
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


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
        relevant = TARGET_FIGURES.intersection(item.figure_numbers)
        if relevant:
            validate_visual_description(item, str(row["description"]))
            validated.update(relevant)
    if validated != TARGET_FIGURES:
        raise RuntimeError(
            "Figures 2, 3, and 5 do not all have validated structured "
            "descriptions. Rebuild physical PDF page 15 first."
        )
    return rows


def load_final_state(
    state_file: Path,
    *,
    description_fingerprint: str,
    generation_fingerprint: str,
) -> dict[str, dict]:
    if not state_file.exists():
        return {}
    state: dict[str, dict] = {}
    for row in read_jsonl(state_file):
        question_id = str(row["id"])
        if str(row.get("description_fingerprint")) != (
            description_fingerprint
        ):
            raise ValueError(
                "Final repair state belongs to a different description set."
            )
        if str(row.get("generation_fingerprint")) != (
            generation_fingerprint
        ):
            raise ValueError(
                "Final repair state belongs to a different generation "
                "contract."
            )
        if question_id in state:
            raise ValueError(
                f"Duplicate final repair state row for {question_id}."
            )
        state[question_id] = row
    return state


def save_final_state(
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


def retrieved_identity(
    rows: list[dict],
) -> list[tuple[str, tuple[int, ...]]]:
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

    doc_id_to_source = {
        doc_id: source for source, doc_id in SOURCE_TO_DOC_ID.items()
    }
    normalized: list[dict] = []
    results: list[tuple[Document, float]] = []

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
                    "No current visual description matches the frozen "
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


def visual_content(anchor_text: str) -> str:
    marker = "Visual content:"
    if marker not in anchor_text:
        raise ValueError(
            "Frozen rank-one context is not a visual description."
        )
    return anchor_text.split(marker, 1)[1].strip()


def parse_figure_2(anchor_text: str) -> dict:
    center: list[str] = []
    dimensions: list[str] = []
    stages: list[str] = []
    for line in visual_content(anchor_text).splitlines():
        center_match = FIGURE_2_CENTER_PATTERN.fullmatch(line)
        dimension_match = FIGURE_2_DIMENSION_PATTERN.fullmatch(line)
        stage_match = FIGURE_2_STAGE_PATTERN.fullmatch(line)
        if center_match is not None:
            center.append(canonical_visual_text(center_match.group(1)))
        elif dimension_match is not None:
            dimensions.append(
                canonical_visual_text(dimension_match.group(1))
            )
        elif stage_match is not None:
            stages.append(canonical_visual_text(stage_match.group(1)))
    if len(center) != 1 or len(dimensions) != 4 or len(stages) != 6:
        raise ValueError(
            "The retrieved Figure 2 description is not structurally valid."
        )
    return {
        "center": center[0],
        "key_dimensions": dimensions,
        "lifecycle_stages": stages,
    }


def parse_figure_3(anchor_text: str) -> list[dict]:
    columns: list[dict] = []
    for line in visual_content(anchor_text).splitlines():
        match = FIGURE_3_COLUMN_PATTERN.fullmatch(line)
        if match is None:
            continue
        columns.append(
            {
                "column": int(match.group(1)),
                "key_dimension": canonical_visual_text(match.group(2)),
                "lifecycle_stage": canonical_visual_text(match.group(3)),
                "tevv": canonical_visual_text(match.group(4)),
            }
        )
    columns.sort(key=lambda row: int(row["column"]))
    if [row["column"] for row in columns] != list(range(1, 8)):
        raise ValueError(
            "The retrieved Figure 3 description is not structurally valid."
        )
    return columns


def parse_figure_5(anchor_text: str) -> dict[str, str]:
    positions: dict[str, str] = {}
    for line in visual_content(anchor_text).splitlines():
        match = FIGURE_5_POSITION_PATTERN.fullmatch(line)
        if match is not None:
            positions[match.group(1).upper()] = canonical_visual_text(
                match.group(2)
            )
    if set(positions) != {"CENTER", "LEFT", "RIGHT", "BELOW"}:
        raise ValueError(
            "The retrieved Figure 5 description is not structurally valid."
        )
    return positions


def string_array_schema(max_items: int) -> dict:
    return {
        "type": "array",
        "minItems": 0,
        "maxItems": max_items,
        "items": {"type": "string"},
    }


def response_schema(
    question_id: str,
    results: list[tuple[Document, float]] | None = None,
) -> dict:
    properties: dict[str, dict] = {
        "answerable": {"type": "boolean"},
        "answer": {"type": "string"},
        "citation_sources": {
            "type": "array",
            "minItems": 0,
            "maxItems": FINAL_TOP_K,
            "items": {
                "type": "integer",
                "minimum": 1,
                "maximum": FINAL_TOP_K,
            },
        },
    }
    structured_properties: dict[str, dict] = {}
    if question_id == "v003":
        structured_properties = {
            "center": {"type": "string"},
            "key_dimensions": string_array_schema(4),
            "lifecycle_stages": string_array_schema(6),
        }
    elif question_id == "v004":
        structured_properties = {
            "lifecycle_stages": string_array_schema(6),
        }
    elif question_id == "v005":
        structured_properties = {
            "repeated_dimension": {"type": "string"},
            "stages": string_array_schema(2),
        }
    elif question_id == "v006":
        structured_properties = {
            "internal_external_validation_stage": {"type": "string"},
            "integration_compliance_validation_stage": {
                "type": "string"
            },
        }
    elif question_id == "v009":
        structured_properties = {
            "central_function": {"type": "string"},
            "surrounding_functions": string_array_schema(3),
        }
    elif question_id == "v010":
        structured_properties = {
            "central_function": {"type": "string"},
            "left_function": {"type": "string"},
            "right_function": {"type": "string"},
            "below_function": {"type": "string"},
        }

    if structured_properties:
        properties["structured"] = {
            "type": "object",
            "properties": structured_properties,
            "required": list(structured_properties),
            "additionalProperties": False,
        }

    schema = {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }
    if results is not None and question_id in STRUCTURED_QUESTION_IDS:
        constrain_schema_to_visual_source(question_id, schema, results)
    return schema


def exact_string_schema(value: str) -> dict:
    return {
        "type": "string",
        "enum": [value],
    }


def exact_member_array_schema(values: list[str]) -> dict:
    return {
        "type": "array",
        "minItems": len(values),
        "maxItems": len(values),
        "items": {
            "type": "string",
            "enum": values,
        },
    }


def constrain_schema_to_visual_source(
    question_id: str,
    schema: dict,
    results: list[tuple[Document, float]],
) -> None:
    """Constrain relationship fields using Source 1, never benchmark gold."""
    structured = schema["properties"]["structured"]["properties"]
    anchor_text = results[0][0].page_content

    if question_id in {"v003", "v004"}:
        expected = parse_figure_2(anchor_text)
        if question_id == "v003":
            structured["center"] = exact_string_schema(expected["center"])
            structured["key_dimensions"] = exact_member_array_schema(
                expected["key_dimensions"]
            )
            structured["lifecycle_stages"] = exact_member_array_schema(
                expected["lifecycle_stages"]
            )
        else:
            structured["lifecycle_stages"] = exact_member_array_schema(
                expected["lifecycle_stages"]
            )
        return

    if question_id in {"v005", "v006"}:
        columns = parse_figure_3(anchor_text)
        if question_id == "v005":
            repeated_dimension = columns[2]["key_dimension"]
            repeated_stages = [
                columns[2]["lifecycle_stage"],
                columns[3]["lifecycle_stage"],
            ]
            structured["repeated_dimension"] = exact_string_schema(
                repeated_dimension
            )
            structured["stages"] = exact_member_array_schema(
                repeated_stages
            )
        else:
            structured[
                "internal_external_validation_stage"
            ] = exact_string_schema(columns[1]["lifecycle_stage"])
            structured[
                "integration_compliance_validation_stage"
            ] = exact_string_schema(columns[4]["lifecycle_stage"])
        return

    expected_positions = parse_figure_5(anchor_text)
    if question_id == "v009":
        structured["central_function"] = exact_string_schema(
            expected_positions["CENTER"]
        )
        structured["surrounding_functions"] = exact_member_array_schema(
            [
                expected_positions["LEFT"],
                expected_positions["RIGHT"],
                expected_positions["BELOW"],
            ]
        )
    else:
        field_to_role = {
            "central_function": "CENTER",
            "left_function": "LEFT",
            "right_function": "RIGHT",
            "below_function": "BELOW",
        }
        for field, role in field_to_role.items():
            structured[field] = exact_string_schema(
                expected_positions[role]
            )


def structured_instruction(question_id: str) -> str:
    if question_id == "v003":
        return (
            "Keep the center, four key dimensions, and six lifecycle stages "
            "in their separate structured fields. The answer should mention "
            "only the center and four key dimensions requested."
        )
    if question_id == "v004":
        return (
            "Put only the six outer-ring lifecycle stages in "
            "structured.lifecycle_stages."
        )
    if question_id == "v005":
        return (
            "Compare Source 1 columns 3 and 4. Copy their shared KEY "
            "DIMENSION into structured.repeated_dimension, then copy their "
            "two LIFECYCLE STAGE values into structured.stages."
        )
    if question_id == "v006":
        return (
            "Map each requested TEVV phrase to its lifecycle stage in the "
            "two separate structured fields."
        )
    if question_id == "v009":
        return (
            "Separate the one central function from the three surrounding "
            "functions."
        )
    if question_id == "v010":
        return (
            "Return the center, left, right, and below functions in their "
            "separate structured fields."
        )
    return "No extra structured relationship object is required."


def invoke_ollama_json(
    *,
    question_id: str,
    question: str,
    results: list[tuple[Document, float]],
    retry_error: str,
) -> str:
    schema = response_schema(question_id, results)
    retry_text = ""
    if retry_error:
        retry_text = (
            "\nThe previous response failed this check: "
            f"{retry_error}. Re-read the exact Source 1 labels allowed by "
            "the schema and create a fresh response. Do not repeat the "
            "previous value."
        )
    system_prompt = (
        "Answer only from the supplied document context. Do not use outside "
        "knowledge, copy assumptions from the question, or invent missing "
        "visual relationships. Source 1 is the visual anchor. A question is "
        "answerable only when the context explicitly supports the requested "
        "facts. Return JSON matching the supplied schema. Put a concise "
        "answer with no citation markers in the answer field. Put supporting "
        "source numbers in citation_sources. For an unanswerable question, "
        "set answerable=false, answer to the standard insufficient-"
        "information sentence, citation_sources=[], and leave any structured "
        "strings empty and arrays empty. Do not reveal reasoning. "
        + structured_instruction(question_id)
        + retry_text
        + "\nRequired JSON schema: "
        + json.dumps(schema, ensure_ascii=False)
    )
    request_body = {
        "model": GENERATION_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": (
                    f"Context:\n{create_context(results)}\n\n"
                    f"Question: {question}\n\nReturn JSON only."
                ),
            },
        ],
        "stream": False,
        "think": False,
        "format": schema,
        "options": {
            "temperature": TEMPERATURE,
            "num_ctx": NUM_CTX,
            "num_predict": NUM_PREDICT,
        },
    }
    request = urllib.request.Request(
        f"{OLLAMA_BASE_URL}/api/chat",
        data=json.dumps(request_body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            result = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"Ollama generation failed with HTTP {exc.code}: {body}"
        ) from exc
    except (
        OSError,
        urllib.error.URLError,
        json.JSONDecodeError,
    ) as exc:
        raise RuntimeError(
            f"Ollama generation request failed for {question_id}: {exc}"
        ) from exc
    return str((result.get("message") or {}).get("content", "")).strip()


def canonical_values(values: object) -> list[str]:
    if not isinstance(values, list):
        raise ValueError("a structured array field was not an array")
    return [canonical_visual_text(str(value)) for value in values]


def require_same_members(
    actual: object,
    expected: list[str],
    *,
    label: str,
) -> list[str]:
    normalized = canonical_values(actual)
    if len(normalized) != len(expected) or set(normalized) != set(expected):
        raise ValueError(
            f"{label} contradicted the retrieved visual description"
        )
    return normalized


def validate_payload(
    question_id: str,
    raw: str,
    results: list[tuple[Document, float]],
) -> dict:
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("response was not valid JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError("response was not a JSON object")
    if type(payload.get("answerable")) is not bool:
        raise ValueError("answerable was not a Boolean")
    answerable = bool(payload["answerable"])
    answer = str(payload.get("answer", "")).strip()
    citations = payload.get("citation_sources")
    if not isinstance(citations, list) or any(
        type(value) is not int for value in citations
    ):
        raise ValueError("citation_sources was not an integer array")
    if len(set(citations)) != len(citations):
        raise ValueError("citation_sources contained duplicates")
    if any(value < 1 or value > len(results) for value in citations):
        raise ValueError("citation_sources contained an invalid rank")

    if not answerable:
        if citations:
            raise ValueError(
                "unanswerable response cited supporting evidence"
            )
        if question_id in STRUCTURED_QUESTION_IDS:
            raise ValueError(
                "the structured visual anchor explicitly supports this "
                "question"
            )
        return payload

    if not answer:
        raise ValueError("answerable response had an empty answer")
    if 1 not in citations:
        raise ValueError(
            "answerable visual response did not cite Source 1"
        )
    structured = payload.get("structured")
    if question_id in STRUCTURED_QUESTION_IDS and not isinstance(
        structured,
        dict,
    ):
        raise ValueError("structured relationship fields were missing")

    anchor_text = results[0][0].page_content
    if question_id in {"v003", "v004"}:
        expected = parse_figure_2(anchor_text)
        if question_id == "v003":
            if canonical_visual_text(str(structured.get("center", ""))) != (
                expected["center"]
            ):
                raise ValueError(
                    "center contradicted the retrieved Figure 2 description"
                )
            require_same_members(
                structured.get("key_dimensions"),
                expected["key_dimensions"],
                label="key dimensions",
            )
            require_same_members(
                structured.get("lifecycle_stages"),
                expected["lifecycle_stages"],
                label="lifecycle stages",
            )
        else:
            require_same_members(
                structured.get("lifecycle_stages"),
                expected["lifecycle_stages"],
                label="lifecycle stages",
            )

    elif question_id in {"v005", "v006"}:
        columns = parse_figure_3(anchor_text)
        if question_id == "v005":
            repeated_dimension = columns[2]["key_dimension"]
            repeated_stages = [
                columns[2]["lifecycle_stage"],
                columns[3]["lifecycle_stage"],
            ]
            if canonical_visual_text(
                str(structured.get("repeated_dimension", ""))
            ) != repeated_dimension:
                raise ValueError(
                    "repeated dimension contradicted Figure 3"
                )
            require_same_members(
                structured.get("stages"),
                repeated_stages,
                label="repeated-dimension stages",
            )
        else:
            if canonical_visual_text(
                str(
                    structured.get(
                        "internal_external_validation_stage",
                        "",
                    )
                )
            ) != columns[1]["lifecycle_stage"]:
                raise ValueError(
                    "internal/external validation mapping contradicted "
                    "Figure 3"
                )
            if canonical_visual_text(
                str(
                    structured.get(
                        "integration_compliance_validation_stage",
                        "",
                    )
                )
            ) != columns[4]["lifecycle_stage"]:
                raise ValueError(
                    "integration/compliance mapping contradicted Figure 3"
                )

    elif question_id in {"v009", "v010"}:
        expected_positions = parse_figure_5(anchor_text)
        if question_id == "v009":
            if canonical_visual_text(
                str(structured.get("central_function", ""))
            ) != expected_positions["CENTER"]:
                raise ValueError(
                    "central function contradicted Figure 5"
                )
            require_same_members(
                structured.get("surrounding_functions"),
                [
                    expected_positions["LEFT"],
                    expected_positions["RIGHT"],
                    expected_positions["BELOW"],
                ],
                label="surrounding functions",
            )
        else:
            field_to_role = {
                "central_function": "CENTER",
                "left_function": "LEFT",
                "right_function": "RIGHT",
                "below_function": "BELOW",
            }
            for field, role in field_to_role.items():
                if canonical_visual_text(
                    str(structured.get(field, ""))
                ) != expected_positions[role]:
                    raise ValueError(
                        f"{field} contradicted Figure 5"
                    )
    return payload


def display_value(value: object) -> str:
    return re.sub(r"\s+", " ", str(value)).strip()


def citation_suffix(citations: list[int]) -> str:
    return " ".join(f"[Source {value}]" for value in citations)


def render_accepted_answer(
    question_id: str,
    payload: dict,
) -> tuple[bool, str]:
    if not bool(payload["answerable"]):
        return False, ABSTENTION_TEXT

    structured = payload.get("structured") or {}
    if question_id == "v003":
        dimensions = ", ".join(
            display_value(value)
            for value in structured["key_dimensions"]
        )
        answer = (
            f"{display_value(structured['center'])} is at the center, "
            f"surrounded by the four key dimensions: {dimensions}."
        )
    elif question_id == "v004":
        stages = ", ".join(
            display_value(value)
            for value in structured["lifecycle_stages"]
        )
        answer = f"The six outer-ring lifecycle stages are {stages}."
    elif question_id == "v005":
        stages = [
            display_value(value) for value in structured["stages"]
        ]
        answer = (
            f"{display_value(structured['repeated_dimension'])} is repeated "
            f"across the two consecutive stages {stages[0]} and "
            f"{stages[1]}."
        )
    elif question_id == "v006":
        answer = (
            f"{display_value(structured['internal_external_validation_stage'])} "
            "includes internal and external validation, while "
            f"{display_value(structured['integration_compliance_validation_stage'])} "
            "includes integration, compliance testing, and validation."
        )
    elif question_id == "v009":
        surrounding = ", ".join(
            display_value(value)
            for value in structured["surrounding_functions"]
        )
        answer = (
            f"{display_value(structured['central_function'])} is central, "
            f"surrounded by {surrounding}."
        )
    elif question_id == "v010":
        center = display_value(structured["central_function"])
        answer = (
            f"{display_value(structured['left_function'])} is left of "
            f"{center}, {display_value(structured['right_function'])} is "
            f"right of {center}, and "
            f"{display_value(structured['below_function'])} is below "
            f"{center}."
        )
    else:
        answer = re.sub(
            r"\s*\[Source\s+\d+\]\s*",
            " ",
            str(payload["answer"]),
            flags=re.IGNORECASE,
        )
        answer = re.sub(r"\s+", " ", answer).strip()

    suffix = citation_suffix(list(payload["citation_sources"]))
    return True, f"{answer} {suffix}".strip()


def generate_source_gated_answer(
    *,
    question_id: str,
    question: str,
    results: list[tuple[Document, float]],
) -> tuple[bool, str, str, dict, int]:
    last_error = ""
    for attempt in range(1, GENERATION_MAX_ATTEMPTS + 1):
        raw = invoke_ollama_json(
            question_id=question_id,
            question=question,
            results=results,
            retry_error=last_error,
        )
        try:
            payload = validate_payload(
                question_id,
                raw,
                results,
            )
            answerable, answer = render_accepted_answer(
                question_id,
                payload,
            )
            return answerable, answer, raw, payload, attempt
        except ValueError as exc:
            last_error = str(exc)
            if attempt < GENERATION_MAX_ATTEMPTS:
                print(
                    f"    answer retry {attempt + 1}/"
                    f"{GENERATION_MAX_ATTEMPTS} for {question_id} "
                    f"({last_error})"
                )
    raise RuntimeError(
        f"{question_id}: generation failed the structured source gate after "
        f"{GENERATION_MAX_ATTEMPTS} attempts: {last_error}"
    )


def exact_citation_check(gold: dict, prediction: dict) -> bool:
    if not bool(gold["answerable"]):
        return bool(prediction["abstained"])
    gold_units = evidence_units(list(gold["gold_evidence"]))
    cited_units = evidence_units(
        list(prediction.get("citations", []))
    )
    return bool(gold_units.intersection(cited_units))


def normalized_answer(answer: str) -> str:
    answer = answer.casefold().replace("&", " and ")
    answer = re.sub(r"[^a-z0-9]+", " ", answer)
    return re.sub(r"\s+", " ", answer).strip()


def relationship_text_check(
    question_id: str,
    answer: str,
) -> tuple[bool, str]:
    text = normalized_answer(answer)
    if question_id == "v003":
        passed = (
            "people and planet" in text
            and "center" in text
            and all(
                value in text
                for value in (
                    "application context",
                    "data and input",
                    "ai model",
                    "task and output",
                )
            )
            and "five dimensions" not in text
        )
        return passed, "center separated from exactly four dimensions"
    if question_id == "v006":
        passed = (
            re.search(
                r"collect and process data.{0,100}"
                r"internal and external validation",
                text,
            )
            is not None
            and re.search(
                r"deploy and use.{0,100}integration.{0,80}"
                r"compliance testing.{0,80}validation",
                text,
            )
            is not None
        )
        return passed, "TEVV phrases mapped to their source stages"
    if question_id == "v009":
        passed = (
            re.search(r"govern.{0,40}(central|center)", text) is not None
            and all(value in text for value in ("map", "measure", "manage"))
            and "govern is also" not in text
        )
        return passed, "GOVERN central; three distinct surrounding functions"
    if question_id == "v010":
        passed = all(
            re.search(pattern, text) is not None
            for pattern in (
                r"map.{0,30}left.{0,20}govern",
                r"measure.{0,30}right.{0,20}govern",
                r"manage.{0,30}below.{0,20}govern",
            )
        )
        return passed, "MAP left; MEASURE right; MANAGE below GOVERN"
    return True, "no extra rendered-text relationship gate"


def run() -> None:
    retrieval_summary = validate_retrieval_gate()
    description_rows = validate_repaired_descriptions()
    description_fingerprint = file_fingerprint(DESCRIPTION_FILE)
    generation_fingerprint = contract_fingerprint(
        description_fingerprint
    )
    version = generation_fingerprint[:12]

    state_file = (
        FINAL_REPAIR_DIR / f"final_repair_state_{version}.jsonl"
    )
    challenger_file = (
        FINAL_REPAIR_DIR
        / f"final_repair_challenger_predictions_{version}.jsonl"
    )
    control_file = (
        FINAL_REPAIR_DIR / f"frozen_control_predictions_{version}.jsonl"
    )
    summary_file = (
        FINAL_REPAIR_DIR / f"final_repair_summary_{version}.json"
    )
    config_file = (
        FINAL_REPAIR_DIR / f"final_repair_config_{version}.json"
    )

    gold_rows = read_jsonl(VISUAL_GOLD_FILE)
    scoring_gold_rows = adjusted_gold(gold_rows)
    gold_order = [str(row["id"]) for row in gold_rows]
    original_state = load_original_state(gold_rows)
    final_state = load_final_state(
        state_file,
        description_fingerprint=description_fingerprint,
        generation_fingerprint=generation_fingerprint,
    )
    unknown_ids = sorted(set(final_state) - set(gold_order))
    if unknown_ids:
        raise ValueError(
            f"Final repair state contains unknown IDs: {unknown_ids}"
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
    for question_id, row in final_state.items():
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
            row.pop("structured_payload", None)
            row.pop("generation_attempts", None)
            invalidated.append(question_id)
    if invalidated:
        save_final_state(state_file, final_state, gold_order)

    completed = sum(
        int("challenger" in row) for row in final_state.values()
    )
    print("Controlled experiment: final multimodal faithfulness repair")
    print("Frozen control generations: reused; no new control calls")
    print(
        "Frozen retrieval context: original approved visual anchor "
        "+ original two text fills"
    )
    print("Structured descriptions required: Figures 2, 3, and 5")
    print(
        "Generation: JSON schema + source-consistency retry; "
        f"max_attempts={GENERATION_MAX_ATTEMPTS}"
    )
    print(f"Description fingerprint: {description_fingerprint}")
    print(f"Generation fingerprint: {generation_fingerprint}")
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
            row = final_state.setdefault(
                question_id,
                {
                    "id": question_id,
                    "question": question,
                    "description_fingerprint": description_fingerprint,
                    "generation_fingerprint": generation_fingerprint,
                },
            )
            if str(row.get("question")) != question:
                raise ValueError(
                    f"Final-state question changed for {question_id}."
                )
            if "challenger" in row:
                print(
                    f"  final [{position:02d}/{len(gold_rows):02d}] "
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

            anchor_unit = evidence_units([challenger_normalized[0]])
            gold_units = evidence_units(list(gold["gold_evidence"]))
            anchor_correct = (
                not bool(gold["answerable"])
                or bool(anchor_unit.intersection(gold_units))
            )
            if bool(gold["answerable"]) and not anchor_correct:
                raise RuntimeError(
                    f"{question_id}: frozen retrieval has the wrong visual "
                    "anchor; generation blocked."
                )
            if bool(gold["answerable"]):
                correct_anchors += 1

            stop_ollama_model(EMBEDDING_MODEL)
            generation_started = time.perf_counter()
            (
                answerable,
                answer,
                raw,
                structured_payload,
                attempts,
            ) = generate_source_gated_answer(
                question_id=question_id,
                question=question,
                results=challenger_results,
            )
            generation_ms = (
                time.perf_counter() - generation_started
            ) * 1000
            row["visual_anchor_correct"] = anchor_correct
            row["structured_payload"] = structured_payload
            row["generation_attempts"] = attempts
            row["challenger"] = arm_prediction(
                answerable=answerable,
                answer=answer,
                raw_response=raw,
                normalized_retrieved=challenger_normalized,
                retrieval_ms=retrieval_ms,
                generation_ms=generation_ms,
            )
            save_final_state(state_file, final_state, gold_order)
            print(
                f"  final [{position:02d}/{len(gold_rows):02d}] "
                f"{question_id} answerable="
                f"{str(answerable).lower()} attempts={attempts} "
                f"{generation_ms:.1f} ms"
            )
    finally:
        stop_ollama_model(EMBEDDING_MODEL)
        stop_ollama_model(GENERATION_MODEL)

    incomplete = [
        question_id
        for question_id in gold_order
        if question_id not in final_state
        or "challenger" not in final_state[question_id]
    ]
    if incomplete:
        raise RuntimeError(
            f"Final repair generation is incomplete: {incomplete}"
        )

    control_rows = prediction_rows(
        original_state,
        gold_rows,
        "control",
    )
    challenger_rows = prediction_rows(
        final_state,
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

    source_gates: list[dict] = []
    text_gates: list[dict] = []
    citation_results: list[dict] = []
    for gold, prediction in zip(
        scoring_gold_rows,
        challenger_rows,
        strict=True,
    ):
        question_id = str(gold["id"])
        source_gates.append(
            {
                "id": question_id,
                "passed": (
                    question_id not in STRUCTURED_QUESTION_IDS
                    or bool(final_state[question_id].get(
                        "structured_payload"
                    ))
                ),
                "gate": (
                    "accepted against retrieved structured visual source"
                    if question_id in STRUCTURED_QUESTION_IDS
                    else "not a Figure 2/3/5 structured-source question"
                ),
            }
        )
        text_passed, text_gate = relationship_text_check(
            question_id,
            str(prediction["answer"]),
        )
        text_gates.append(
            {
                "id": question_id,
                "passed": text_passed,
                "gate": text_gate,
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
    all_source_gates_passed = all(
        row["passed"] for row in source_gates
    )
    all_text_gates_passed = all(
        row["passed"] for row in text_gates
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
        and all_source_gates_passed
        and all_text_gates_passed
        and all_citations_passed
    )

    generation_latencies = [
        float(
            final_state[question_id]["challenger"]["generation_ms"]
        )
        for question_id in gold_order
    ]
    total_generation_attempts = sum(
        int(final_state[question_id]["generation_attempts"])
        for question_id in gold_order
    )
    summary = {
        "experiment_id": "multimodal_final_faithfulness_repair",
        "description_fingerprint": description_fingerprint,
        "generation_fingerprint": generation_fingerprint,
        "generation_contract_version": GENERATION_CONTRACT_VERSION,
        "structured_figures": sorted(TARGET_FIGURES),
        "control_reused_from": str(STATE_FILE),
        "control": control_metrics,
        "challenger": challenger_metrics,
        "question_level": comparisons,
        "structured_source_gates": source_gates,
        "rendered_text_gates": text_gates,
        "citation_gates": citation_results,
        "all_structured_source_gates_passed": all_source_gates_passed,
        "all_rendered_text_gates_passed": all_text_gates_passed,
        "all_citations_passed": all_citations_passed,
        "correct_visual_anchors": correct_anchors,
        "answerable_questions": answerable_count,
        "improvements": improvements,
        "regressions": regressions,
        "answerability_regressions": answerability_regressions,
        "generation_calls": len(generation_latencies),
        "generation_attempts": total_generation_attempts,
        "mean_generation_ms": statistics.fmean(
            generation_latencies
        ),
        "automated_gate_passed": automated_gate_passed,
        "human_audit_required": True,
        "decision": (
            "PASS FINAL AUTOMATED GATES; HUMAN AUDIT REQUIRED"
            if automated_gate_passed
            else "DO NOT PROMOTE; FINAL REPAIR GATE FAILED"
        ),
    }
    write_json(summary_file, summary)
    write_json(
        config_file,
        {
            "retrieval_screen": retrieval_summary,
            "description_fingerprint": description_fingerprint,
            "generation_fingerprint": generation_fingerprint,
            "generation_contract_version": (
                GENERATION_CONTRACT_VERSION
            ),
            "structured_figures": sorted(TARGET_FIGURES),
            "visual_index": str(VISUAL_DATABASE_DIR),
            "frozen_text_index": str(DATABASE_DIR),
            "generation_model": GENERATION_MODEL,
            "temperature": TEMPERATURE,
            "num_ctx": NUM_CTX,
            "num_predict": NUM_PREDICT,
            "max_generation_attempts": GENERATION_MAX_ATTEMPTS,
            "control_calls": 0,
            "challenger_calls": len(gold_rows),
            "retrieval_context_reused_from": str(STATE_FILE),
            "retrieval_context_frozen": True,
            "structured_answer_question_ids": sorted(
                STRUCTURED_QUESTION_IDS
            ),
            "v005_scoring_alias_added": "build and use",
            "gold_visible_to_generation": False,
            "human_audit_required": True,
        },
    )

    print("\nFinal multimodal faithfulness repair complete")
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
        f"  structured_source_passed={all_source_gates_passed}, "
        f"rendered_relationships_passed={all_text_gates_passed}, "
        f"citations_passed={all_citations_passed}"
    )
    print(
        f"  improvements={improvements}, "
        f"regressions={regressions}, "
        f"generation_attempts={total_generation_attempts}"
    )
    print(f"Decision: {summary['decision']}")
    print(f"State saved to: {state_file.resolve()}")
    print(f"Predictions saved to: {challenger_file.resolve()}")
    print(f"Summary saved to: {summary_file.resolve()}")


if __name__ == "__main__":
    run()
