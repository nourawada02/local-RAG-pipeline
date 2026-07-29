from __future__ import annotations

import argparse
import base64
import json
import re
import shutil
import statistics
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pymupdf
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from pypdf import PdfReader

from build_index import COLLECTION_NAME, EMBEDDING_MODEL
from rag_query import GENERATION_MODEL, stop_ollama_model
from run_balanced_hybrid_evaluation import (
    FINAL_TOP_K,
    normalize_balanced_retrieved,
    page_key,
    reciprocal_rank_fusion_all,
)
from run_decomposed_hybrid_evaluation import (
    normalize_decomposed_retrieved,
    retrieve_with_decomposition,
)
from run_hybrid_evaluation import BM25Index, load_frozen_chunks
from run_ocr_evaluation import (
    CORPUS_DIR,
    DATABASE_DIR,
    RESULTS_DIR,
    SOURCE_TO_DOC_ID,
    open_evaluation_vector_store,
    read_jsonl,
    validate_inputs,
)


# Controlled multimodal experiment:
# - The frozen selective-OCR, 1000/200, mxbai, balanced-hybrid control is
#   never rebuilt or modified.
# - Figure pages are detected from ordinary "Fig. N." caption text, not from
#   benchmark labels or gold pages.
# - Qwen describes each detected figure page once at ingestion time.
# - Descriptions are embedded with the frozen text embedding model in a
#   separate Chroma collection.
# - Only explicitly visual questions may receive one visual-description
#   anchor. The remaining two pages come from the unchanged frozen control.
# - Gold evidence is read only after both selections are complete.
# - This runner screens retrieval only. It performs no answer generation.
EXPERIMENT_DIR = RESULTS_DIR / "multimodal_visual_sweep"
VISUAL_PAGE_DIR = EXPERIMENT_DIR / "rendered_figure_pages"
DESCRIPTION_FILE = EXPERIMENT_DIR / "visual_descriptions.jsonl"
VISUAL_DATABASE_DIR = EXPERIMENT_DIR / "visual_chroma_db"
DETAIL_FILE = EXPERIMENT_DIR / "multimodal_retrieval_by_question.jsonl"
SUMMARY_FILE = EXPERIMENT_DIR / "multimodal_retrieval_summary.json"
CONFIG_FILE = EXPERIMENT_DIR / "multimodal_run_config.json"
VISUAL_GOLD_FILE = (
    Path("rag-evaluation-starter")
    / "evaluation"
    / "multimodal_gold_questions.jsonl"
)
ORIGINAL_GOLD_FILE = (
    Path("rag-evaluation-starter")
    / "evaluation"
    / "gold_questions.jsonl"
)

VISUAL_COLLECTION_NAME = "multimodal_visual_descriptions"
VISION_MODEL = GENERATION_MODEL
VISION_DPI = 180
VISION_TEMPERATURE = 0.0
VISION_NUM_CTX = 4096
VISION_NUM_PREDICT = 700
VISION_MAX_ATTEMPTS = 3
OLLAMA_BASE_URL = "http://localhost:11434"
MIN_DESCRIPTION_CHARACTERS = 220
VISUAL_CANDIDATE_K = 5

FIGURE_CAPTION_PATTERN = re.compile(
    r"(?mi)^\s*Fig\.\s*(\d+)\.",
)
VISUAL_QUERY_PATTERN = re.compile(
    r"\b("
    r"figure|fig\.?|diagram|chart|visual|wheel|outer\s+ring|inner\s+ring|"
    r"positioned|vertical\s+characteristic|top\s+row"
    r")\b",
    flags=re.IGNORECASE,
)


@dataclass(frozen=True)
class FigurePage:
    source: str
    page: int
    figure_numbers: tuple[int, ...]
    caption_text: str


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


def detect_figure_pages() -> list[FigurePage]:
    """
    Detect figure-bearing native PDF pages using caption syntax only.

    The scanned corpus is intentionally not sent to the vision model here.
    Selective OCR already handles its image-only text; calling that
    "multimodal" would not test diagrams or visual relationships.
    """
    detected: list[FigurePage] = []

    for pdf_path in sorted(CORPUS_DIR.glob("*.pdf")):
        try:
            reader = PdfReader(str(pdf_path), strict=False)
        except Exception:
            continue

        for page_number, page in enumerate(reader.pages, start=1):
            try:
                text = page.extract_text() or ""
            except Exception:
                continue
            figure_numbers = tuple(
                int(value)
                for value in FIGURE_CAPTION_PATTERN.findall(text)
            )
            if not figure_numbers:
                continue

            caption_lines = [
                normalize_space(line)
                for line in text.splitlines()
                if re.match(r"^\s*Fig\.\s*\d+\.", line)
            ]
            detected.append(
                FigurePage(
                    source=pdf_path.name,
                    page=page_number,
                    figure_numbers=figure_numbers,
                    caption_text=" ".join(caption_lines),
                )
            )

    detected.sort(
        key=lambda item: (
            item.source,
            item.page,
            item.figure_numbers,
        )
    )
    if not detected:
        raise RuntimeError(
            "No figure pages were detected from 'Fig. N.' captions."
        )
    return detected


def verify_ollama_vision_capability() -> None:
    payload = json.dumps({"model": VISION_MODEL}).encode("utf-8")
    request = urllib.request.Request(
        f"{OLLAMA_BASE_URL}/api/show",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            model_info = json.loads(response.read().decode("utf-8"))
    except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            "Could not inspect the Ollama model. Confirm Ollama is running "
            f"and '{VISION_MODEL}' is installed."
        ) from exc

    capabilities = {
        str(value).casefold()
        for value in model_info.get("capabilities", [])
    }
    if "vision" not in capabilities:
        raise RuntimeError(
            f"Ollama reports that '{VISION_MODEL}' has no vision "
            "capability. Update Ollama and confirm the installed tag with "
            f"'ollama show {VISION_MODEL}'."
        )


def render_figure_page(item: FigurePage) -> tuple[Path, bytes]:
    pdf_path = CORPUS_DIR / item.source
    VISUAL_PAGE_DIR.mkdir(parents=True, exist_ok=True)
    output_path = (
        VISUAL_PAGE_DIR
        / f"{Path(item.source).stem}-p{item.page:03d}.png"
    )

    with pymupdf.open(str(pdf_path)) as document:
        if not 1 <= item.page <= document.page_count:
            raise ValueError(
                f"Invalid physical page {item.page} for {item.source}."
            )
        page = document.load_page(item.page - 1)
        matrix = pymupdf.Matrix(
            VISION_DPI / 72,
            VISION_DPI / 72,
        )
        clip = None
        if 2 in item.figure_numbers:
            # Enlarge the lifecycle wheel and exclude unrelated body prose.
            clip = pymupdf.Rect(70, 65, 542, 382)
        elif 3 in item.figure_numbers:
            # Figure 3 is sideways and its lower rows are dense. The repair
            # needs only Key Dimensions, Lifecycle Stage, and TEVV, so crop
            # away Activities, Representative Actors, and the caption.
            clip = pymupdf.Rect(105, 65, 255, 690)
        elif 5 in item.figure_numbers:
            # Isolate the Core diagram so spatial labels dominate the input.
            clip = pymupdf.Rect(75, 175, 537, 548)
        if 3 in item.figure_numbers:
            # Figure 3 is stored sideways. Physical rotation is much more
            # reliable than asking a small vision model to rotate mentally.
            matrix = matrix.prerotate(90)
        pixmap = page.get_pixmap(
            matrix=matrix,
            clip=clip,
            colorspace=pymupdf.csRGB,
            alpha=False,
        )
        image_bytes = pixmap.tobytes("png")

    output_path.write_bytes(image_bytes)
    return output_path, image_bytes


def vision_prompt(
    item: FigurePage,
    *,
    retrying_after_quality_failure: bool = False,
) -> str:
    figures = ", ".join(
        f"Figure {number}" for number in item.figure_numbers
    )
    if 2 in item.figure_numbers:
        prompt = (
            "Transcribe the lifecycle wheel into the supplied JSON schema. "
            "Record only: (1) the single center label, (2) the four Key "
            "Dimension labels in the inner grouping, and (3) the six "
            "Lifecycle Stage labels in the outer ring. Keep key dimensions "
            "and lifecycle stages in separate arrays. Do not count the "
            "center as a key dimension, and do not move lifecycle stages "
            "into the key_dimensions array. Preserve visible wording. "
            "Return JSON only."
        )
    elif 3 in item.figure_numbers:
        prompt = (
            "Read the seven visible columns from left to right. For every "
            "column, transcribe its column number, Key Dimension, Lifecycle "
            "Stage, and TEVV text into the supplied JSON schema. Use columns "
            "1 through 7 exactly once. Omit the repeated words 'TEVV "
            "includes' from the tevv value. Preserve the remaining visible "
            "wording. Do not infer, summarize, combine columns, or move a "
            "TEVV phrase to a neighboring column. Return JSON only."
        )
    elif 5 in item.figure_numbers:
        prompt = (
            "Transcribe the four labeled positions into the supplied JSON "
            "schema. Use CENTER, LEFT, RIGHT, and BELOW exactly once. For "
            "each position, record the visible function and explanatory "
            "text. "
            "Read positions relative to the central circle. Do not call the "
            "center function an outer function, do not invent a top-right "
            "position, and do not repeat the response. Return JSON only."
        )
    else:
        prompt = (
            "You are transcribing a document figure for retrieval. Inspect "
            "only the labeled visual, not its caption or surrounding "
            "paragraphs. Return plain text under exactly these headings: "
            "FIGURE TYPE, EXACT LABELS, GROUPING AND MAPPINGS, SPATIAL "
            "RELATIONSHIPS, and TEXT INSIDE ELEMENTS. Enumerate every "
            "visible wheel segment, column, row, node, or box. Preserve "
            "exact label wording. Explicitly map each outer label to its "
            "inner group and each column to its row values. State "
            "relationships such as 'X is left of Y', 'A surrounds B', or "
            "'stage S belongs under dimension D'. Do not summarize the "
            "page, copy body paragraphs, infer progression from decorative "
            "lines, or add facts that are not visible. Do not answer any "
            "user question and do not use a Markdown table."
        )
    if retrying_after_quality_failure:
        prompt += (
            "\nYour previous transcription failed the exact structural and "
            "consistency checks. Re-read the image carefully and produce a "
            "fresh transcription in the required format."
        )
    return (
        f"{prompt}\n\n"
        f"Source file: {item.source}\n"
        f"Physical PDF page: {item.page}\n"
        f"Detected caption labels: {figures}\n"
        f"Extracted caption cue: {item.caption_text or '(none)'}"
    )


def clean_vision_output(raw: str) -> str:
    cleaned = raw.strip()
    cleaned = re.sub(
        r"(?is)<think>.*?</think>",
        "",
        cleaned,
    ).strip()
    if "</think>" in cleaned:
        cleaned = cleaned.split("</think>", 1)[1].strip()
    if cleaned.startswith("```"):
        first_newline = cleaned.find("\n")
        if first_newline != -1:
            cleaned = cleaned[first_newline + 1 :]
        if cleaned.rstrip().endswith("```"):
            cleaned = cleaned.rstrip()[:-3]
    return "\n".join(
        line.strip()
        for line in cleaned.splitlines()
        if line.strip()
    ).strip()


def canonical_visual_text(value: str) -> str:
    value = value.casefold().replace("&", " and ")
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return normalize_space(value)


FIGURE_3_EXPECTED_COLUMNS = (
    (
        "application context",
        "plan and design",
        "audit and impact assessment",
    ),
    (
        "data and input",
        "collect and process data",
        "internal and external validation",
    ),
    ("ai model", "build and use model", "model testing"),
    ("ai model", "verify and validate", "model testing"),
    (
        "task and output",
        "deploy and use",
        "integration compliance testing and validation",
    ),
    (
        "application context",
        "operate and monitor",
        "audit and impact assessment",
    ),
    (
        "people and planet",
        "use or impacted by",
        "audit and impact assessment",
    ),
)
FIGURE_3_COLUMN_PATTERN = re.compile(
    r"(?i)^\s*COLUMN\s+([1-7])\s*\|\s*"
    r"KEY\s+DIMENSION\s*:\s*(.*?)\s*\|\s*"
    r"LIFECYCLE\s+STAGE\s*:\s*(.*?)\s*\|\s*"
    r"TEVV\s*:\s*(.*?)\s*$"
)
FIGURE_2_EXPECTED_CENTER = "people and planet"
FIGURE_2_EXPECTED_DIMENSIONS = {
    "application context",
    "data and input",
    "ai model",
    "task and output",
}
FIGURE_2_EXPECTED_STAGES = {
    "plan and design",
    "collect and process data",
    "build and use model",
    "verify and validate",
    "deploy and use",
    "operate and monitor",
}
FIGURE_2_CENTER_PATTERN = re.compile(
    r"(?i)^\s*CENTER\s*:\s*(.*?)\s*$"
)
FIGURE_2_DIMENSION_PATTERN = re.compile(
    r"(?i)^\s*KEY\s+DIMENSION\s*:\s*(.*?)\s*$"
)
FIGURE_2_STAGE_PATTERN = re.compile(
    r"(?i)^\s*LIFECYCLE\s+STAGE\s*:\s*(.*?)\s*$"
)
FIGURE_5_EXPECTED_POSITIONS = {
    "CENTER": "govern",
    "LEFT": "map",
    "RIGHT": "measure",
    "BELOW": "manage",
}
FIGURE_5_POSITION_PATTERN = re.compile(
    r"(?i)^\s*(CENTER|LEFT|RIGHT|BELOW)\s*:\s*"
    r"([A-Za-z& -]+?)(?:\s*\|.*)?\s*$"
)


def visual_output_schema(item: FigurePage) -> dict | None:
    """
    Constrain the three repaired figures structurally at generation time.

    The source-grounded validator below remains authoritative for factual
    correctness. The schema only prevents harmless formatting drift.
    """
    if 2 in item.figure_numbers:
        return {
            "type": "object",
            "properties": {
                "center": {"type": "string"},
                "key_dimensions": {
                    "type": "array",
                    "minItems": 4,
                    "maxItems": 4,
                    "items": {"type": "string"},
                },
                "lifecycle_stages": {
                    "type": "array",
                    "minItems": 6,
                    "maxItems": 6,
                    "items": {"type": "string"},
                },
            },
            "required": [
                "center",
                "key_dimensions",
                "lifecycle_stages",
            ],
            "additionalProperties": False,
        }
    if 3 in item.figure_numbers:
        return {
            "type": "object",
            "properties": {
                "columns": {
                    "type": "array",
                    "minItems": 7,
                    "maxItems": 7,
                    "items": {
                        "type": "object",
                        "properties": {
                            "column": {
                                "type": "integer",
                                "minimum": 1,
                                "maximum": 7,
                            },
                            "key_dimension": {"type": "string"},
                            "lifecycle_stage": {"type": "string"},
                            "tevv": {"type": "string"},
                        },
                        "required": [
                            "column",
                            "key_dimension",
                            "lifecycle_stage",
                            "tevv",
                        ],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["columns"],
            "additionalProperties": False,
        }
    if 5 in item.figure_numbers:
        return {
            "type": "object",
            "properties": {
                "positions": {
                    "type": "array",
                    "minItems": 4,
                    "maxItems": 4,
                    "items": {
                        "type": "object",
                        "properties": {
                            "role": {
                                "type": "string",
                                "enum": [
                                    "CENTER",
                                    "LEFT",
                                    "RIGHT",
                                    "BELOW",
                                ],
                            },
                            "function": {"type": "string"},
                            "explanatory_text": {"type": "string"},
                        },
                        "required": [
                            "role",
                            "function",
                            "explanatory_text",
                        ],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["positions"],
            "additionalProperties": False,
        }
    return None


def normalize_structured_visual_output(
    item: FigurePage,
    raw: str,
) -> str:
    """
    Convert schema-constrained model fields into stable retrieval text.

    No expected label or mapping is inserted here: every value in the
    normalized description comes from the model response.
    """
    cleaned = clean_vision_output(raw)
    if not ({2, 3, 5}.intersection(item.figure_numbers)):
        return cleaned

    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ValueError(
            "structured vision output was not valid JSON"
        ) from exc
    if not isinstance(payload, dict):
        raise ValueError("structured vision output was not a JSON object")

    if 2 in item.figure_numbers:
        try:
            center = str(payload["center"]).strip()
            key_dimensions = payload["key_dimensions"]
            lifecycle_stages = payload["lifecycle_stages"]
        except KeyError as exc:
            raise ValueError(
                "Figure 2 JSON omitted a required field"
            ) from exc
        if not center:
            raise ValueError("Figure 2 JSON contained an empty center")
        if not isinstance(key_dimensions, list):
            raise ValueError(
                "Figure 2 JSON did not contain a key_dimensions array"
            )
        if not isinstance(lifecycle_stages, list):
            raise ValueError(
                "Figure 2 JSON did not contain a lifecycle_stages array"
            )
        dimensions = [str(value).strip() for value in key_dimensions]
        stages = [str(value).strip() for value in lifecycle_stages]
        if any(not value for value in dimensions + stages):
            raise ValueError(
                "Figure 2 JSON contained an empty array value"
            )
        return "\n".join(
            [f"CENTER: {center}"]
            + [f"KEY DIMENSION: {value}" for value in dimensions]
            + [f"LIFECYCLE STAGE: {value}" for value in stages]
        )

    if 3 in item.figure_numbers:
        columns = payload.get("columns")
        if not isinstance(columns, list):
            raise ValueError(
                "Figure 3 JSON did not contain a columns array"
            )
        lines: list[str] = []
        for value in columns:
            if not isinstance(value, dict):
                raise ValueError(
                    "Figure 3 columns contained a non-object value"
                )
            try:
                column = int(value["column"])
                key_dimension = str(value["key_dimension"]).strip()
                lifecycle_stage = str(
                    value["lifecycle_stage"]
                ).strip()
                tevv = str(value["tevv"]).strip()
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    "Figure 3 JSON omitted a required column field"
                ) from exc
            if not key_dimension or not lifecycle_stage or not tevv:
                raise ValueError(
                    "Figure 3 JSON contained an empty column field"
                )
            tevv = re.sub(
                r"(?i)^\s*TEVV\s+includes\s+",
                "",
                tevv,
            ).strip()
            lines.append(
                f"COLUMN {column} | KEY DIMENSION: {key_dimension} | "
                f"LIFECYCLE STAGE: {lifecycle_stage} | TEVV: {tevv}"
            )
        return "\n".join(lines)

    positions = payload.get("positions")
    if not isinstance(positions, list):
        raise ValueError(
            "Figure 5 JSON did not contain a positions array"
        )
    position_lines: list[str] = []
    for value in positions:
        if not isinstance(value, dict):
            raise ValueError(
                "Figure 5 positions contained a non-object value"
            )
        try:
            role = str(value["role"]).strip().upper()
            function = str(value["function"]).strip()
            explanatory_text = str(
                value["explanatory_text"]
            ).strip()
        except KeyError as exc:
            raise ValueError(
                "Figure 5 JSON omitted a required position field"
            ) from exc
        if not role or not function or not explanatory_text:
            raise ValueError(
                "Figure 5 JSON contained an empty position field"
            )
        position_lines.append(
            f"{role}: {function} | {explanatory_text}"
        )
    return "\n".join(position_lines)


def validate_visual_description(
    item: FigurePage,
    description: str,
) -> None:
    if len(description) < MIN_DESCRIPTION_CHARACTERS:
        raise ValueError(
            f"description was only {len(description)} characters"
        )

    if 2 in item.figure_numbers:
        centers: list[str] = []
        dimensions: list[str] = []
        stages: list[str] = []
        for line in description.splitlines():
            center_match = FIGURE_2_CENTER_PATTERN.fullmatch(line)
            dimension_match = FIGURE_2_DIMENSION_PATTERN.fullmatch(line)
            stage_match = FIGURE_2_STAGE_PATTERN.fullmatch(line)
            matches = [
                match
                for match in (
                    center_match,
                    dimension_match,
                    stage_match,
                )
                if match is not None
            ]
            if len(matches) != 1:
                raise ValueError(
                    "Figure 2 output mixed or omitted the center, key "
                    "dimension, and lifecycle stage line types"
                )
            value = canonical_visual_text(matches[0].group(1))
            if center_match is not None:
                centers.append(value)
            elif dimension_match is not None:
                dimensions.append(value)
            else:
                stages.append(value)
        if centers != [FIGURE_2_EXPECTED_CENTER]:
            raise ValueError(
                "Figure 2 failed source-grounded center validation"
            )
        if (
            len(dimensions) != 4
            or set(dimensions) != FIGURE_2_EXPECTED_DIMENSIONS
        ):
            raise ValueError(
                "Figure 2 failed source-grounded four-dimension validation"
            )
        if (
            len(stages) != 6
            or set(stages) != FIGURE_2_EXPECTED_STAGES
        ):
            raise ValueError(
                "Figure 2 failed source-grounded six-stage validation"
            )

    if 3 in item.figure_numbers:
        parsed: dict[int, tuple[str, str, str]] = {}
        for line in description.splitlines():
            match = FIGURE_3_COLUMN_PATTERN.fullmatch(line)
            if match is None:
                raise ValueError(
                    "Figure 3 output did not use the exact seven-column "
                    "line format"
                )
            column = int(match.group(1))
            if column in parsed:
                raise ValueError(
                    f"Figure 3 repeated column {column}"
                )
            parsed[column] = tuple(
                canonical_visual_text(value)
                for value in match.groups()[1:]
            )
        if sorted(parsed) != list(range(1, 8)):
            raise ValueError(
                "Figure 3 did not return columns 1 through 7 exactly once"
            )
        for column, expected in enumerate(
            FIGURE_3_EXPECTED_COLUMNS,
            start=1,
        ):
            if parsed[column] != expected:
                raise ValueError(
                    f"Figure 3 column {column} failed source-grounded "
                    "mapping validation"
                )

    if 5 in item.figure_numbers:
        parsed_positions: dict[str, str] = {}
        for line in description.splitlines():
            match = FIGURE_5_POSITION_PATTERN.fullmatch(line)
            if match is None:
                raise ValueError(
                    "Figure 5 output did not use the exact four-position "
                    "line format"
                )
            role = match.group(1).upper()
            if role in parsed_positions:
                raise ValueError(f"Figure 5 repeated role {role}")
            parsed_positions[role] = canonical_visual_text(match.group(2))
        if set(parsed_positions) != set(FIGURE_5_EXPECTED_POSITIONS):
            raise ValueError(
                "Figure 5 did not return CENTER, LEFT, RIGHT, and BELOW "
                "exactly once"
            )
        if parsed_positions != FIGURE_5_EXPECTED_POSITIONS:
            raise ValueError(
                "Figure 5 failed source-grounded position validation"
            )


def call_ollama_vision(item: FigurePage, image_bytes: bytes) -> str:
    last_quality_error = ""
    for attempt in range(1, VISION_MAX_ATTEMPTS + 1):
        request_body = {
            "model": VISION_MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": vision_prompt(
                        item,
                        retrying_after_quality_failure=attempt > 1,
                    ),
                    "images": [
                        base64.b64encode(image_bytes).decode("ascii")
                    ],
                }
            ],
            "stream": False,
            "think": False,
            "options": {
                "temperature": VISION_TEMPERATURE,
                "num_ctx": VISION_NUM_CTX,
                "num_predict": VISION_NUM_PREDICT,
            },
        }
        schema = visual_output_schema(item)
        if schema is not None:
            request_body["format"] = schema
        request = urllib.request.Request(
            f"{OLLAMA_BASE_URL}/api/chat",
            data=json.dumps(request_body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(
                request,
                timeout=300,
            ) as response:
                result = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"Ollama vision request failed with HTTP {exc.code}: {body}"
            ) from exc
        except (
            OSError,
            urllib.error.URLError,
            json.JSONDecodeError,
        ) as exc:
            raise RuntimeError(
                f"Ollama vision request failed for {item.source}, "
                f"page {item.page}: {exc}"
            ) from exc

        try:
            description = normalize_structured_visual_output(
                item,
                str((result.get("message") or {}).get("content", "")),
            )
            validate_visual_description(item, description)
            return description
        except ValueError as exc:
            last_quality_error = str(exc)
            if attempt < VISION_MAX_ATTEMPTS:
                print(
                    f"    quality retry {attempt + 1}/"
                    f"{VISION_MAX_ATTEMPTS} for {item.source}:p{item.page} "
                    f"({last_quality_error})"
                )

    raise RuntimeError(
        f"Vision description for {item.source}, page {item.page} failed "
        f"the quality gate after {VISION_MAX_ATTEMPTS} attempts: "
        f"{last_quality_error}"
    )


def description_key(row: dict) -> tuple[str, int]:
    return str(row["source"]), int(row["page"])


def load_completed_descriptions() -> dict[tuple[str, int], dict]:
    if not DESCRIPTION_FILE.exists():
        return {}
    completed: dict[tuple[str, int], dict] = {}
    for row in read_jsonl(DESCRIPTION_FILE):
        key = description_key(row)
        if key in completed:
            raise ValueError(
                f"Duplicate visual description for {key}."
            )
        completed[key] = row
    return completed


def visual_document(row: dict, chunk_number: int) -> Document:
    figures = ", ".join(
        f"Figure {int(value)}" for value in row["figure_numbers"]
    )
    content = (
        f"Visual description from {row['source']}, physical PDF page "
        f"{int(row['page'])}. {figures}. "
        f"Caption: {row.get('caption_text') or '(not extracted)'}. "
        f"Visual content: {row['description']}"
    )
    return Document(
        page_content=content,
        metadata={
            "source": str(row["source"]),
            "page": int(row["page"]),
            "chunk": int(chunk_number),
            "content_type": "visual_description",
            "extraction_method": "vision",
            "figure_numbers": ",".join(
                str(int(value)) for value in row["figure_numbers"]
            ),
            "vision_model": VISION_MODEL,
        },
    )


def build_visual_index(
    force_descriptions: bool = False,
    force_pages: set[int] | None = None,
) -> None:
    validate_inputs()
    if not DATABASE_DIR.exists():
        raise FileNotFoundError(
            "The frozen selective-OCR index is missing. Do not rebuild it "
            "inside this experiment."
        )
    if not VISUAL_GOLD_FILE.exists():
        raise FileNotFoundError(
            f"Visual benchmark not found: {VISUAL_GOLD_FILE}"
        )

    figure_pages = detect_figure_pages()
    detected_numbers = sorted(
        number
        for item in figure_pages
        for number in item.figure_numbers
    )
    print(
        f"Detected {len(figure_pages)} figure pages: "
        + ", ".join(
            (
                f"{item.source}:p{item.page} "
                f"(Fig. {','.join(map(str, item.figure_numbers))})"
            )
            for item in figure_pages
        )
    )

    if force_descriptions:
        completed: dict[tuple[str, int], dict] = {}
    else:
        completed = load_completed_descriptions()
        for key in list(completed):
            if force_pages and key[1] in force_pages:
                completed.pop(key)

    detected_pages = {item.page for item in figure_pages}
    unknown_force_pages = set(force_pages or ()) - detected_pages
    if unknown_force_pages:
        raise ValueError(
            "--force-pages contains pages without detected figures: "
            + ", ".join(map(str, sorted(unknown_force_pages)))
        )

    verify_ollama_vision_capability()
    rows: list[dict] = []
    pending_count = sum(
        (item.source, item.page) not in completed
        for item in figure_pages
    )
    print(
        f"Visual descriptions: completed={len(completed)}, "
        f"pending={pending_count}"
    )

    try:
        for position, item in enumerate(figure_pages, start=1):
            key = (item.source, item.page)
            if key in completed:
                rows.append(completed[key])
                print(
                    f"  describe [{position}/{len(figure_pages)}] "
                    f"{item.source}:p{item.page} reused"
                )
                continue

            stop_ollama_model(EMBEDDING_MODEL)
            image_path, image_bytes = render_figure_page(item)
            started = time.perf_counter()
            description = call_ollama_vision(item, image_bytes)
            latency_ms = (time.perf_counter() - started) * 1000
            row = {
                "source": item.source,
                "page": item.page,
                "figure_numbers": list(item.figure_numbers),
                "caption_text": item.caption_text,
                "description": description,
                "vision_model": VISION_MODEL,
                "render_dpi": VISION_DPI,
                "rendered_image": str(image_path),
                "vision_latency_ms": round(latency_ms, 3),
            }
            completed[key] = row
            rows.append(row)
            ordered_rows = [
                completed[(page.source, page.page)]
                for page in figure_pages
                if (page.source, page.page) in completed
            ]
            write_jsonl(DESCRIPTION_FILE, ordered_rows)
            print(
                f"  describe [{position}/{len(figure_pages)}] "
                f"{item.source}:p{item.page} "
                f"{len(description)} chars {latency_ms:.0f} ms"
            )
    finally:
        stop_ollama_model(VISION_MODEL)

    rows = [
        completed[(item.source, item.page)]
        for item in figure_pages
    ]
    write_jsonl(DESCRIPTION_FILE, rows)
    documents = [
        visual_document(row, chunk_number=900000 + index)
        for index, row in enumerate(rows, start=1)
    ]

    if VISUAL_DATABASE_DIR.exists():
        shutil.rmtree(VISUAL_DATABASE_DIR)

    print(
        f"Embedding {len(documents)} visual descriptions with "
        f"{EMBEDDING_MODEL}"
    )
    vector_store = Chroma(
        collection_name=VISUAL_COLLECTION_NAME,
        embedding_function=OllamaEmbeddings(model=EMBEDDING_MODEL),
        collection_metadata={"hnsw:space": "cosine"},
        persist_directory=str(VISUAL_DATABASE_DIR),
    )
    try:
        vector_store.add_documents(documents)
    finally:
        stop_ollama_model(EMBEDDING_MODEL)

    config = {
        "experiment_id": "selective_visual_description_retrieval",
        "control": (
            "frozen selective-OCR 1000/200 balanced-hybrid "
            "with explicit both-and decomposition"
        ),
        "independent_variable": (
            "one vision-description anchor for explicitly visual queries"
        ),
        "figure_detection": "native caption regex: line starts with Fig. N.",
        "detected_figure_numbers": detected_numbers,
        "detected_figure_pages": [
            {
                "source": item.source,
                "page": item.page,
                "figure_numbers": list(item.figure_numbers),
            }
            for item in figure_pages
        ],
        "vision_model": VISION_MODEL,
        "vision_render_dpi": VISION_DPI,
        "vision_temperature": VISION_TEMPERATURE,
        "vision_num_ctx": VISION_NUM_CTX,
        "vision_num_predict": VISION_NUM_PREDICT,
        "embedding_model": EMBEDDING_MODEL,
        "visual_collection": VISUAL_COLLECTION_NAME,
        "visual_chunks": len(documents),
        "final_context_pages": FINAL_TOP_K,
        "answer_generation": False,
        "gold_used_after_selection_only": True,
    }
    write_json(CONFIG_FILE, config)
    print(f"Descriptions saved to: {DESCRIPTION_FILE.resolve()}")
    print(f"Visual index saved to: {VISUAL_DATABASE_DIR.resolve()}")


def open_visual_vector_store() -> Chroma:
    if not VISUAL_DATABASE_DIR.exists():
        raise FileNotFoundError(
            "The visual index is missing. First run:\n"
            "python run_multimodal_visual_retrieval_sweep.py --build-index"
        )
    return Chroma(
        collection_name=VISUAL_COLLECTION_NAME,
        embedding_function=OllamaEmbeddings(model=EMBEDDING_MODEL),
        persist_directory=str(VISUAL_DATABASE_DIR),
    )


def is_visual_query(question: str) -> bool:
    return VISUAL_QUERY_PATTERN.search(question) is not None


def validate_original_questions_are_frozen() -> None:
    routed = [
        str(row["id"])
        for row in read_jsonl(ORIGINAL_GOLD_FILE)
        if is_visual_query(str(row["question"]))
    ]
    if routed:
        raise RuntimeError(
            "The visual router would alter original benchmark questions: "
            + ", ".join(routed)
        )


def normalize_control(
    results: list[tuple[Document, float]],
    decomposed_selected,
    balanced_selected,
) -> list[dict]:
    if decomposed_selected is None:
        return normalize_balanced_retrieved(balanced_selected)
    return normalize_decomposed_retrieved(decomposed_selected)


def retrieve_visual_anchor(
    visual_store: Chroma,
    visual_bm25: BM25Index,
    question: str,
) -> tuple[Document, dict]:
    dense_results = visual_store.similarity_search_with_score(
        query=question,
        k=VISUAL_CANDIDATE_K,
    )
    bm25_results = visual_bm25.search(
        query=question,
        k=VISUAL_CANDIDATE_K,
    )
    fused = reciprocal_rank_fusion_all(
        dense_results=dense_results,
        bm25_results=bm25_results,
    )
    if not fused:
        raise RuntimeError("Visual retrieval returned no candidates.")

    anchor = fused[0]
    metadata = anchor.document.metadata
    return anchor.document, {
        "dense_rank": anchor.dense_rank,
        "dense_distance": anchor.dense_distance,
        "bm25_rank": anchor.bm25_rank,
        "bm25_score": anchor.bm25_score,
        "visual_rrf_score": anchor.rrf_score,
    }


def challenger_selection(
    control_results: list[tuple[Document, float]],
    control_normalized: list[dict],
    visual_document: Document,
    visual_diagnostics: dict,
) -> tuple[list[tuple[Document, float]], list[dict]]:
    selected_results: list[tuple[Document, float]] = [
        (visual_document, -float(visual_diagnostics["visual_rrf_score"]))
    ]
    selected_normalized: list[dict] = [
        {
            "rank": 1,
            "doc_id": SOURCE_TO_DOC_ID[
                str(visual_document.metadata["source"])
            ],
            "pages": [int(visual_document.metadata["page"])],
            "chunk": int(visual_document.metadata["chunk"]),
            "distance": visual_diagnostics["dense_distance"],
            "hybrid_rrf_score": (
                visual_diagnostics["visual_rrf_score"]
            ),
            "dense_rank": visual_diagnostics["dense_rank"],
            "bm25_rank": visual_diagnostics["bm25_rank"],
            "bm25_score": visual_diagnostics["bm25_score"],
            "selection_role": "visual_anchor",
            "content_type": "visual_description",
            "text": visual_document.page_content.strip(),
        }
    ]
    selected_pages = {page_key(visual_document)}

    for result, normalized in zip(
        control_results,
        control_normalized,
    ):
        document, score = result
        if page_key(document) in selected_pages:
            continue
        selected_results.append((document, score))
        selected_pages.add(page_key(document))
        row = dict(normalized)
        row["rank"] = len(selected_normalized) + 1
        row["selection_role"] = "frozen_control_fill"
        row["content_type"] = str(
            document.metadata.get("content_type", "text")
        )
        selected_normalized.append(row)
        if len(selected_results) == FINAL_TOP_K:
            break

    if len(selected_results) != FINAL_TOP_K:
        raise RuntimeError(
            f"Multimodal selector produced {len(selected_results)} pages; "
            f"expected {FINAL_TOP_K}."
        )
    return selected_results, selected_normalized


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


def question_retrieval_metrics(
    retrieved: list[dict],
    gold: list[dict],
) -> dict[str, float]:
    gold_units = evidence_units(gold)
    seen: set[tuple[str, int]] = set()
    first_rank: int | None = None
    for rank, item in enumerate(retrieved[:FINAL_TOP_K], start=1):
        relevant = evidence_units([item]) & gold_units
        if relevant and first_rank is None:
            first_rank = rank
        seen.update(relevant)
    return {
        "hit@3": float(bool(seen)),
        "recall@3": (
            len(seen) / len(gold_units) if gold_units else 0.0
        ),
        "mrr@3": 0.0 if first_rank is None else 1.0 / first_rank,
    }


def aggregate(rows: list[dict], key: str) -> dict[str, float]:
    answerable = [row for row in rows if row["answerable"]]
    return {
        metric: statistics.fmean(
            float(row[key][metric]) for row in answerable
        )
        for metric in ("hit@3", "recall@3", "mrr@3")
    }


def run_retrieval_screen() -> None:
    validate_inputs()
    validate_original_questions_are_frozen()
    if not VISUAL_GOLD_FILE.exists():
        raise FileNotFoundError(
            f"Visual benchmark not found: {VISUAL_GOLD_FILE}"
        )
    if not DESCRIPTION_FILE.exists() or not VISUAL_DATABASE_DIR.exists():
        raise FileNotFoundError(
            "Visual descriptions/index are missing. First run:\n"
            "python run_multimodal_visual_retrieval_sweep.py --build-index"
        )

    gold_rows = read_jsonl(VISUAL_GOLD_FILE)
    if not gold_rows:
        raise ValueError("The visual benchmark is empty.")
    non_visual_ids = [
        str(row["id"])
        for row in gold_rows
        if not is_visual_query(str(row["question"]))
    ]
    if non_visual_ids:
        raise ValueError(
            "Visual benchmark questions do not pass the label-blind "
            f"visual router: {non_visual_ids}"
        )

    text_store = open_evaluation_vector_store()
    text_bm25 = BM25Index(load_frozen_chunks(text_store))
    visual_store = open_visual_vector_store()
    visual_documents = [
        visual_document(row, chunk_number=900000 + index)
        for index, row in enumerate(
            read_jsonl(DESCRIPTION_FILE),
            start=1,
        )
    ]
    visual_bm25 = BM25Index(visual_documents)

    print("Controlled experiment: selective multimodal visual retrieval")
    print(f"Reusing frozen OCR index: {DATABASE_DIR.resolve()}")
    print(
        f"Visual descriptions: {len(visual_documents)} figure pages; "
        f"model={VISION_MODEL}; embedder={EMBEDDING_MODEL}"
    )
    print(
        "Control: frozen decomposed balanced-hybrid top 3"
    )
    print(
        "Challenger: one visual anchor + two unique frozen control pages"
    )
    print(
        "Original 30-question benchmark: mechanically frozen; "
        "zero visual routes"
    )
    print("Final answer generation: disabled")
    print("Gold evidence: post-selection evaluation only")

    rows: list[dict] = []
    try:
        for position, gold in enumerate(gold_rows, start=1):
            question_id = str(gold["id"])
            question = str(gold["question"])
            started = time.perf_counter()

            (
                control_results,
                decomposed_selected,
                balanced_selected,
                subqueries,
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
            anchor_document, visual_diagnostics = retrieve_visual_anchor(
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
                visual_document=anchor_document,
                visual_diagnostics=visual_diagnostics,
            )
            retrieval_ms = (time.perf_counter() - started) * 1000

            answerable = bool(gold["answerable"])
            control_metrics = (
                question_retrieval_metrics(
                    control_normalized,
                    list(gold["gold_evidence"]),
                )
                if answerable
                else {"hit@3": 0.0, "recall@3": 0.0, "mrr@3": 0.0}
            )
            challenger_metrics = (
                question_retrieval_metrics(
                    challenger_normalized,
                    list(gold["gold_evidence"]),
                )
                if answerable
                else {"hit@3": 0.0, "recall@3": 0.0, "mrr@3": 0.0}
            )
            anchor_source = str(anchor_document.metadata["source"])
            anchor_page = int(anchor_document.metadata["page"])
            row = {
                "id": question_id,
                "question": question,
                "type": str(gold["type"]),
                "answerable": answerable,
                "router": {
                    "routed": True,
                    "reason": "explicit visual-language cue",
                },
                "decomposition": {
                    "applied": len(subqueries) > 1,
                    "subqueries": (
                        subqueries if len(subqueries) > 1 else []
                    ),
                },
                "control_retrieved": control_normalized,
                "challenger_retrieved": challenger_normalized,
                "visual_anchor": {
                    "doc_id": SOURCE_TO_DOC_ID[anchor_source],
                    "page": anchor_page,
                    **visual_diagnostics,
                },
                "control_metrics": control_metrics,
                "challenger_metrics": challenger_metrics,
                "retrieval_ms": round(retrieval_ms, 3),
            }
            rows.append(row)
            marker = (
                "unanswerable"
                if not answerable
                else (
                    "improved"
                    if challenger_metrics["recall@3"]
                    > control_metrics["recall@3"]
                    else "same"
                )
            )
            print(
                f"  retrieve [{position:02d}/{len(gold_rows):02d}] "
                f"{question_id} anchor={anchor_source}:p{anchor_page} "
                f"result={marker} {retrieval_ms:.1f} ms"
            )
    finally:
        stop_ollama_model(EMBEDDING_MODEL)

    control = aggregate(rows, "control_metrics")
    challenger = aggregate(rows, "challenger_metrics")
    improvements = sum(
        row["answerable"]
        and row["challenger_metrics"]["recall@3"]
        > row["control_metrics"]["recall@3"]
        for row in rows
    )
    regressions = sum(
        row["answerable"]
        and row["challenger_metrics"]["recall@3"]
        < row["control_metrics"]["recall@3"]
        for row in rows
    )
    correct_anchors = sum(
        row["answerable"]
        and (
            (
                row["visual_anchor"]["doc_id"],
                row["visual_anchor"]["page"],
            )
            in evidence_units(
                next(
                    gold["gold_evidence"]
                    for gold in gold_rows
                    if str(gold["id"]) == row["id"]
                )
            )
        )
        for row in rows
    )
    promotion = (
        challenger["recall@3"] > control["recall@3"]
        and challenger["hit@3"] >= control["hit@3"]
        and challenger["mrr@3"] >= control["mrr@3"]
        and regressions == 0
    )
    summary = {
        "experiment_id": "selective_multimodal_visual_retrieval",
        "counts": {
            "total_visual_questions": len(rows),
            "answerable": sum(row["answerable"] for row in rows),
            "unanswerable": sum(
                not row["answerable"] for row in rows
            ),
            "visual_descriptions": len(visual_documents),
            "original_questions_routed": 0,
        },
        "control": control,
        "challenger": challenger,
        "question_level_improvements": improvements,
        "question_level_regressions": regressions,
        "correct_visual_anchors": correct_anchors,
        "answerable_visual_questions": sum(
            row["answerable"] for row in rows
        ),
        "promotion_gate_passed": promotion,
        "decision": (
            "PASS RETRIEVAL SCREEN; run paired answer generation"
            if promotion
            else "DO NOT PROMOTE; audit visual descriptions and anchors"
        ),
        "answer_generation_run": False,
    }
    write_jsonl(DETAIL_FILE, rows)
    write_json(SUMMARY_FILE, summary)

    print("\nMultimodal visual retrieval screening complete")
    print(
        "  control: "
        f"hit@3={control['hit@3']:.4f}, "
        f"recall@3={control['recall@3']:.4f}, "
        f"mrr@3={control['mrr@3']:.4f}"
    )
    print(
        "  challenger: "
        f"hit@3={challenger['hit@3']:.4f}, "
        f"recall@3={challenger['recall@3']:.4f}, "
        f"mrr@3={challenger['mrr@3']:.4f}"
    )
    print(f"  improvements: {improvements}")
    print(f"  regressions: {regressions}")
    print(
        f"  correct visual anchors: {correct_anchors}/"
        f"{sum(row['answerable'] for row in rows)}"
    )
    print(f"Decision: {summary['decision']}")
    print(f"Summary saved to: {SUMMARY_FILE.resolve()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build and screen selective vision descriptions against the "
            "frozen text-only RAG retrieval control."
        )
    )
    parser.add_argument(
        "--build-index",
        action="store_true",
        help=(
            "Detect figure pages, generate/reuse vision descriptions, "
            "and build the separate visual-description Chroma index."
        ),
    )
    parser.add_argument(
        "--force-descriptions",
        action="store_true",
        help=(
            "Regenerate every vision description. Valid only with "
            "--build-index."
        ),
    )
    parser.add_argument(
        "--force-pages",
        nargs="+",
        type=int,
        default=[],
        metavar="PAGE",
        help=(
            "Regenerate only the listed physical PDF figure pages and reuse "
            "all other descriptions. Valid only with --build-index."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.force_descriptions and not args.build_index:
        raise SystemExit(
            "--force-descriptions requires --build-index."
        )
    if args.force_pages and not args.build_index:
        raise SystemExit(
            "--force-pages requires --build-index."
        )
    if args.force_descriptions and args.force_pages:
        raise SystemExit(
            "Use either --force-descriptions or --force-pages, not both."
        )
    if args.build_index:
        build_visual_index(
            force_descriptions=args.force_descriptions,
            force_pages=set(args.force_pages),
        )
    else:
        run_retrieval_screen()


if __name__ == "__main__":
    main()
