from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import time
from pathlib import Path

import pymupdf
import pytesseract
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from PIL import Image
from pypdf import PdfReader
from pytesseract import TesseractNotFoundError

from build_index import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
)
from rag_query import (
    GENERATION_MODEL,
    TOP_K,
    generate_answer,
    retrieve_chunks,
    stop_ollama_model,
)


EVALUATION_DIR = Path("rag-evaluation-starter")
CORPUS_DIR = EVALUATION_DIR / "data" / "corpus"
GOLD_FILE = EVALUATION_DIR / "evaluation" / "gold_questions.jsonl"
RESULTS_DIR = EVALUATION_DIR / "results"

# This experiment has separate artifacts. It never overwrites the baseline.
DATABASE_DIR = RESULTS_DIR / "ocr_chroma_db"
PREDICTIONS_FILE = RESULTS_DIR / "ocr_predictions.jsonl"
CONFIG_FILE = RESULTS_DIR / "ocr_run_config.json"
EXTRACTION_REPORT_FILE = RESULTS_DIR / "ocr_extraction_report.json"

ABSTENTION_MARKERS = (
    "I do not have enough information in the retrieved documents",
    "there is no information regarding",
)

OCR_DPI = 300
OCR_LANGUAGE = "eng"
OCR_CONFIG = "--oem 1 --psm 3"
OCR_TIMEOUT_SECONDS = 180
EMBEDDING_BATCH_SIZE = 16

SOURCE_TO_DOC_ID = {
    "nist_ai_rmf_1_0.pdf": "nist_ai_rmf_1_0",
    "nist_ai_600_1_genai_profile.pdf": "nist_ai_600_1",
    "nist_sp_800_218a_scanned.pdf": "nist_sp_800_218a_scan",
}


def is_abstention(answer: str) -> bool:
    normalized_answer = answer.casefold()
    return any(marker.casefold() in normalized_answer for marker in ABSTENTION_MARKERS)


def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{path}:{line_number}: invalid JSON: {exc}"
                ) from exc
    return rows


def validate_inputs() -> None:
    if not CORPUS_DIR.exists():
        raise FileNotFoundError(f"Corpus folder not found: {CORPUS_DIR}")
    if not GOLD_FILE.exists():
        raise FileNotFoundError(f"Gold question file not found: {GOLD_FILE}")

    actual_files = {path.name for path in CORPUS_DIR.glob("*.pdf")}
    expected_files = set(SOURCE_TO_DOC_ID)
    if actual_files != expected_files:
        raise ValueError(
            "The evaluation corpus must contain exactly the expected three "
            f"PDFs. Expected={sorted(expected_files)}, "
            f"found={sorted(actual_files)}"
        )


def configure_tesseract() -> str:
    """Find Tesseract on Windows or PATH and verify English OCR support."""
    candidates: list[Path] = []

    configured_command = os.environ.get("TESSERACT_CMD")
    if configured_command:
        configured_path = Path(configured_command)
        if configured_path.is_dir():
            configured_path = configured_path / "tesseract.exe"
        candidates.append(configured_path)

    command_on_path = shutil.which("tesseract")
    if command_on_path:
        candidates.append(Path(command_on_path))

    candidates.extend(
        [
            Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe"),
            Path(r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"),
        ]
    )

    local_app_data = os.environ.get("LOCALAPPDATA")
    if local_app_data:
        candidates.append(
            Path(local_app_data)
            / "Programs"
            / "Tesseract-OCR"
            / "tesseract.exe"
        )

    for candidate in candidates:
        if candidate.is_file():
            pytesseract.pytesseract.tesseract_cmd = str(candidate)
            break

    try:
        version = str(pytesseract.get_tesseract_version())
        languages = set(pytesseract.get_languages(config=""))
    except TesseractNotFoundError as exc:
        raise RuntimeError(
            "Tesseract OCR was not found. Install Tesseract, reopen the "
            "terminal, and run: tesseract --version"
        ) from exc

    if OCR_LANGUAGE not in languages:
        raise RuntimeError(
            f"Tesseract is installed, but language '{OCR_LANGUAGE}' is "
            f"missing. Available languages: {sorted(languages)}"
        )

    print(f"Tesseract version: {version}")
    print(f"OCR language: {OCR_LANGUAGE}")
    print(f"OCR render resolution: {OCR_DPI} DPI")
    return version


def ocr_page(page: pymupdf.Page) -> str:
    """Render one PDF page in memory and recognize its text."""
    pixmap = page.get_pixmap(
        dpi=OCR_DPI,
        colorspace=pymupdf.csGRAY,
        alpha=False,
    )
    image = Image.frombytes(
        "L",
        (pixmap.width, pixmap.height),
        pixmap.samples,
    )
    return pytesseract.image_to_string(
        image,
        lang=OCR_LANGUAGE,
        config=OCR_CONFIG,
        timeout=OCR_TIMEOUT_SECONDS,
    ).strip()


def load_evaluation_pages() -> tuple[list[Document], dict]:
    """
    Use native extraction first and OCR only when a page has no native text.

    This preserves the baseline text for readable PDFs while making image-only
    pages searchable.
    """
    documents: list[Document] = []
    report: dict = {
        "strategy": "native text with selective OCR fallback",
        "ocr_engine": "Tesseract",
        "ocr_dpi": OCR_DPI,
        "ocr_language": OCR_LANGUAGE,
        "documents": {},
    }
    failed_ocr_pages: list[str] = []

    for pdf_path in sorted(CORPUS_DIR.glob("*.pdf")):
        print(f"Reading: {pdf_path.name}")
        reader = PdfReader(str(pdf_path))

        native_pages = 0
        ocr_pages = 0
        document_report = {
            "total_pages": len(reader.pages),
            "native_pages": 0,
            "ocr_attempted_pages": 0,
            "ocr_successful_pages": 0,
        }

        with pymupdf.open(str(pdf_path)) as rendered_pdf:
            if rendered_pdf.page_count != len(reader.pages):
                raise ValueError(
                    f"Page-count mismatch for {pdf_path.name}: "
                    f"pypdf={len(reader.pages)}, "
                    f"PyMuPDF={rendered_pdf.page_count}"
                )

            for page_number, page in enumerate(reader.pages, start=1):
                text = (page.extract_text() or "").strip()
                extraction_method = "native"

                if text:
                    native_pages += 1
                    document_report["native_pages"] += 1
                else:
                    document_report["ocr_attempted_pages"] += 1
                    print(f"  OCR fallback on page {page_number}")
                    text = ocr_page(rendered_pdf.load_page(page_number - 1))
                    extraction_method = "ocr"

                    if not text:
                        failed_ocr_pages.append(
                            f"{pdf_path.name}, page {page_number}"
                        )
                        print(
                            f"  WARNING: OCR returned no text on page "
                            f"{page_number}"
                        )
                        continue

                    ocr_pages += 1
                    document_report["ocr_successful_pages"] += 1

                metadata = {
                    "source": pdf_path.name,
                    "page": page_number,
                    "extraction_method": extraction_method,
                }
                if extraction_method == "ocr":
                    metadata["ocr_engine"] = "tesseract"

                documents.append(
                    Document(
                        page_content=text,
                        metadata=metadata,
                    )
                )

        report["documents"][pdf_path.name] = document_report
        print(
            f"  Loaded {native_pages + ocr_pages} of {len(reader.pages)} "
            f"pages: native={native_pages}, OCR={ocr_pages}"
        )

    if failed_ocr_pages:
        raise RuntimeError(
            "OCR produced empty text for pages that had no native text: "
            + "; ".join(failed_ocr_pages)
        )

    if not documents:
        raise ValueError("No readable text was extracted from the corpus.")

    report["total_loaded_pages"] = len(documents)
    report["total_native_pages"] = sum(
        item["native_pages"] for item in report["documents"].values()
    )
    report["total_ocr_pages"] = sum(
        item["ocr_successful_pages"]
        for item in report["documents"].values()
    )
    return documents, report


def save_run_config(tesseract_version: str) -> None:
    config = {
        "experiment_id": "ocr_fallback",
        "independent_variable": "document extraction",
        "corpus": str(CORPUS_DIR),
        "collection_name": COLLECTION_NAME,
        "chunking": "RecursiveCharacterTextSplitter",
        "chunk_size_characters": CHUNK_SIZE,
        "chunk_overlap_characters": CHUNK_OVERLAP,
        "embedding_model": EMBEDDING_MODEL,
        "embedding_batch_size": EMBEDDING_BATCH_SIZE,
        "retrieval": "dense cosine similarity",
        "top_k": TOP_K,
        "generation_model": GENERATION_MODEL,
        "temperature": 0.2,
        "num_ctx": 4096,
        "num_predict": 256,
        "reasoning": False,
        "ocr": {
            "enabled": True,
            "strategy": "fallback only when native extraction is empty",
            "engine": "Tesseract",
            "engine_version": tesseract_version,
            "language": OCR_LANGUAGE,
            "render_dpi": OCR_DPI,
            "config": OCR_CONFIG,
        },
    }
    CONFIG_FILE.write_text(
        json.dumps(config, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def build_ocr_index() -> None:
    """Build an OCR-enabled index while keeping baseline settings frozen."""
    validate_inputs()
    tesseract_version = configure_tesseract()
    pages, extraction_report = load_evaluation_pages()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(pages)

    for chunk_number, chunk in enumerate(chunks, start=1):
        chunk.metadata["chunk"] = chunk_number

    print(f"\nLoaded pages: {len(pages)}")
    print(f"Native pages: {extraction_report['total_native_pages']}")
    print(f"OCR pages: {extraction_report['total_ocr_pages']}")
    print(f"Created chunks: {len(chunks)}")
    print(f"Chunk size: {CHUNK_SIZE} characters")
    print(f"Chunk overlap: {CHUNK_OVERLAP} characters")

    if DATABASE_DIR.exists():
        print(f"\nRemoving old OCR experiment database: {DATABASE_DIR}")
        shutil.rmtree(DATABASE_DIR)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    save_run_config(tesseract_version)
    EXTRACTION_REPORT_FILE.write_text(
        json.dumps(extraction_report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=OllamaEmbeddings(model=EMBEDDING_MODEL),
        collection_metadata={"hnsw:space": "cosine"},
        persist_directory=str(DATABASE_DIR),
    )

    try:
        total_batches = (
            len(chunks) + EMBEDDING_BATCH_SIZE - 1
        ) // EMBEDDING_BATCH_SIZE
        for batch_number, start in enumerate(
            range(0, len(chunks), EMBEDDING_BATCH_SIZE),
            start=1,
        ):
            batch = chunks[start : start + EMBEDDING_BATCH_SIZE]
            vector_store.add_documents(batch)
            print(
                f"  Embedded batch {batch_number}/{total_batches} "
                f"({min(start + len(batch), len(chunks))}/{len(chunks)} chunks)"
            )
    finally:
        stop_ollama_model(EMBEDDING_MODEL)

    print(f"\nOCR evaluation index saved to: {DATABASE_DIR.resolve()}")
    print(f"Extraction report saved to: {EXTRACTION_REPORT_FILE.resolve()}")
    print("The baseline database and baseline results were not changed.")


def open_evaluation_vector_store() -> Chroma:
    if not DATABASE_DIR.exists():
        raise FileNotFoundError(
            "The OCR evaluation index does not exist. First run:\n"
            "python run_ocr_evaluation.py --build-index"
        )

    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=OllamaEmbeddings(model=EMBEDDING_MODEL),
        persist_directory=str(DATABASE_DIR),
    )


def normalize_retrieved(
    results: list[tuple[Document, float]],
) -> list[dict]:
    normalized: list[dict] = []

    for rank, (document, distance) in enumerate(results, start=1):
        source = str(document.metadata.get("source", ""))
        if source not in SOURCE_TO_DOC_ID:
            raise ValueError(f"Unknown source filename in Chroma: {source}")

        normalized.append(
            {
                "rank": rank,
                "doc_id": SOURCE_TO_DOC_ID[source],
                "pages": [int(document.metadata["page"])],
                "chunk": int(document.metadata["chunk"]),
                "distance": float(distance),
                "text": document.page_content.strip(),
            }
        )

    return normalized


def extract_citations(
    answer: str,
    retrieved: list[dict],
) -> list[dict]:
    """Map generated [Source N] labels back to document-page citations."""
    cited_ranks: set[int] = set()
    labels = re.findall(
        r"\[([^\]]*Sources?[^\]]*)\]",
        answer,
        flags=re.IGNORECASE,
    )
    for label in labels:
        cited_ranks.update(int(value) for value in re.findall(r"\d+", label))

    citations: list[dict] = []
    seen: set[tuple[str, int]] = set()

    for rank in sorted(cited_ranks):
        if not 1 <= rank <= len(retrieved):
            continue
        result = retrieved[rank - 1]
        for page in result["pages"]:
            unit = (result["doc_id"], int(page))
            if unit in seen:
                continue
            seen.add(unit)
            citations.append(
                {
                    "doc_id": result["doc_id"],
                    "pages": [int(page)],
                }
            )

    return citations


def load_completed_predictions() -> dict[str, dict]:
    if not PREDICTIONS_FILE.exists():
        return {}

    rows = read_jsonl(PREDICTIONS_FILE)
    completed: dict[str, dict] = {}
    for row in rows:
        question_id = str(row["id"])
        if question_id in completed:
            raise ValueError(
                f"Duplicate prediction ID in {PREDICTIONS_FILE}: "
                f"{question_id}"
            )
        completed[question_id] = row
    return completed


def run_evaluation() -> None:
    validate_inputs()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    gold_rows = read_jsonl(GOLD_FILE)
    completed = load_completed_predictions()
    gold_ids = {str(row["id"]) for row in gold_rows}
    extra_ids = sorted(set(completed) - gold_ids)
    if extra_ids:
        raise ValueError(
            f"Predictions contain IDs not present in the gold set: {extra_ids}"
        )

    vector_store = open_evaluation_vector_store()
    llm = ChatOllama(
        model=GENERATION_MODEL,
        temperature=0.2,
        num_ctx=4096,
        num_predict=256,
        reasoning=False,
    )

    pending = [
        row for row in gold_rows if str(row["id"]) not in completed
    ]
    print(
        f"Gold questions: {len(gold_rows)} | "
        f"already completed: {len(completed)} | pending: {len(pending)}"
    )

    try:
        with PREDICTIONS_FILE.open("a", encoding="utf-8") as output:
            for position, gold in enumerate(pending, start=1):
                question_id = str(gold["id"])
                question = str(gold["question"])
                print(
                    f"\n[{position}/{len(pending)}] "
                    f"{question_id}: {question}"
                )

                total_start = time.perf_counter()

                stop_ollama_model(GENERATION_MODEL)
                retrieval_start = time.perf_counter()
                results = retrieve_chunks(
                    vector_store=vector_store,
                    question=question,
                )
                retrieval_ms = (
                    time.perf_counter() - retrieval_start
                ) * 1000

                normalized_retrieved = normalize_retrieved(results)

                stop_ollama_model(EMBEDDING_MODEL)
                generation_start = time.perf_counter()
                answer = generate_answer(
                    llm=llm,
                    question=question,
                    results=results,
                )
                generation_ms = (
                    time.perf_counter() - generation_start
                ) * 1000
                total_ms = (time.perf_counter() - total_start) * 1000

                prediction = {
                    "id": question_id,
                    "answer": answer,
                    "abstained": is_abstention(answer),
                    "retrieved": normalized_retrieved,
                    "citations": extract_citations(
                        answer,
                        normalized_retrieved,
                    ),
                    "retrieval_ms": round(retrieval_ms, 3),
                    "generation_ms": round(generation_ms, 3),
                    "total_ms": round(total_ms, 3),
                }

                output.write(
                    json.dumps(prediction, ensure_ascii=False) + "\n"
                )
                output.flush()
                print(f"Answer: {answer}")
                print(
                    f"Latency: retrieval={retrieval_ms:.0f} ms, "
                    f"generation={generation_ms:.0f} ms, "
                    f"total={total_ms:.0f} ms"
                )
                stop_ollama_model(GENERATION_MODEL)
    finally:
        stop_ollama_model(EMBEDDING_MODEL)
        stop_ollama_model(GENERATION_MODEL)

    final_rows = load_completed_predictions()
    if len(final_rows) != len(gold_rows):
        raise RuntimeError(
            f"Only {len(final_rows)} of {len(gold_rows)} predictions exist."
        )

    print(f"\nCompleted all {len(gold_rows)} questions.")
    print(f"Predictions saved to: {PREDICTIONS_FILE.resolve()}")
    print("\nCalculate OCR metrics with:")
    print(
        r"python rag-evaluation-starter\evaluation\metrics.py "
        r"--gold rag-evaluation-starter\evaluation\gold_questions.jsonl "
        r"--predictions "
        r"rag-evaluation-starter\results\ocr_predictions.jsonl "
        r"--output rag-evaluation-starter\results\ocr_metrics.json"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build and measure the selective-OCR local-RAG experiment."
        )
    )
    parser.add_argument(
        "--build-index",
        action="store_true",
        help="Build a separate OCR-enabled Chroma evaluation index.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.build_index:
        build_ocr_index()
    else:
        run_evaluation()


if __name__ == "__main__":
    main()
