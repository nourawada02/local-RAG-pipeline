from __future__ import annotations

import io
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

import pymupdf
import pytesseract
from docx import Document as WordDocument
from PIL import Image
from pypdf import PdfReader
from pytesseract import TesseractNotFoundError


TEXT_EXTENSIONS = {
    ".txt",
    ".md",
    ".csv",
    ".json",
    ".html",
    ".htm",
    ".xml",
    ".rst",
}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
SUPPORTED_EXTENSIONS = TEXT_EXTENSIONS | IMAGE_EXTENSIONS | {".pdf", ".docx"}


class DocumentError(ValueError):
    """Raised when an uploaded document cannot be safely ingested."""


@dataclass(frozen=True)
class ExtractedPage:
    text: str
    page: int
    extraction_method: str


def configure_tesseract() -> None:
    candidates: list[Path] = []
    configured = os.getenv("TESSERACT_CMD")
    if configured:
        configured_path = Path(configured)
        if configured_path.is_dir():
            configured_path = configured_path / "tesseract.exe"
        candidates.append(configured_path)
    on_path = shutil.which("tesseract")
    if on_path:
        candidates.append(Path(on_path))
    candidates.extend(
        [
            Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe"),
            Path(r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"),
        ]
    )
    local_app_data = os.getenv("LOCALAPPDATA")
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
        pytesseract.get_tesseract_version()
    except TesseractNotFoundError as exc:
        raise DocumentError(
            "This document requires OCR, but Tesseract was not found. "
            "Install Tesseract or set TESSERACT_CMD."
        ) from exc


def _ocr_image(image: Image.Image, language: str) -> str:
    configure_tesseract()
    return pytesseract.image_to_string(
        image,
        lang=language,
        config="--oem 1 --psm 3",
        timeout=180,
    ).strip()


def _extract_pdf(data: bytes, dpi: int, language: str) -> list[ExtractedPage]:
    try:
        reader = PdfReader(io.BytesIO(data))
        rendered_pdf = pymupdf.open(stream=data, filetype="pdf")
    except Exception as exc:
        raise DocumentError(f"The uploaded PDF could not be opened: {exc}") from exc

    pages: list[ExtractedPage] = []
    empty_after_ocr: list[int] = []
    try:
        if len(reader.pages) != rendered_pdf.page_count:
            raise DocumentError("PDF readers reported different page counts.")
        for page_number, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()
            method = "native"
            if not text:
                pixmap = rendered_pdf.load_page(page_number - 1).get_pixmap(
                    dpi=dpi,
                    colorspace=pymupdf.csGRAY,
                    alpha=False,
                )
                image = Image.frombytes(
                    "L", (pixmap.width, pixmap.height), pixmap.samples
                )
                text = _ocr_image(image, language)
                method = "ocr"
            if not text:
                empty_after_ocr.append(page_number)
                continue
            pages.append(ExtractedPage(text, page_number, method))
    finally:
        rendered_pdf.close()

    if empty_after_ocr:
        page_list = ", ".join(str(page) for page in empty_after_ocr)
        raise DocumentError(
            "Text extraction and OCR both returned empty text for PDF "
            f"page(s): {page_list}. The document was not indexed."
        )
    return pages


def extract_document(
    filename: str,
    data: bytes,
    ocr_dpi: int,
    ocr_language: str,
) -> list[ExtractedPage]:
    extension = Path(filename).suffix.casefold()
    if extension not in SUPPORTED_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise DocumentError(
            f"Unsupported file type '{extension or 'none'}'. "
            f"Supported types: {supported}."
        )
    if extension == ".pdf":
        pages = _extract_pdf(data, ocr_dpi, ocr_language)
    elif extension in IMAGE_EXTENSIONS:
        try:
            image = Image.open(io.BytesIO(data))
            text = _ocr_image(image, ocr_language)
        except DocumentError:
            raise
        except Exception as exc:
            raise DocumentError(f"The uploaded image could not be opened: {exc}") from exc
        pages = [ExtractedPage(text, 1, "ocr")] if text else []
    elif extension == ".docx":
        try:
            document = WordDocument(io.BytesIO(data))
            text = "\n".join(
                paragraph.text for paragraph in document.paragraphs
                if paragraph.text.strip()
            ).strip()
        except Exception as exc:
            raise DocumentError(f"The DOCX file could not be opened: {exc}") from exc
        pages = [ExtractedPage(text, 1, "docx")] if text else []
    else:
        try:
            text = data.decode("utf-8-sig").strip()
        except UnicodeDecodeError as exc:
            raise DocumentError("Text files must use UTF-8 encoding.") from exc
        pages = [ExtractedPage(text, 1, "native")] if text else []

    if not pages:
        raise DocumentError("The uploaded document contains no readable text.")
    return pages

