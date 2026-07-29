from __future__ import annotations

import hashlib
import json
import shutil
import urllib.request
from pathlib import Path

import fitz


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "data" / "corpus_manifest.json"
CORPUS_DIR = ROOT / "data" / "corpus"
REFERENCE_DIR = ROOT / "data" / "reference_sources"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "rag-evaluation-dataset/1.0"},
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        with destination.open("wb") as output:
            shutil.copyfileobj(response, output)


def rasterize_without_text_layer(source: Path, destination: Path, dpi: int = 170) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    source_pdf = fitz.open(source)
    scanned_pdf = fitz.open()
    scale = dpi / 72
    matrix = fitz.Matrix(scale, scale)

    for source_page in source_pdf:
        pixmap = source_page.get_pixmap(matrix=matrix, colorspace=fitz.csGRAY, alpha=False)
        target_page = scanned_pdf.new_page(width=source_page.rect.width, height=source_page.rect.height)
        target_page.insert_image(target_page.rect, stream=pixmap.tobytes("png"))

    scanned_pdf.save(destination, garbage=4, deflate=True)
    scanned_pdf.close()
    source_pdf.close()


def verify_pdf(path: Path, expected_pages: int, text_expected: bool) -> None:
    document = fitz.open(path)
    if len(document) != expected_pages:
        raise ValueError(f"{path.name}: expected {expected_pages} pages, found {len(document)}")
    extracted_characters = sum(len(page.get_text().strip()) for page in document)
    document.close()
    if text_expected and extracted_characters < 1000:
        raise ValueError(f"{path.name}: unexpectedly little native text")
    if not text_expected and extracted_characters != 0:
        raise ValueError(f"{path.name}: rasterized PDF still contains a text layer")


def main() -> None:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    REFERENCE_DIR.mkdir(parents=True, exist_ok=True)
    output_records = []

    for document in manifest["indexed_documents"]:
        destination = CORPUS_DIR / document["filename"]
        if document["extraction_expected"] == "native":
            download(document["source_url"], destination)
            verify_pdf(destination, document["expected_pages"], text_expected=True)
        else:
            clean_source = REFERENCE_DIR / "nist_sp_800_218a_clean_reference.pdf"
            download(document["source_url"], clean_source)
            verify_pdf(clean_source, document["expected_pages"], text_expected=True)
            if not destination.exists():
                rasterize_without_text_layer(clean_source, destination)
            verify_pdf(destination, document["expected_pages"], text_expected=False)

        output_records.append(
            {
                "doc_id": document["doc_id"],
                "filename": document["filename"],
                "pages": document["expected_pages"],
                "sha256": sha256(destination),
                "extraction_expected": document["extraction_expected"],
            }
        )
        print(f"Ready: {destination.name}")

    generated_manifest = {
        "dataset_name": manifest["dataset_name"],
        "version": manifest["version"],
        "documents": output_records,
    }
    generated_path = ROOT / "data" / "corpus_manifest.generated.json"
    generated_path.write_text(
        json.dumps(generated_manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Generated manifest: {generated_path}")
    print("Index only files inside data/corpus/.")


if __name__ == "__main__":
    main()
