from __future__ import annotations

import argparse
import json
import re
import shutil
import time
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

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
DATABASE_DIR = RESULTS_DIR / "baseline_chroma_db"
PREDICTIONS_FILE = RESULTS_DIR / "baseline_predictions.jsonl"
CONFIG_FILE = RESULTS_DIR / "baseline_run_config.json"

ABSTENTION_PREFIX = "I do not have enough information in the retrieved documents"

SOURCE_TO_DOC_ID = {
    "nist_ai_rmf_1_0.pdf": "nist_ai_rmf_1_0",
    "nist_ai_600_1_genai_profile.pdf": "nist_ai_600_1",
    "nist_sp_800_218a_scanned.pdf": "nist_sp_800_218a",
}


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


def load_evaluation_pages() -> list[Document]:
    documents: list[Document] = []

    for pdf_path in sorted(CORPUS_DIR.glob("*.pdf")):
        print(f"Reading: {pdf_path.name}")
        reader = PdfReader(str(pdf_path))
        readable_pages = 0

        for page_number, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()

            if not text:
                print(f"  Skipping empty page {page_number}")
                continue

            documents.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": pdf_path.name,
                        "page": page_number,
                    },
                )
            )
            readable_pages += 1

        print(
            f"  Extracted {readable_pages} readable pages "
            f"out of {len(reader.pages)}"
        )

    if not documents:
        raise ValueError("No readable text was extracted from the corpus.")

    return documents


def build_baseline_index() -> None:
    """Build a separate index using the frozen baseline settings."""
    validate_inputs()
    pages = load_evaluation_pages()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(pages)

    for chunk_number, chunk in enumerate(chunks, start=1):
        chunk.metadata["chunk"] = chunk_number

    print(f"\nReadable pages: {len(pages)}")
    print(f"Created chunks: {len(chunks)}")
    print(f"Chunk size: {CHUNK_SIZE} characters")
    print(f"Chunk overlap: {CHUNK_OVERLAP} characters")

    if DATABASE_DIR.exists():
        print(f"\nRemoving old evaluation database: {DATABASE_DIR}")
        shutil.rmtree(DATABASE_DIR)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)

    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        collection_metadata={"hnsw:space": "cosine"},
        persist_directory=str(DATABASE_DIR),
    )

    stop_ollama_model(EMBEDDING_MODEL)
    print(f"\nBaseline evaluation index saved to: {DATABASE_DIR.resolve()}")
    print("Your original documents/ and chroma_db/ were not changed.")


def open_evaluation_vector_store() -> Chroma:
    if not DATABASE_DIR.exists():
        raise FileNotFoundError(
            "The baseline evaluation index does not exist. First run:\n"
            "python run_baseline_evaluation.py --build-index"
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


def save_run_config() -> None:
    config = {
        "experiment_id": "baseline",
        "corpus": str(CORPUS_DIR),
        "collection_name": COLLECTION_NAME,
        "chunking": "RecursiveCharacterTextSplitter",
        "chunk_size_characters": CHUNK_SIZE,
        "chunk_overlap_characters": CHUNK_OVERLAP,
        "embedding_model": EMBEDDING_MODEL,
        "retrieval": "dense cosine similarity",
        "top_k": TOP_K,
        "generation_model": GENERATION_MODEL,
        "temperature": 0.2,
        "num_ctx": 4096,
        "num_predict": 256,
        "reasoning": False,
        "ocr": False,
    }
    CONFIG_FILE.write_text(
        json.dumps(config, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def run_evaluation() -> None:
    validate_inputs()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    save_run_config()

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
                    "abstained": ABSTENTION_PREFIX.casefold() in answer.casefold(),
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
    print("\nCalculate metrics with:")
    print(
        r"python rag-evaluation-starter\evaluation\metrics.py "
        r"--gold rag-evaluation-starter\evaluation\gold_questions.jsonl "
        r"--predictions "
        r"rag-evaluation-starter\results\baseline_predictions.jsonl "
        r"--output rag-evaluation-starter\results\baseline_metrics.json"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build and measure the frozen local-RAG baseline."
    )
    parser.add_argument(
        "--build-index",
        action="store_true",
        help="Build a separate Chroma index for the evaluation corpus.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.build_index:
        build_baseline_index()
    else:
        run_evaluation()


if __name__ == "__main__":
    main()
