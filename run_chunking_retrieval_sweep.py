from __future__ import annotations

import gc
import hashlib
import json
import math
import shutil
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from build_index import COLLECTION_NAME, EMBEDDING_MODEL
from rag_query import GENERATION_MODEL, stop_ollama_model
from run_balanced_hybrid_evaluation import (
    normalize_balanced_retrieved,
    page_key,
)
from run_decomposed_hybrid_evaluation import (
    decompose_question,
    normalize_decomposed_retrieved,
    retrieve_with_decomposition,
)
from run_hybrid_evaluation import BM25Index, load_frozen_chunks
from run_ocr_evaluation import (
    CORPUS_DIR,
    GOLD_FILE,
    RESULTS_DIR,
    configure_tesseract,
    load_evaluation_pages,
    read_jsonl,
    validate_inputs,
)


# Controlled Part 2 experiment: only recursive chunk size and overlap change.
# OCR, embedding model, vector database, balanced hybrid retrieval, deterministic
# query decomposition, candidate counts, and final top-3 page selection remain
# fixed. Values are characters because RecursiveCharacterTextSplitter is used
# without a tokenizer-specific length function.
@dataclass(frozen=True)
class ChunkingConfig:
    name: str
    chunk_size_characters: int
    chunk_overlap_characters: int


CONFIGS = (
    ChunkingConfig("small_500_100", 500, 100),
    ChunkingConfig("control_1000_200", 1000, 200),
    ChunkingConfig("large_1500_300", 1500, 300),
)

SWEEP_DIR = RESULTS_DIR / "chunking_sweep"
PAGE_CACHE_FILE = SWEEP_DIR / "ocr_pages.jsonl"
PAGE_CACHE_MANIFEST_FILE = SWEEP_DIR / "ocr_pages_manifest.json"
SUMMARY_FILE = SWEEP_DIR / "chunking_retrieval_sweep_results.json"
EMBEDDING_BATCH_SIZE = 16
FINAL_TOP_K = 3


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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def corpus_fingerprints() -> dict[str, str]:
    return {
        path.name: sha256_file(path)
        for path in sorted(CORPUS_DIR.glob("*.pdf"))
    }


def page_cache_is_valid(fingerprints: dict[str, str]) -> bool:
    if not PAGE_CACHE_FILE.exists() or not PAGE_CACHE_MANIFEST_FILE.exists():
        return False
    try:
        manifest = json.loads(
            PAGE_CACHE_MANIFEST_FILE.read_text(encoding="utf-8")
        )
        rows = read_jsonl(PAGE_CACHE_FILE)
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    return (
        manifest.get("corpus_sha256") == fingerprints
        and manifest.get("page_count") == len(rows)
        and len(rows) > 0
    )


def load_or_create_page_cache() -> tuple[list[Document], dict]:
    fingerprints = corpus_fingerprints()
    if page_cache_is_valid(fingerprints):
        rows = read_jsonl(PAGE_CACHE_FILE)
        pages = [
            Document(
                page_content=str(row["text"]),
                metadata=dict(row["metadata"]),
            )
            for row in rows
        ]
        manifest = json.loads(
            PAGE_CACHE_MANIFEST_FILE.read_text(encoding="utf-8")
        )
        print(f"Reusing verified OCR page cache: {PAGE_CACHE_FILE.resolve()}")
        return pages, manifest

    print("Creating one shared OCR page cache for all chunking settings...")
    tesseract_version = configure_tesseract()
    pages, extraction_report = load_evaluation_pages()
    rows = [
        {
            "text": document.page_content,
            "metadata": dict(document.metadata),
        }
        for document in pages
    ]
    manifest = {
        "corpus_sha256": fingerprints,
        "page_count": len(pages),
        "extraction": extraction_report,
        "tesseract_version": tesseract_version,
    }
    write_jsonl(PAGE_CACHE_FILE, rows)
    write_json(PAGE_CACHE_MANIFEST_FILE, manifest)
    return pages, manifest


def paths_for(config: ChunkingConfig) -> dict[str, Path]:
    return {
        "database": SWEEP_DIR / f"{config.name}_chroma_db",
        "index_manifest": SWEEP_DIR / f"{config.name}_index_manifest.json",
        "predictions": SWEEP_DIR / f"{config.name}_retrieval.jsonl",
        "metrics": SWEEP_DIR / f"{config.name}_retrieval_metrics.json",
    }


def split_pages(
    pages: list[Document],
    config: ChunkingConfig,
) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.chunk_size_characters,
        chunk_overlap=config.chunk_overlap_characters,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(pages)
    for chunk_number, chunk in enumerate(chunks, start=1):
        chunk.metadata["chunk"] = chunk_number
        chunk.metadata["chunking_config"] = config.name
    return chunks


def expected_index_manifest(
    config: ChunkingConfig,
    fingerprints: dict[str, str],
    chunk_count: int,
) -> dict:
    return {
        "experiment": "chunking_retrieval_sweep",
        "independent_variable": "recursive chunk size and overlap",
        "chunking": "RecursiveCharacterTextSplitter",
        **asdict(config),
        "embedding_model": EMBEDDING_MODEL,
        "vector_database": "Chroma",
        "distance": "cosine",
        "collection_name": COLLECTION_NAME,
        "corpus_sha256": fingerprints,
        "chunk_count": chunk_count,
        "complete": True,
    }


def index_is_valid(
    config: ChunkingConfig,
    fingerprints: dict[str, str],
    chunk_count: int,
) -> bool:
    paths = paths_for(config)
    if (
        not paths["database"].exists()
        or not paths["index_manifest"].exists()
    ):
        return False
    try:
        actual = json.loads(
            paths["index_manifest"].read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError):
        return False
    expected = expected_index_manifest(
        config=config,
        fingerprints=fingerprints,
        chunk_count=chunk_count,
    )
    return all(actual.get(key) == value for key, value in expected.items())


def build_index(
    pages: list[Document],
    config: ChunkingConfig,
    fingerprints: dict[str, str],
) -> dict:
    paths = paths_for(config)
    chunks = split_pages(pages, config)
    average_chunk_characters = statistics.fmean(
        len(chunk.page_content) for chunk in chunks
    )

    if index_is_valid(config, fingerprints, len(chunks)):
        print(
            f"Reusing {config.name} index: {len(chunks)} chunks at "
            f"{paths['database'].resolve()}"
        )
        manifest = json.loads(
            paths["index_manifest"].read_text(encoding="utf-8")
        )
        manifest["average_chunk_characters"] = average_chunk_characters
        manifest["build_ms"] = None
        return manifest

    # These directories are owned only by this sweep. An incomplete or stale
    # experiment index is rebuilt instead of being mistaken for a valid one.
    if paths["database"].exists():
        shutil.rmtree(paths["database"])
    if paths["index_manifest"].exists():
        paths["index_manifest"].unlink()

    print(
        f"\nBuilding {config.name}: size={config.chunk_size_characters}, "
        f"overlap={config.chunk_overlap_characters}, chunks={len(chunks)}"
    )
    build_start = time.perf_counter()
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        collection_metadata={"hnsw:space": "cosine"},
        persist_directory=str(paths["database"]),
    )

    for batch_start in range(0, len(chunks), EMBEDDING_BATCH_SIZE):
        batch = chunks[batch_start : batch_start + EMBEDDING_BATCH_SIZE]
        ids = [
            f"{config.name}-chunk-{batch_start + offset + 1:06d}"
            for offset in range(len(batch))
        ]
        vector_store.add_documents(batch, ids=ids)
        completed = batch_start + len(batch)
        print(f"  Embedded {completed}/{len(chunks)} chunks")

    stored = vector_store.get(include=[])
    stored_count = len(stored.get("ids") or [])
    if stored_count != len(chunks):
        raise RuntimeError(
            f"{config.name} stored {stored_count} chunks; "
            f"expected {len(chunks)}."
        )

    build_ms = (time.perf_counter() - build_start) * 1000
    manifest = expected_index_manifest(
        config=config,
        fingerprints=fingerprints,
        chunk_count=len(chunks),
    )
    manifest["average_chunk_characters"] = average_chunk_characters
    manifest["build_ms"] = build_ms
    write_json(paths["index_manifest"], manifest)
    del vector_store
    del embeddings
    gc.collect()
    return manifest


def open_index(config: ChunkingConfig) -> Chroma:
    database = paths_for(config)["database"]
    if not database.exists():
        raise FileNotFoundError(f"Missing sweep index: {database}")
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=OllamaEmbeddings(model=EMBEDDING_MODEL),
        persist_directory=str(database),
    )


def normalize_retrieval(
    decomposed_selected,
    balanced_selected,
) -> list[dict]:
    if decomposed_selected is not None:
        return normalize_decomposed_retrieved(decomposed_selected)
    return normalize_balanced_retrieved(balanced_selected)


def run_retrieval(
    config: ChunkingConfig,
    gold_rows: list[dict],
) -> list[dict]:
    vector_store = open_index(config)
    chunks = load_frozen_chunks(vector_store)
    bm25_index = BM25Index(chunks)
    predictions: list[dict] = []

    print(
        f"\nEvaluating {config.name}: {len(chunks)} chunks, "
        f"{len(gold_rows)} questions"
    )
    for position, gold in enumerate(gold_rows, start=1):
        question = str(gold["question"])
        retrieval_start = time.perf_counter()
        (
            results,
            decomposed_selected,
            balanced_selected,
            subqueries,
            query_runs,
        ) = retrieve_with_decomposition(
            vector_store=vector_store,
            bm25_index=bm25_index,
            question=question,
        )
        retrieval_ms = (time.perf_counter() - retrieval_start) * 1000

        if len(results) != FINAL_TOP_K:
            raise RuntimeError(
                f"{config.name} selected {len(results)} chunks for "
                f"{gold['id']}; expected {FINAL_TOP_K}."
            )
        if len({page_key(document) for document, _ in results}) != FINAL_TOP_K:
            raise RuntimeError(
                f"{config.name} produced duplicate pages for {gold['id']}."
            )

        predictions.append(
            {
                "id": str(gold["id"]),
                "question": question,
                "retrieved": normalize_retrieval(
                    decomposed_selected=decomposed_selected,
                    balanced_selected=balanced_selected,
                ),
                "retrieval_ms": retrieval_ms,
                "decomposed": len(subqueries) > 1,
                "retrieval_queries": subqueries,
                "query_runs": [
                    {
                        "query_index": run.query_index,
                        "query": run.query,
                        "dense_candidates": run.dense_count,
                        "bm25_candidates": run.bm25_count,
                    }
                    for run in query_runs
                ],
            }
        )
        print(
            f"  [{position:02d}/{len(gold_rows)}] {gold['id']} "
            f"{retrieval_ms:.1f} ms"
        )

    del vector_store
    gc.collect()
    return predictions


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


def score_one(
    retrieved: list[dict],
    gold_evidence: list[dict],
) -> dict[str, float]:
    gold_units = evidence_units(gold_evidence)
    seen_gold: set[tuple[str, int]] = set()
    first_relevant_rank: int | None = None

    for rank, result in enumerate(retrieved[:FINAL_TOP_K], start=1):
        relevant = evidence_units([result]) & gold_units
        if relevant and first_relevant_rank is None:
            first_relevant_rank = rank
        seen_gold.update(relevant)

    return {
        "hit_at_3": float(bool(seen_gold)),
        "recall_at_3": len(seen_gold) / len(gold_units),
        "mrr_at_3": (
            0.0 if first_relevant_rank is None else 1.0 / first_relevant_rank
        ),
    }


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (
        ordered[upper] - ordered[lower]
    ) * (position - lower)


def aggregate_metrics(
    gold_rows: list[dict],
    predictions: list[dict],
    index_manifest: dict,
) -> dict:
    by_id = {row["id"]: row for row in predictions}
    expected_ids = {str(row["id"]) for row in gold_rows}
    if set(by_id) != expected_ids:
        raise ValueError("Retrieval prediction IDs do not match gold IDs.")

    metric_values = {
        "hit_at_3": [],
        "recall_at_3": [],
        "mrr_at_3": [],
    }
    by_type: dict[str, dict[str, list[float]]] = {}
    latencies: list[float] = []

    for gold in gold_rows:
        prediction = by_id[str(gold["id"])]
        latencies.append(float(prediction["retrieval_ms"]))
        if not bool(gold["answerable"]):
            continue

        scores = score_one(
            retrieved=prediction["retrieved"],
            gold_evidence=gold["gold_evidence"],
        )
        question_type = str(gold["type"])
        type_values = by_type.setdefault(
            question_type,
            {
                "hit_at_3": [],
                "recall_at_3": [],
                "mrr_at_3": [],
            },
        )
        for name, value in scores.items():
            metric_values[name].append(value)
            type_values[name].append(value)

    return {
        "config": {
            key: index_manifest[key]
            for key in (
                "name",
                "chunk_size_characters",
                "chunk_overlap_characters",
                "chunk_count",
                "average_chunk_characters",
            )
        },
        "retrieval": {
            name: statistics.fmean(values)
            for name, values in metric_values.items()
        },
        "by_question_type": {
            question_type: {
                name: statistics.fmean(values)
                for name, values in values_by_metric.items()
            }
            for question_type, values_by_metric in sorted(by_type.items())
        },
        "latency": {
            "median_ms": percentile(latencies, 0.5),
            "p95_ms": percentile(latencies, 0.95),
        },
        "index_build_ms": index_manifest.get("build_ms"),
        "answer_generation_performed": False,
    }


def main() -> None:
    validate_inputs()
    SWEEP_DIR.mkdir(parents=True, exist_ok=True)
    stop_ollama_model(GENERATION_MODEL)

    pages, page_manifest = load_or_create_page_cache()
    fingerprints = dict(page_manifest["corpus_sha256"])
    gold_rows = read_jsonl(GOLD_FILE)
    summaries: list[dict] = []

    try:
        for config in CONFIGS:
            index_manifest = build_index(
                pages=pages,
                config=config,
                fingerprints=fingerprints,
            )
            predictions = run_retrieval(config, gold_rows)
            metrics = aggregate_metrics(
                gold_rows=gold_rows,
                predictions=predictions,
                index_manifest=index_manifest,
            )
            paths = paths_for(config)
            write_jsonl(paths["predictions"], predictions)
            write_json(paths["metrics"], metrics)
            summaries.append(metrics)
    finally:
        stop_ollama_model(EMBEDDING_MODEL)

    ranked = sorted(
        summaries,
        key=lambda item: (
            -item["retrieval"]["recall_at_3"],
            -item["retrieval"]["hit_at_3"],
            -item["retrieval"]["mrr_at_3"],
            item["config"]["chunk_count"],
        ),
    )
    result = {
        "experiment": "Part 2 controlled chunking retrieval sweep",
        "independent_variable": "recursive chunk size and overlap",
        "controls": {
            "ocr": "native extraction with selective Tesseract fallback",
            "embedding_model": EMBEDDING_MODEL,
            "vector_database": "Chroma cosine",
            "retrieval": (
                "balanced dense plus BM25 hybrid with deterministic "
                "both-and query decomposition"
            ),
            "final_top_k": FINAL_TOP_K,
            "generation": "not run during screening",
        },
        "ranking_rule": (
            "recall@3, then hit@3, then MRR@3, then fewer chunks"
        ),
        "results": summaries,
        "provisional_winner": ranked[0]["config"]["name"],
        "next_step": (
            "Run full answer generation only for the provisional winner and "
            "compare it with the frozen decomposed-hybrid control."
        ),
    }
    write_json(SUMMARY_FILE, result)

    print("\nChunking retrieval sweep complete")
    for item in ranked:
        retrieval = item["retrieval"]
        config = item["config"]
        print(
            f"  {config['name']}: chunks={config['chunk_count']}, "
            f"hit@3={retrieval['hit_at_3']:.4f}, "
            f"recall@3={retrieval['recall_at_3']:.4f}, "
            f"mrr@3={retrieval['mrr_at_3']:.4f}"
        )
    print(f"Provisional winner: {ranked[0]['config']['name']}")
    print(f"Summary saved to: {SUMMARY_FILE.resolve()}")


if __name__ == "__main__":
    main()
