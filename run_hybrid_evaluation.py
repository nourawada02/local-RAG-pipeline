from __future__ import annotations

import json
import math
import re
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from langchain_core.documents import Document
from langchain_ollama import ChatOllama

from build_index import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
)
from rag_query import (
    GENERATION_MODEL,
    generate_answer,
    stop_ollama_model,
)
from run_ocr_evaluation import (
    DATABASE_DIR,
    GOLD_FILE,
    RESULTS_DIR,
    SOURCE_TO_DOC_ID,
    extract_citations,
    is_abstention,
    open_evaluation_vector_store,
    read_jsonl,
    validate_inputs,
)


# Controlled experiment: retrieval strategy changes, while the OCR index,
# chunking, embedding model, generator, prompt, and final context size stay
# frozen.
FINAL_TOP_K = 3
DENSE_CANDIDATE_K = 20
BM25_CANDIDATE_K = 20
RRF_CONSTANT = 60
BM25_K1 = 1.5
BM25_B = 0.75

PREDICTIONS_FILE = RESULTS_DIR / "hybrid_predictions.jsonl"
CONFIG_FILE = RESULTS_DIR / "hybrid_run_config.json"

TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


def chunk_key(document: Document) -> tuple[str, int, int]:
    return (
        str(document.metadata.get("source", "")),
        int(document.metadata.get("page", -1)),
        int(document.metadata.get("chunk", -1)),
    )


def tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.casefold())


class BM25Index:
    """Small in-memory BM25 index over the frozen Chroma chunks."""

    def __init__(
        self,
        documents: list[Document],
        k1: float = BM25_K1,
        b: float = BM25_B,
    ) -> None:
        if not documents:
            raise ValueError("Cannot build BM25 over an empty corpus.")

        self.documents = documents
        self.k1 = k1
        self.b = b
        self.term_frequencies: list[Counter[str]] = []
        self.document_lengths: list[int] = []
        document_frequencies: Counter[str] = Counter()

        for document in documents:
            terms = tokenize(document.page_content)
            frequencies = Counter(terms)
            self.term_frequencies.append(frequencies)
            self.document_lengths.append(len(terms))
            document_frequencies.update(frequencies.keys())

        self.corpus_size = len(documents)
        self.average_document_length = (
            sum(self.document_lengths) / self.corpus_size
        )
        self.inverse_document_frequencies = {
            term: math.log(
                1.0
                + (
                    self.corpus_size
                    - frequency
                    + 0.5
                )
                / (frequency + 0.5)
            )
            for term, frequency in document_frequencies.items()
        }

    def search(
        self,
        query: str,
        k: int,
    ) -> list[tuple[Document, float]]:
        query_terms = set(tokenize(query))
        scores: list[tuple[float, int]] = []

        for index, frequencies in enumerate(self.term_frequencies):
            document_length = self.document_lengths[index]
            length_normalizer = self.k1 * (
                1.0
                - self.b
                + self.b
                * document_length
                / max(self.average_document_length, 1.0)
            )
            score = 0.0

            for term in query_terms:
                term_frequency = frequencies.get(term, 0)
                if term_frequency == 0:
                    continue
                inverse_document_frequency = (
                    self.inverse_document_frequencies.get(term, 0.0)
                )
                score += inverse_document_frequency * (
                    term_frequency * (self.k1 + 1.0)
                    / (term_frequency + length_normalizer)
                )

            if score > 0.0:
                scores.append((score, index))

        scores.sort(
            key=lambda item: (
                -item[0],
                chunk_key(self.documents[item[1]]),
            )
        )
        return [
            (self.documents[index], score)
            for score, index in scores[:k]
        ]


@dataclass
class FusedCandidate:
    document: Document
    rrf_score: float = 0.0
    dense_rank: int | None = None
    dense_distance: float | None = None
    bm25_rank: int | None = None
    bm25_score: float | None = None


def load_frozen_chunks(vector_store) -> list[Document]:
    collection = vector_store.get(include=["documents", "metadatas"])
    texts = collection.get("documents") or []
    metadatas = collection.get("metadatas") or []

    if len(texts) != len(metadatas):
        raise RuntimeError(
            "Chroma returned different document and metadata counts."
        )
    if not texts:
        raise RuntimeError("The frozen OCR index contains no chunks.")

    documents = [
        Document(page_content=str(text), metadata=dict(metadata or {}))
        for text, metadata in zip(texts, metadatas)
    ]

    keys = [chunk_key(document) for document in documents]
    if len(keys) != len(set(keys)):
        raise RuntimeError(
            "The frozen OCR index contains duplicate chunk metadata."
        )

    documents.sort(key=chunk_key)
    return documents


def reciprocal_rank_fusion(
    dense_results: list[tuple[Document, float]],
    bm25_results: list[tuple[Document, float]],
) -> list[FusedCandidate]:
    candidates: dict[tuple[str, int, int], FusedCandidate] = {}

    for rank, (document, distance) in enumerate(dense_results, start=1):
        key = chunk_key(document)
        candidate = candidates.setdefault(
            key,
            FusedCandidate(document=document),
        )
        candidate.dense_rank = rank
        candidate.dense_distance = float(distance)
        candidate.rrf_score += 1.0 / (RRF_CONSTANT + rank)

    for rank, (document, score) in enumerate(bm25_results, start=1):
        key = chunk_key(document)
        candidate = candidates.setdefault(
            key,
            FusedCandidate(document=document),
        )
        candidate.bm25_rank = rank
        candidate.bm25_score = float(score)
        candidate.rrf_score += 1.0 / (RRF_CONSTANT + rank)

    ranked = sorted(
        candidates.values(),
        key=lambda candidate: (
            -candidate.rrf_score,
            candidate.dense_rank
            if candidate.dense_rank is not None
            else math.inf,
            candidate.bm25_rank
            if candidate.bm25_rank is not None
            else math.inf,
            chunk_key(candidate.document),
        ),
    )
    return ranked[:FINAL_TOP_K]


def retrieve_hybrid(
    vector_store,
    bm25_index: BM25Index,
    question: str,
) -> tuple[
    list[tuple[Document, float]],
    list[FusedCandidate],
    int,
    int,
]:
    dense_results = vector_store.similarity_search_with_score(
        query=question,
        k=DENSE_CANDIDATE_K,
    )
    bm25_results = bm25_index.search(
        query=question,
        k=BM25_CANDIDATE_K,
    )

    fused = reciprocal_rank_fusion(dense_results, bm25_results)
    generation_results = [
        (candidate.document, -candidate.rrf_score)
        for candidate in fused
    ]
    return (
        generation_results,
        fused,
        len(dense_results),
        len(bm25_results),
    )


def normalize_hybrid_retrieved(
    fused: list[FusedCandidate],
) -> list[dict]:
    normalized: list[dict] = []

    for rank, candidate in enumerate(fused, start=1):
        document = candidate.document
        source = str(document.metadata.get("source", ""))
        if source not in SOURCE_TO_DOC_ID:
            raise ValueError(f"Unknown source filename in Chroma: {source}")

        normalized.append(
            {
                "rank": rank,
                "doc_id": SOURCE_TO_DOC_ID[source],
                "pages": [int(document.metadata["page"])],
                "chunk": int(document.metadata["chunk"]),
                "distance": (
                    float(candidate.dense_distance)
                    if candidate.dense_distance is not None
                    else None
                ),
                "hybrid_rrf_score": float(candidate.rrf_score),
                "dense_rank": candidate.dense_rank,
                "bm25_rank": candidate.bm25_rank,
                "bm25_score": candidate.bm25_score,
                "text": document.page_content.strip(),
            }
        )

    return normalized


def save_run_config(corpus_size: int) -> None:
    config = {
        "experiment_id": "ocr_hybrid_rrf_top_3",
        "independent_variable": "retrieval strategy",
        "control_experiment": "ocr_fallback_top_k_3",
        "reused_index": str(DATABASE_DIR),
        "collection_name": COLLECTION_NAME,
        "corpus_chunks": corpus_size,
        "chunking": "RecursiveCharacterTextSplitter",
        "chunk_size_characters": CHUNK_SIZE,
        "chunk_overlap_characters": CHUNK_OVERLAP,
        "embedding_model": EMBEDDING_MODEL,
        "retrieval": {
            "dense": {
                "method": "Chroma cosine similarity",
                "candidate_k": DENSE_CANDIDATE_K,
            },
            "lexical": {
                "method": "BM25",
                "candidate_k": BM25_CANDIDATE_K,
                "k1": BM25_K1,
                "b": BM25_B,
                "storage": "in memory from frozen OCR chunks",
            },
            "fusion": {
                "method": "equal-weight Reciprocal Rank Fusion",
                "rrf_constant": RRF_CONSTANT,
            },
            "final_top_k": FINAL_TOP_K,
        },
        "generation_model": GENERATION_MODEL,
        "temperature": 0.2,
        "num_ctx": 4096,
        "num_predict": 256,
        "reasoning": False,
        "ocr": {
            "enabled": True,
            "strategy": "reuse frozen selective-OCR index",
        },
        "predictions_file": str(PREDICTIONS_FILE),
    }
    CONFIG_FILE.write_text(
        json.dumps(config, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


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
    if not DATABASE_DIR.exists():
        raise FileNotFoundError(
            "The frozen OCR index was not found at "
            f"{DATABASE_DIR}. Do not rebuild it for this experiment."
        )

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
    frozen_chunks = load_frozen_chunks(vector_store)
    bm25_index = BM25Index(frozen_chunks)
    save_run_config(corpus_size=len(frozen_chunks))

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
    print("Controlled experiment: selective OCR with hybrid retrieval")
    print(f"Reusing frozen OCR index: {DATABASE_DIR.resolve()}")
    print(
        f"Hybrid retrieval: dense top {DENSE_CANDIDATE_K} + "
        f"BM25 top {BM25_CANDIDATE_K} -> RRF top {FINAL_TOP_K}"
    )
    print(f"BM25 corpus: {len(frozen_chunks)} frozen OCR chunks")
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
                (
                    results,
                    fused,
                    dense_count,
                    bm25_count,
                ) = retrieve_hybrid(
                    vector_store=vector_store,
                    bm25_index=bm25_index,
                    question=question,
                )
                retrieval_ms = (
                    time.perf_counter() - retrieval_start
                ) * 1000

                if dense_count != DENSE_CANDIDATE_K:
                    raise RuntimeError(
                        f"Expected {DENSE_CANDIDATE_K} dense candidates, "
                        f"but received {dense_count}."
                    )
                if len(results) != FINAL_TOP_K:
                    raise RuntimeError(
                        f"Expected {FINAL_TOP_K} fused chunks, "
                        f"but received {len(results)}."
                    )

                normalized_retrieved = normalize_hybrid_retrieved(fused)

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
                print(
                    f"Candidates: dense={dense_count}, bm25={bm25_count}; "
                    f"fused chunks={len(results)}"
                )
                print(
                    "Fused ranks: "
                    + ", ".join(
                        (
                            f"{rank}:dense={candidate.dense_rank or '-'},"
                            f"bm25={candidate.bm25_rank or '-'}"
                        )
                        for rank, candidate in enumerate(fused, start=1)
                    )
                )
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
    print(f"Run configuration saved to: {CONFIG_FILE.resolve()}")
    print("\nCalculate hybrid metrics with:")
    print(
        r"python rag-evaluation-starter\evaluation\metrics.py "
        r"--gold rag-evaluation-starter\evaluation\gold_questions.jsonl "
        r"--predictions "
        r"rag-evaluation-starter\results\hybrid_predictions.jsonl "
        r"--output rag-evaluation-starter\results\hybrid_metrics.json"
    )


def main() -> None:
    run_evaluation()


if __name__ == "__main__":
    main()
