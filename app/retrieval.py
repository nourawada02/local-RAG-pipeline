from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass


TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
BOTH_PATTERN = re.compile(
    r"^(?P<stem>.+?)\bboth\s+(?P<left>.+?)\s+and\s+"
    r"(?P<right>.+?)[?.!]*$",
    flags=re.IGNORECASE,
)


@dataclass(frozen=True)
class Chunk:
    text: str
    document_id: str
    filename: str
    page: int
    chunk: int
    extraction_method: str

    @property
    def key(self) -> tuple[str, int, int]:
        return (self.document_id, self.page, self.chunk)

    @property
    def page_key(self) -> tuple[str, int]:
        return (self.document_id, self.page)


@dataclass
class FusedCandidate:
    chunk: Chunk
    rrf_score: float = 0.0
    dense_rank: int | None = None
    dense_distance: float | None = None
    bm25_rank: int | None = None
    bm25_score: float | None = None


def tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.casefold())


def decompose_question(question: str) -> list[str]:
    """Split only explicit 'both X and Y' questions."""
    normalized = re.sub(r"\s+", " ", question).strip()
    match = BOTH_PATTERN.match(normalized)
    if match is None:
        return [normalized]

    stem = match.group("stem").strip()
    left = match.group("left").strip(" \t\r\n,;:.?!")
    right = match.group("right").strip(" \t\r\n,;:.?!")
    if not stem or len(left.split()) < 2 or len(right.split()) < 2:
        return [normalized]
    return [f"{stem} {left}?", f"{stem} {right}?"]


class BM25Index:
    def __init__(
        self,
        chunks: list[Chunk],
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        if not chunks:
            raise ValueError("Cannot build BM25 over an empty corpus.")
        self.chunks = chunks
        self.k1 = k1
        self.b = b
        self.term_frequencies: list[Counter[str]] = []
        self.document_lengths: list[int] = []
        document_frequencies: Counter[str] = Counter()

        for chunk in chunks:
            terms = tokenize(chunk.text)
            frequencies = Counter(terms)
            self.term_frequencies.append(frequencies)
            self.document_lengths.append(len(terms))
            document_frequencies.update(frequencies.keys())

        self.corpus_size = len(chunks)
        self.average_length = sum(self.document_lengths) / self.corpus_size
        self.idf = {
            term: math.log(
                1.0
                + (self.corpus_size - frequency + 0.5)
                / (frequency + 0.5)
            )
            for term, frequency in document_frequencies.items()
        }

    def search(self, query: str, k: int) -> list[tuple[Chunk, float]]:
        query_terms = set(tokenize(query))
        scored: list[tuple[float, int]] = []

        for index, frequencies in enumerate(self.term_frequencies):
            document_length = self.document_lengths[index]
            normalizer = self.k1 * (
                1.0
                - self.b
                + self.b * document_length / max(self.average_length, 1.0)
            )
            score = 0.0
            for term in query_terms:
                frequency = frequencies.get(term, 0)
                if not frequency:
                    continue
                score += self.idf.get(term, 0.0) * (
                    frequency * (self.k1 + 1.0)
                    / (frequency + normalizer)
                )
            if score > 0:
                scored.append((score, index))

        scored.sort(key=lambda item: (-item[0], self.chunks[item[1]].key))
        return [(self.chunks[index], score) for score, index in scored[:k]]


def reciprocal_rank_fusion(
    dense_results: list[tuple[Chunk, float]],
    bm25_results: list[tuple[Chunk, float]],
    constant: int = 60,
) -> list[FusedCandidate]:
    candidates: dict[tuple[str, int, int], FusedCandidate] = {}
    for rank, (chunk, distance) in enumerate(dense_results, start=1):
        candidate = candidates.setdefault(
            chunk.key, FusedCandidate(chunk=chunk)
        )
        candidate.dense_rank = rank
        candidate.dense_distance = float(distance)
        candidate.rrf_score += 1.0 / (constant + rank)

    for rank, (chunk, score) in enumerate(bm25_results, start=1):
        candidate = candidates.setdefault(
            chunk.key, FusedCandidate(chunk=chunk)
        )
        candidate.bm25_rank = rank
        candidate.bm25_score = float(score)
        candidate.rrf_score += 1.0 / (constant + rank)

    return sorted(
        candidates.values(),
        key=lambda candidate: (
            -candidate.rrf_score,
            candidate.dense_rank
            if candidate.dense_rank is not None
            else math.inf,
            candidate.bm25_rank
            if candidate.bm25_rank is not None
            else math.inf,
            candidate.chunk.key,
        ),
    )


def select_balanced(
    dense_results: list[tuple[Chunk, float]],
    bm25_results: list[tuple[Chunk, float]],
    fused_results: list[FusedCandidate],
    top_k: int = 3,
) -> list[FusedCandidate]:
    """Reserve dense and lexical anchors, then fill by fused rank."""
    by_key = {candidate.chunk.key: candidate for candidate in fused_results}
    selected: list[FusedCandidate] = []
    pages: set[tuple[str, int]] = set()

    def add(chunk: Chunk) -> None:
        if len(selected) >= top_k or chunk.page_key in pages:
            return
        selected.append(by_key[chunk.key])
        pages.add(chunk.page_key)

    for chunk, _ in dense_results:
        add(chunk)
        if selected:
            break
    for chunk, _ in bm25_results:
        previous_count = len(selected)
        add(chunk)
        if len(selected) > previous_count:
            break
    for candidate in fused_results:
        add(candidate.chunk)
        if len(selected) >= top_k:
            break
    return selected

