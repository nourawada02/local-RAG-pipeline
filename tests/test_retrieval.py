from __future__ import annotations

import unittest

from app.retrieval import (
    BM25Index,
    Chunk,
    decompose_question,
    reciprocal_rank_fusion,
    select_balanced,
)


def make_chunk(number: int, page: int, text: str) -> Chunk:
    return Chunk(
        text=text,
        document_id="doc-a",
        filename="example.pdf",
        page=page,
        chunk=number,
        extraction_method="native",
    )


class DecompositionTests(unittest.TestCase):
    def test_explicit_both_question_is_split(self) -> None:
        question = (
            "How should teams protect both model artifacts and the inputs "
            "and outputs handled by AI code?"
        )
        self.assertEqual(
            decompose_question(question),
            [
                "How should teams protect model artifacts?",
                "How should teams protect the inputs and outputs handled by AI code?",
            ],
        )

    def test_normal_question_is_not_split(self) -> None:
        question = "What are the four AI RMF functions?"
        self.assertEqual(decompose_question(question), [question])


class RetrievalTests(unittest.TestCase):
    def setUp(self) -> None:
        self.first = make_chunk(1, 1, "general model security guidance")
        self.second = make_chunk(2, 2, "GV-1.2 training data provenance")
        self.third = make_chunk(3, 3, "incident response and disclosure")

    def test_bm25_rewards_exact_rare_identifier(self) -> None:
        index = BM25Index([self.first, self.second, self.third])
        result = index.search("What does GV-1.2 require?", k=3)
        self.assertEqual(result[0][0], self.second)

    def test_rrf_rewards_candidate_found_by_both_retrievers(self) -> None:
        fused = reciprocal_rank_fusion(
            dense_results=[(self.first, 0.1), (self.second, 0.2)],
            bm25_results=[(self.second, 9.0), (self.third, 5.0)],
        )
        self.assertEqual(fused[0].chunk, self.second)

    def test_balanced_selector_returns_distinct_pages(self) -> None:
        duplicate_page = make_chunk(4, 1, "another chunk on page one")
        dense = [(self.first, 0.1), (duplicate_page, 0.2)]
        lexical = [(duplicate_page, 8.0), (self.second, 7.0)]
        fused = reciprocal_rank_fusion(
            dense,
            [*lexical, (self.third, 6.0)],
        )
        selected = select_balanced(dense, lexical, fused, top_k=3)
        pages = [candidate.chunk.page for candidate in selected]
        self.assertEqual(pages, [1, 2, 3])


if __name__ == "__main__":
    unittest.main()

