from __future__ import annotations

import json
import math
import time
from collections import Counter
from dataclasses import dataclass

from langchain_ollama import ChatOllama

from build_index import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
)
from rag_query import GENERATION_MODEL, generate_answer, stop_ollama_model
from run_balanced_hybrid_evaluation import (
    FINAL_TOP_K,
    normalize_balanced_retrieved,
    page_key,
    retrieve_balanced_hybrid,
)
from run_decomposed_hybrid_evaluation import (
    DecomposedSelection,
    QueryRun,
    decompose_question,
    normalize_decomposed_retrieved,
    retrieve_query,
    select_decomposed_top_3,
)
from run_hybrid_evaluation import (
    BM25_B,
    BM25_CANDIDATE_K,
    BM25_K1,
    DENSE_CANDIDATE_K,
    RRF_CONSTANT,
    BM25Index,
    FusedCandidate,
    load_frozen_chunks,
    tokenize,
)
from run_ocr_evaluation import (
    DATABASE_DIR,
    GOLD_FILE,
    RESULTS_DIR,
    extract_citations,
    is_abstention,
    open_evaluation_vector_store,
    read_jsonl,
    validate_inputs,
)


# Controlled experiment: only decomposed subqueries receive a deterministic
# pseudo-relevance-feedback (PRF) pass. All ordinary questions retain the
# frozen balanced-hybrid retrieval policy.
PREDICTIONS_FILE = (
    RESULTS_DIR / "prf_decomposed_hybrid_predictions.jsonl"
)
CONFIG_FILE = RESULTS_DIR / "prf_decomposed_hybrid_run_config.json"

PRF_FEEDBACK_PAGES = 3
PRF_EXPANSION_TERMS = 6

# Function words and generic instruction words are excluded so feedback terms
# carry topical information rather than merely repeating question grammar.
PRF_STOPWORDS = {
    "about",
    "after",
    "again",
    "against",
    "also",
    "among",
    "and",
    "any",
    "are",
    "because",
    "been",
    "before",
    "being",
    "both",
    "but",
    "can",
    "could",
    "did",
    "does",
    "doing",
    "each",
    "for",
    "from",
    "had",
    "has",
    "have",
    "having",
    "how",
    "into",
    "its",
    "may",
    "might",
    "more",
    "most",
    "must",
    "not",
    "only",
    "other",
    "our",
    "out",
    "over",
    "should",
    "such",
    "than",
    "that",
    "the",
    "their",
    "them",
    "then",
    "there",
    "these",
    "they",
    "this",
    "those",
    "through",
    "under",
    "using",
    "very",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "while",
    "who",
    "why",
    "will",
    "with",
    "would",
    "your",
}


@dataclass(frozen=True)
class PRFTrace:
    query_index: int
    original_query: str
    expanded_query: str
    expansion_terms: list[str]
    feedback_pages: list[tuple[str, int]]
    first_pass: QueryRun
    second_pass: QueryRun


def unique_feedback_candidates(
    ranked: list[FusedCandidate],
) -> list[FusedCandidate]:
    """Return the strongest candidates from distinct physical pages."""
    selected: list[FusedCandidate] = []
    seen_pages: set[tuple[str, int]] = set()

    for candidate in ranked:
        page = page_key(candidate.document)
        if page in seen_pages:
            continue
        selected.append(candidate)
        seen_pages.add(page)
        if len(selected) == PRF_FEEDBACK_PAGES:
            break

    return selected


def select_expansion_terms(
    query: str,
    first_pass: QueryRun,
    bm25_index: BM25Index,
) -> tuple[list[str], list[tuple[str, int]]]:
    """
    Select corpus-derived PRF terms from the strongest first-pass pages.

    Terms appearing across more feedback pages are preferred. Within equal
    support, BM25 inverse-document frequency, local frequency, and retrieval
    rank determine the order. No gold answer, question ID, or synonym list is
    consulted.
    """
    query_terms = set(tokenize(query))
    feedback = unique_feedback_candidates(first_pass.fused_ranked)
    support: Counter[str] = Counter()
    weighted_scores: Counter[str] = Counter()

    for rank, candidate in enumerate(feedback, start=1):
        frequencies = Counter(tokenize(candidate.document.page_content))
        rank_weight = 1.0 / rank

        for term, frequency in frequencies.items():
            if (
                term in query_terms
                or term in PRF_STOPWORDS
                or len(term) < 4
                or not term.isalpha()
            ):
                continue

            inverse_document_frequency = (
                bm25_index.inverse_document_frequencies.get(term)
            )
            if inverse_document_frequency is None:
                continue

            support[term] += 1
            weighted_scores[term] += (
                rank_weight
                * inverse_document_frequency
                * (1.0 + math.log(frequency))
            )

    ranked_terms = sorted(
        weighted_scores,
        key=lambda term: (
            -support[term],
            -weighted_scores[term],
            term,
        ),
    )
    selected_terms = ranked_terms[:PRF_EXPANSION_TERMS]
    feedback_pages = [
        page_key(candidate.document) for candidate in feedback
    ]
    return selected_terms, feedback_pages


def expand_subquery(
    vector_store,
    bm25_index: BM25Index,
    query: str,
    query_index: int,
) -> PRFTrace:
    """Run first-pass retrieval, derive PRF terms, then retrieve again."""
    first_pass = retrieve_query(
        vector_store=vector_store,
        bm25_index=bm25_index,
        query=query,
        query_index=query_index,
    )
    terms, feedback_pages = select_expansion_terms(
        query=query,
        first_pass=first_pass,
        bm25_index=bm25_index,
    )
    expanded_query = (
        f"{query.rstrip()} {' '.join(terms)}" if terms else query
    )
    second_pass = retrieve_query(
        vector_store=vector_store,
        bm25_index=bm25_index,
        query=expanded_query,
        query_index=query_index,
    )
    return PRFTrace(
        query_index=query_index,
        original_query=query,
        expanded_query=expanded_query,
        expansion_terms=terms,
        feedback_pages=feedback_pages,
        first_pass=first_pass,
        second_pass=second_pass,
    )


def retrieve_with_prf_decomposition(
    vector_store,
    bm25_index: BM25Index,
    question: str,
) -> tuple[
    list,
    list[DecomposedSelection] | None,
    list,
    list[str],
    list[PRFTrace],
]:
    subqueries = decompose_question(question)

    # Preserve the balanced-hybrid control exactly for ordinary questions.
    if len(subqueries) == 1:
        (
            results,
            balanced_selected,
            _dense_count,
            _bm25_count,
        ) = retrieve_balanced_hybrid(
            vector_store=vector_store,
            bm25_index=bm25_index,
            question=question,
        )
        return results, None, balanced_selected, subqueries, []

    traces = [
        expand_subquery(
            vector_store=vector_store,
            bm25_index=bm25_index,
            query=subquery,
            query_index=index,
        )
        for index, subquery in enumerate(subqueries, start=1)
    ]
    full_question_run = retrieve_query(
        vector_store=vector_store,
        bm25_index=bm25_index,
        query=question,
        query_index=0,
    )
    selected = select_decomposed_top_3(
        subquery_runs=[trace.second_pass for trace in traces],
        full_question_run=full_question_run,
    )
    results = [
        (item.candidate.document, -item.candidate.rrf_score)
        for item in selected
    ]
    return results, selected, [], subqueries, traces


def save_run_config(corpus_size: int) -> None:
    config = {
        "experiment_id": "ocr_prf_decomposed_balanced_hybrid_top_3",
        "independent_variable": (
            "pseudo-relevance-feedback expansion for decomposed subqueries"
        ),
        "control_experiment": "ocr_decomposed_balanced_hybrid_top_3",
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
                "candidate_scoring": "equal-weight Reciprocal Rank Fusion",
                "rrf_constant": RRF_CONSTANT,
                "decomposition_trigger": "explicit both X and Y syntax",
                "decomposed_selection": [
                    "best second-pass RRF page for first subquery",
                    "best second-pass RRF page for second subquery",
                    "best remaining full-question RRF page",
                ],
                "non_decomposed_selection": (
                    "unchanged balanced-hybrid top-3 policy"
                ),
                "page_deduplication": True,
            },
            "pseudo_relevance_feedback": {
                "applies_to": "decomposed subqueries only",
                "feedback_unique_pages": PRF_FEEDBACK_PAGES,
                "maximum_expansion_terms": PRF_EXPANSION_TERMS,
                "term_source": "first-pass retrieved text",
                "term_ranking": (
                    "feedback-page support, BM25 IDF, local frequency, "
                    "and retrieval rank"
                ),
                "uses_gold_labels_or_evidence": False,
                "uses_handwritten_synonyms": False,
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
    print("Controlled experiment: PRF-decomposed balanced hybrid top-3")
    print(f"Reusing frozen OCR index: {DATABASE_DIR.resolve()}")
    print("PRF trigger: decomposed 'both X and Y' subqueries only")
    print(
        f"PRF: {PRF_FEEDBACK_PAGES} feedback pages, "
        f"up to {PRF_EXPANSION_TERMS} corpus-derived terms"
    )
    print("All ordinary questions: unchanged balanced-hybrid retrieval")
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
                    decomposed_selected,
                    balanced_selected,
                    subqueries,
                    prf_traces,
                ) = retrieve_with_prf_decomposition(
                    vector_store=vector_store,
                    bm25_index=bm25_index,
                    question=question,
                )
                retrieval_ms = (
                    time.perf_counter() - retrieval_start
                ) * 1000

                if len(results) != FINAL_TOP_K:
                    raise RuntimeError(
                        f"Expected {FINAL_TOP_K} selected chunks, "
                        f"but received {len(results)}."
                    )
                if len({page_key(document) for document, _ in results}) != 3:
                    raise RuntimeError(
                        "PRF-decomposed fusion produced duplicate pages."
                    )

                if decomposed_selected is None:
                    normalized_retrieved = normalize_balanced_retrieved(
                        balanced_selected
                    )
                else:
                    normalized_retrieved = normalize_decomposed_retrieved(
                        decomposed_selected
                    )

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
                    "decomposition": {
                        "applied": len(subqueries) > 1,
                        "subqueries": (
                            subqueries if len(subqueries) > 1 else []
                        ),
                    },
                    "pseudo_relevance_feedback": {
                        "applied": bool(prf_traces),
                        "queries": [
                            {
                                "query_index": trace.query_index,
                                "original_query": trace.original_query,
                                "expanded_query": trace.expanded_query,
                                "expansion_terms": trace.expansion_terms,
                                "feedback_pages": [
                                    {
                                        "source": source,
                                        "page": page,
                                    }
                                    for source, page in trace.feedback_pages
                                ],
                            }
                            for trace in prf_traces
                        ],
                    },
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
                    "Decomposition: "
                    + (
                        " | ".join(subqueries)
                        if len(subqueries) > 1
                        else "not applied"
                    )
                )
                if prf_traces:
                    for trace in prf_traces:
                        print(
                            f"PRF query {trace.query_index}: "
                            f"{trace.expanded_query}"
                        )
                        print(
                            "PRF feedback pages: "
                            + ", ".join(
                                f"{source}:p{page}"
                                for source, page in trace.feedback_pages
                            )
                        )
                else:
                    print("PRF: not applied")
                if decomposed_selected is not None:
                    print(
                        "Selected: "
                        + ", ".join(
                            (
                                f"{rank}:{item.selection_role},"
                                f"query={item.query_index},"
                                f"rrf={item.query_rrf_rank}"
                            )
                            for rank, item in enumerate(
                                decomposed_selected,
                                start=1,
                            )
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
    print("\nCalculate PRF-decomposition metrics with:")
    print(
        r"python rag-evaluation-starter\evaluation\metrics.py "
        r"--gold rag-evaluation-starter\evaluation\gold_questions.jsonl "
        r"--predictions "
        r"rag-evaluation-starter\results"
        r"\prf_decomposed_hybrid_predictions.jsonl "
        r"--output rag-evaluation-starter\results"
        r"\prf_decomposed_hybrid_metrics.json"
    )


def main() -> None:
    run_evaluation()


if __name__ == "__main__":
    main()
