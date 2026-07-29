from metrics import (
    evidence_units,
    exact_match,
    precision_recall_f1,
    ranked_retrieval_metrics,
    required_fact_recall,
    token_f1,
)


def test_text_metrics() -> None:
    assert exact_match("The GOVERN function.", "govern function") == 1.0
    assert token_f1("mitigate or accept", "mitigate transfer avoid or accept") > 0.5
    facts = [["mitigate", "mitigating"], ["accept", "accepting"]]
    assert required_fact_recall("Mitigating or accepting risk.", facts) == 1.0


def test_retrieval_metrics() -> None:
    gold = {("doc-a", 2), ("doc-a", 5)}
    retrieved = [
        {"doc_id": "doc-x", "pages": [1]},
        {"doc_id": "doc-a", "pages": [5]},
        {"doc_id": "doc-a", "pages": [2]},
    ]
    metrics = ranked_retrieval_metrics(retrieved, gold, 3)
    assert metrics["hit@3"] == 1.0
    assert metrics["recall@3"] == 1.0
    assert metrics["mrr@3"] == 0.5


def test_citation_metrics() -> None:
    gold = evidence_units([{"doc_id": "doc-a", "pages": [2, 5]}])
    predicted = evidence_units(
        [{"doc_id": "doc-a", "pages": [2]}, {"doc_id": "doc-b", "pages": [7]}]
    )
    precision, recall, f1 = precision_recall_f1(predicted, gold)
    assert precision == 0.5
    assert recall == 0.5
    assert f1 == 0.5


if __name__ == "__main__":
    test_text_metrics()
    test_retrieval_metrics()
    test_citation_metrics()
    print("All metric tests passed.")
