import pytest

from dynamiq.nodes.rankers.recency import TimeWeightedDocumentRanker
from dynamiq.runnables import RunnableStatus
from dynamiq.types import Document


@pytest.mark.parametrize(
    "top_k, expected_ids", [(0, []), (1, ["recent"]), (2, ["recent", "old"]), (5, ["recent", "old"])]
)
def test_recency_ranker_limits_results_after_adjusting_scores(top_k, expected_ids):
    ranker = TimeWeightedDocumentRanker(top_k=top_k, max_days=1, min_coefficient=0.1)
    documents = [
        Document(id="old", content="Old document", score=0.9, metadata={"date": "01 January, 2000"}),
        Document(id="recent", content="Recent document", score=0.5, metadata={"date": "01 January, 2999"}),
    ]

    result = ranker.run(input_data={"documents": documents})

    assert result.status == RunnableStatus.SUCCESS
    ranked = result.output["documents"]
    assert [doc.id for doc in ranked] == expected_ids
    if ranked:
        assert ranked[0].score == pytest.approx(0.5)


def test_recency_ranker_applies_default_top_k():
    ranker = TimeWeightedDocumentRanker()
    documents = [Document(id=str(i), content=f"Document {i}", score=i / 10, metadata={}) for i in range(7)]

    result = ranker.run(input_data={"documents": documents})

    assert result.status == RunnableStatus.SUCCESS
    assert [doc.id for doc in result.output["documents"]] == ["6", "5", "4", "3", "2"]


def test_recency_ranker_accepts_empty_documents():
    result = TimeWeightedDocumentRanker(top_k=1).run(input_data={"documents": []})

    assert result.status == RunnableStatus.SUCCESS
    assert result.output["documents"] == []
