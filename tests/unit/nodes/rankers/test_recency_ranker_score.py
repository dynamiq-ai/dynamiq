"""Documents without a score must not fail the recency ranker.

``Document.score`` defaults to ``None``, and every converter builds its documents as
``Document(content=..., metadata=...)`` -- its output is exactly the ``{"documents": [...]}``
this ranker takes. Loading files and ranking the chunks by their date is the reason to use a
time-weighted ranker, yet an unscored document made the node fail: multiplying ``None`` by a
coefficient raised a bare ``TypeError``, and a document with no date either raised one from
``sorted``'s comparison. With no similarity to weigh, recency alone has to decide.
"""

import pytest

from dynamiq.nodes.rankers.recency import TimeWeightedDocumentRanker
from dynamiq.runnables import RunnableStatus
from dynamiq.types import Document


def _rank(documents: list[Document], **kwargs) -> list[Document]:
    ranker = TimeWeightedDocumentRanker(**kwargs)
    result = ranker.run(input_data={"documents": documents})

    assert result.status == RunnableStatus.SUCCESS, result.error.message if result.error else None
    return result.output["documents"]


def test_unscored_documents_are_ranked_by_recency():
    documents = [
        Document(id="old", content="Old", metadata={"date": "01 January, 2000"}),
        Document(id="recent", content="Recent", metadata={"date": "01 January, 2999"}),
    ]

    ranked = _rank(documents, max_days=1, min_coefficient=0.1)

    assert [doc.id for doc in ranked] == ["recent", "old"]
    assert ranked[0].score == pytest.approx(1.0)
    assert ranked[1].score == pytest.approx(0.1)


def test_unscored_documents_without_a_date_rank_after_scored_ones():
    documents = [
        Document(id="undated", content="Undated"),
        Document(id="low", content="Low", score=0.2),
        Document(id="high", content="High", score=0.9),
    ]

    ranked = _rank(documents, top_k=3)

    assert [doc.id for doc in ranked] == ["high", "low", "undated"]
    assert ranked[-1].score is None


def test_unscored_documents_without_a_date_keep_their_order():
    documents = [Document(id=str(i), content=f"Document {i}") for i in range(3)]

    ranked = _rank(documents, top_k=3)

    assert [doc.id for doc in ranked] == ["0", "1", "2"]


def test_top_k_still_limits_unscored_documents():
    documents = [
        Document(id=str(year), content=f"Document {year}", metadata={"date": f"01 January, {year}"})
        for year in (2000, 2020, 2999)
    ]

    ranked = _rank(documents, top_k=2)

    assert [doc.id for doc in ranked] == ["2999", "2020"]
