"""Live tests for the Turbopuffer writer and retriever against a real Turbopuffer namespace.

Environment
-----------
    TURBOPUFFER_API_KEY    required; the tests skip without it.
    TURBOPUFFER_REGION     optional; defaults to aws-us-east-1.

Each run writes to a namespace of its own and deletes it afterwards. Embeddings are fixed vectors,
so no embedding provider is needed.
"""

import math
import os
import random
import uuid

import pytest
import turbopuffer

from dynamiq.connections import Turbopuffer
from dynamiq.nodes.retrievers import TurbopufferDocumentRetriever
from dynamiq.nodes.writers import TurbopufferDocumentWriter
from dynamiq.runnables import RunnableConfig, RunnableStatus
from dynamiq.types import Document

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not os.getenv("TURBOPUFFER_API_KEY"), reason="TURBOPUFFER_API_KEY is not set"),
]


def _vector(seed: int, dims: int = 8) -> list[float]:
    rnd = random.Random(seed)
    values = [rnd.uniform(-1, 1) for _ in range(dims)]
    norm = math.sqrt(sum(v * v for v in values))
    return [v / norm for v in values]


@pytest.fixture
def connection():
    return Turbopuffer(region=os.getenv("TURBOPUFFER_REGION") or "aws-us-east-1")


@pytest.fixture
def namespace(connection):
    name = f"dynamiq-test-{uuid.uuid4().hex[:12]}"
    yield name
    try:
        connection.connect().namespace(name).delete_all()
    except turbopuffer.NotFoundError:
        pass


@pytest.fixture
def writer(connection, namespace):
    return TurbopufferDocumentWriter(connection=connection, index_name=namespace)


@pytest.fixture
def retriever(connection, namespace):
    return TurbopufferDocumentRetriever(connection=connection, index_name=namespace, top_k=5)


def _documents() -> list[Document]:
    return [
        Document(
            id="refund-2025",
            content="Customers can request a full refund within 30 days of purchase.",
            embedding=_vector(1),
            metadata={
                "file_id": "f1",
                "dynamiq_item_acl": ["public"],
                "title": "Refund policy",
                "page_number": 1,
                "pdf_info": {"Author": "Support", "Pages": 3},
            },
        ),
        Document(
            id="refund-2022",
            content="Superseded: customers could request a refund within 7 days.",
            embedding=_vector(2),
            metadata={"file_id": "f2", "dynamiq_item_acl": [], "title": "Old refund policy", "page_number": "2"},
        ),
        Document(
            id="oncall",
            content="On-call engineers acknowledge a page within five minutes.",
            embedding=_vector(3),
            metadata={"file_id": "f3", "dynamiq_item_acl": ["external_group:eng"], "title": "On-call"},
        ),
    ]


def _retrieve(retriever: TurbopufferDocumentRetriever, **kwargs) -> list[Document]:
    result = retriever.run(input_data=kwargs, config=RunnableConfig(callbacks=[]))
    assert result.status == RunnableStatus.SUCCESS, result.error
    return [d if isinstance(d, Document) else Document(**d) for d in result.output["documents"]]


def test_retriever_on_never_written_namespace_returns_nothing(retriever):
    assert _retrieve(retriever, embedding=_vector(1), query="refund") == []


def test_write_search_and_delete(writer, retriever):
    result = writer.run(input_data={"documents": _documents()}, config=RunnableConfig(callbacks=[]))
    assert result.status == RunnableStatus.SUCCESS, result.error
    assert result.output["upserted_count"] == 3

    vector = _retrieve(retriever, embedding=_vector(1))
    assert vector[0].id == "refund-2025"
    assert vector[0].score == pytest.approx(1.0, abs=1e-4)

    hybrid = _retrieve(retriever, embedding=_vector(3), query="refund within 30 days", alpha=0.5)
    assert {d.id for d in hybrid} >= {"refund-2025", "oncall"}
    assert all(0 <= d.score <= 1 for d in hybrid)

    acl = {"field": "dynamiq_item_acl", "operator": "contains_any", "value": ["public", "external_group:eng"]}
    allowed = _retrieve(retriever, embedding=_vector(2), query="refund", filters=acl)
    assert {d.id for d in allowed} == {"refund-2025", "oncall"}

    missing = {"field": "does_not_exist", "operator": "==", "value": "x"}
    assert _retrieve(retriever, embedding=_vector(1), filters=missing) == []

    stored = {d.id: d for d in retriever.vector_store.get_documents_by_id(["refund-2025", "refund-2022"])}
    assert stored["refund-2025"].metadata["pdf_info"] == '{"Author": "Support", "Pages": 3}'
    assert stored["refund-2022"].metadata["page_number"] == 2

    retriever.vector_store.delete_documents_by_file_ids(["f1", "f2"])
    assert [d.id for d in retriever.vector_store.list_documents()] == ["oncall"]
