from unittest.mock import MagicMock, patch

import pytest

from dynamiq.components.retrievers.turbopuffer import (
    TurbopufferDocumentRetriever as TurbopufferDocumentRetrieverComponent,
)
from dynamiq.connections import Turbopuffer
from dynamiq.connections.managers import ConnectionManager
from dynamiq.nodes.retrievers import TurbopufferDocumentRetriever
from dynamiq.nodes.retrievers.base import RetrieverInputSchema
from dynamiq.runnables import RunnableConfig
from dynamiq.storages.vector.turbopuffer import TurbopufferVectorStore
from dynamiq.types import Document


@pytest.fixture
def vector_store():
    store = MagicMock(spec=TurbopufferVectorStore)
    store.client = MagicMock()
    return store


@patch("dynamiq.connections.Turbopuffer.connect", return_value=MagicMock())
def test_initialization_with_defaults(mock_connect):
    with patch.object(TurbopufferDocumentRetriever, "init_components"):
        retriever = TurbopufferDocumentRetriever()

    assert isinstance(retriever.connection, Turbopuffer)
    assert retriever.alpha == 0.5


def test_vector_store_params_leave_out_max_vector_distance():
    connection = Turbopuffer(api_key="key", region="aws-us-east-1")
    with patch.object(TurbopufferDocumentRetriever, "init_components"):
        retriever = TurbopufferDocumentRetriever(
            connection=connection, index_name="vs-1", alpha=0.3, max_vector_distance=0.4
        )

    params = retriever.vector_store_params

    assert params["index_name"] == "vs-1"
    assert params["alpha"] == 0.3
    assert "max_vector_distance" not in params
    assert params["connection"] is connection


def test_init_components(vector_store):
    retriever = TurbopufferDocumentRetriever(vector_store=vector_store, top_k=4)

    retriever.init_components(MagicMock(spec=ConnectionManager))

    assert isinstance(retriever.document_retriever, TurbopufferDocumentRetrieverComponent)
    assert retriever.document_retriever.top_k == 4


def test_execute_passes_hybrid_parameters(vector_store):
    retriever = TurbopufferDocumentRetriever(vector_store=vector_store, alpha=0.3)
    retriever.document_retriever = MagicMock()
    retriever.document_retriever.run.return_value = {"documents": [Document(id="1", content="one")]}
    input_data = RetrieverInputSchema(
        embedding=[0.1, 0.2], filters={"file_id": "f1"}, top_k=5, query="fox", similarity_threshold=0.2
    )

    result = retriever.execute(input_data, RunnableConfig(callbacks=[]))

    retriever.document_retriever.run.assert_called_once_with(
        [0.1, 0.2],
        filters={"file_id": "f1"},
        top_k=5,
        content_key=None,
        query="fox",
        alpha=0.3,
        similarity_threshold=0.2,
        max_vector_distance=None,
    )
    assert [d.id for d in result["documents"]] == ["1"]


def test_component_rejects_other_vector_stores():
    with pytest.raises(ValueError):
        TurbopufferDocumentRetrieverComponent(vector_store=MagicMock())


def test_component_hybrid_applies_threshold_to_fused_scores(vector_store):
    vector_store._hybrid_retrieval.return_value = [
        Document(id="1", content="one", score=0.9),
        Document(id="2", content="two", score=0.1),
    ]
    component = TurbopufferDocumentRetrieverComponent(vector_store=vector_store, similarity_threshold=0.5)

    result = component.run([0.1, 0.2], query="fox", alpha=0.5)

    assert [d.id for d in result["documents"]] == ["1"]
    assert vector_store._hybrid_retrieval.call_args.kwargs["alpha"] == 0.5


@pytest.mark.parametrize(
    "threshold, min_score, max_distance",
    [
        (None, None, None),
        (0.8, 0.8, None),
        (1.2, None, 1.2),
    ],
)
def test_component_vector_search_reads_threshold_like_weaviate(vector_store, threshold, min_score, max_distance):
    vector_store._embedding_retrieval.return_value = []
    component = TurbopufferDocumentRetrieverComponent(vector_store=vector_store)

    component.run([0.1, 0.2], similarity_threshold=threshold)

    kwargs = vector_store._embedding_retrieval.call_args.kwargs
    assert kwargs["min_score"] == min_score
    assert kwargs["max_distance"] == max_distance
