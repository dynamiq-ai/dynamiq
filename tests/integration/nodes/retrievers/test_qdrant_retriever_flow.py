from unittest.mock import MagicMock, patch

import pytest

from dynamiq.callbacks import TracingCallbackHandler
from dynamiq.connections import Qdrant as QdrantConnection
from dynamiq.flows import Flow
from dynamiq.nodes.retrievers import QdrantDocumentRetriever
from dynamiq.runnables import RunnableConfig, RunnableResult, RunnableStatus
from dynamiq.storages.vector.qdrant.qdrant import QdrantVectorStore


@pytest.fixture
def mock_qdrant_client():
    return MagicMock()


@pytest.fixture
def mock_qdrant_connection():
    return QdrantConnection(
        id="qdrant-connection",
        api_key="test-qdrant-api-key",
        url="http://test-localhost:6333",
    )


@pytest.fixture
def mock_qdrant_vector_store(mock_qdrant_client, mock_qdrant_connection):
    with patch("qdrant_client.QdrantClient", return_value=mock_qdrant_client):
        return QdrantVectorStore(connection=mock_qdrant_connection)


@pytest.fixture
def mock_query_by_embedding(mock_documents):
    with patch(
        "dynamiq.storages.vector.qdrant.qdrant.QdrantVectorStore._query_by_embedding",
        return_value=mock_documents,
    ) as mock_query_by_embedding:
        yield mock_query_by_embedding


@patch("dynamiq.callbacks.TracingCallbackHandler.on_flow_start")
@patch("dynamiq.callbacks.TracingCallbackHandler.on_flow_end")
@patch("dynamiq.storages.vector.qdrant.qdrant.QdrantVectorStore._set_up_collection")
def test_retrieve_workflow(
    mock_set_up_collection,
    mock_on_flow_end,
    mock_on_flow_start,
    mock_documents,
    mock_qdrant_vector_store,
    mock_query_by_embedding,
    mock_filters,
):
    document_retriever_node = QdrantDocumentRetriever(
        vector_store=mock_qdrant_vector_store,
        index_name="test-collection",
    )

    # Build the retriever flow
    retriever_flow = Flow(
        id="test_retriever_flow",
        nodes=[
            document_retriever_node,
        ],
    )

    # Run the retriever flow
    input_data = {"embedding": [0.1, 0.2, 0.3], "filters": mock_filters, "top_k": 5}
    config = RunnableConfig(callbacks=[TracingCallbackHandler()])
    response = retriever_flow.run(
        input_data=input_data,
        config=config,
    )

    assert response is not None
    assert response.status == RunnableStatus.SUCCESS

    node_id = list(response.output.keys())[0]

    expected_result = RunnableResult(
        status=RunnableStatus.SUCCESS,
        input=input_data,
        output={"documents": [doc.to_dict() for doc in mock_documents]},
    ).to_dict()
    expected_output = {node_id: expected_result}

    assert (
        response.to_dict()
        == RunnableResult(
            status=RunnableStatus.SUCCESS,
            input=input_data,
            output=expected_output,
        ).to_dict()
    )
    assert response.output == expected_output

    mock_on_flow_start.assert_called_once()
    mock_on_flow_end.assert_called_once()
    mock_query_by_embedding.assert_called_once_with(
        query_embedding=input_data["embedding"],
        filters=mock_filters,
        top_k=input_data["top_k"],
        return_embedding=False,
    )

    mock_set_up_collection.assert_not_called()
