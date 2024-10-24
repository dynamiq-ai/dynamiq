from unittest.mock import MagicMock, patch

import pytest

from dynamiq.callbacks import TracingCallbackHandler
from dynamiq.connections import Qdrant as QdrantConnection
from dynamiq.flows import Flow
from dynamiq.nodes.writers import QdrantDocumentWriter
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
def mock_set_up_collection():
    with patch(
        "dynamiq.storages.vector.qdrant.qdrant.QdrantVectorStore._set_up_collection", return_value=None
    ) as mock_set_up:
        yield mock_set_up


@patch("dynamiq.callbacks.TracingCallbackHandler.on_flow_start")
@patch("dynamiq.callbacks.TracingCallbackHandler.on_flow_end")
def test_write_workflow(
    mock_on_flow_end,
    mock_on_flow_start,
    mock_documents,
    mock_qdrant_vector_store,
    mock_set_up_collection,
):
    document_writer_node = QdrantDocumentWriter(
        vector_store=mock_qdrant_vector_store,
        index_name="test-collection",
    )

    # Build the indexing flow
    indexing_flow = Flow(
        id="test_indexing_flow",
        nodes=[
            document_writer_node,
        ],
    )

    # Run the indexing flow
    input_data = {"documents": mock_documents}
    config = RunnableConfig(callbacks=[TracingCallbackHandler()])
    response = indexing_flow.run(
        input_data=input_data,
        config=config,
    )

    assert response is not None
    assert response.status == RunnableStatus.SUCCESS

    node_id = list(response.output.keys())[0]

    expected_result = RunnableResult(
        status=RunnableStatus.SUCCESS,
        input={"documents": mock_documents},
        output={"upserted_count": len(mock_documents)},
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
