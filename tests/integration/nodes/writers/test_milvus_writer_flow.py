from unittest.mock import MagicMock, patch

import pytest

from dynamiq.callbacks import TracingCallbackHandler
from dynamiq.connections import Milvus as MilvusConnection
from dynamiq.connections import MilvusDeploymentType
from dynamiq.flows import Flow
from dynamiq.nodes.writers.milvus import MilvusDocumentWriter
from dynamiq.runnables import RunnableConfig, RunnableResult, RunnableStatus
from dynamiq.storages.vector.milvus.milvus import MilvusVectorStore
from dynamiq.types import Document


@pytest.fixture
def mock_milvus_client():
    mock_client = MagicMock()
    mock_client.has_collection.return_value = True
    return mock_client


@pytest.fixture
def mock_milvus_connection():
    return MilvusConnection(
        deployment_type=MilvusDeploymentType.FILE,
        uri="test.db",
    )


@pytest.fixture
def mock_milvus_vector_store(mock_milvus_client, mock_milvus_connection):
    with patch("pymilvus.MilvusClient", return_value=mock_milvus_client):
        vector_store = MilvusVectorStore(connection=mock_milvus_connection)
        vector_store.write_documents = MagicMock(return_value=2)
        return vector_store


@pytest.fixture
def mock_documents():
    return [
        Document(id="1", content="Document 1", embedding=[0.1, 0.2, 0.3]),
        Document(id="2", content="Document 2", embedding=[0.4, 0.5, 0.6]),
    ]


@patch("dynamiq.callbacks.TracingCallbackHandler.on_flow_start")
@patch("dynamiq.callbacks.TracingCallbackHandler.on_flow_end")
def test_milvus_write_workflow(
    mock_on_flow_end,
    mock_on_flow_start,
    mock_documents,
    mock_milvus_vector_store,
):
    document_writer_node = MilvusDocumentWriter(
        vector_store=mock_milvus_vector_store,
        index_name="test_collection",
    )

    indexing_flow = Flow(
        id="test_milvus_indexing_flow",
        nodes=[
            document_writer_node,
        ],
    )

    input_data = {"documents": [doc.to_dict() for doc in mock_documents]}
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
        input=input_data,
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

    mock_milvus_vector_store.write_documents.assert_called_once_with(input_data["documents"])
