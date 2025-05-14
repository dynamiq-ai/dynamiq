import uuid
from unittest.mock import MagicMock, patch

import pytest
from litellm import APIError, AuthenticationError, BadRequestError, RateLimitError

from dynamiq import Workflow, connections
from dynamiq.callbacks import TracingCallbackHandler
from dynamiq.flows import Flow
from dynamiq.nodes.embedders import VertexAIDocumentEmbedder, VertexAITextEmbedder
from dynamiq.nodes.node import NodeDependency
from dynamiq.nodes.utils import Output
from dynamiq.runnables import RunnableConfig, RunnableStatus
from dynamiq.types import Document


@pytest.fixture
def vertexai_connection():
    return connections.VertexAI(
        id=str(uuid.uuid4()),
        project_id="mock_project",
        private_key_id="mock_key_id",
        private_key="mock_private_key",
        client_email="mock_client_email",
        client_id="mock_client_id",
        auth_uri="https://mock.auth.uri",
        token_uri="https://mock.token.uri",
        auth_provider_x509_cert_url="https://mock.cert.url",
        client_x509_cert_url="https://mock.client.cert.url",
        universe_domain="mock_domain",
        vertex_project_id="mock_vertex_project",
        vertex_project_location="mock_location",
    )


@pytest.fixture
def vertexai_model():
    return "vertex_ai/text-embedding-005"


@pytest.fixture
def vertexai_text_embedder(vertexai_connection, vertexai_model):
    return VertexAITextEmbedder(
        id="text_embedder", name="VertexAITextEmbedder", connection=vertexai_connection, model=vertexai_model
    )


@pytest.fixture
def vertexai_document_embedder(vertexai_connection, vertexai_model):
    return VertexAIDocumentEmbedder(
        id="document_embedder", name="VertexAIDocumentEmbedder", connection=vertexai_connection, model=vertexai_model
    )


@pytest.fixture
def vertexai_text_embedder_workflow(vertexai_text_embedder):
    output_node = Output(id="output_node", depends=[NodeDependency(vertexai_text_embedder)])

    workflow = Workflow(
        id=str(uuid.uuid4()),
        flow=Flow(
            nodes=[vertexai_text_embedder, output_node],
        ),
    )

    return workflow, vertexai_text_embedder, output_node


@pytest.fixture
def vertexai_document_embedder_workflow(vertexai_document_embedder):
    output_node = Output(id="output_node", depends=[NodeDependency(vertexai_document_embedder)])

    workflow = Workflow(
        id=str(uuid.uuid4()),
        flow=Flow(
            nodes=[vertexai_document_embedder, output_node],
        ),
    )

    return workflow, vertexai_document_embedder, output_node


@pytest.fixture
def query_text():
    return "I love pizza!"


@pytest.fixture
def query_input(query_text):
    return {"query": query_text}


@pytest.fixture
def document_content():
    return "Test document content"


@pytest.fixture
def document_input(document_content):
    return {"documents": [Document(content=document_content)]}


def test_workflow_with_vertexai_text_embedder(
    mock_embedding_executor, vertexai_text_embedder_workflow, query_input, vertexai_model, vertexai_connection
):
    workflow, embedder, output_node = vertexai_text_embedder_workflow

    response = workflow.run(
        input_data=query_input,
        config=RunnableConfig(callbacks=[TracingCallbackHandler()]),
    )

    assert response.status == RunnableStatus.SUCCESS

    embedder_result = response.output[embedder.id]
    assert embedder_result["status"] == RunnableStatus.SUCCESS.value
    assert "query" in embedder_result["output"]
    assert embedder_result["output"]["query"] == query_input["query"]
    assert "embedding" in embedder_result["output"]
    assert embedder_result["output"]["embedding"] == [0]
    assert isinstance(embedder_result["output"]["embedding"], list)
    assert len(embedder_result["output"]["embedding"]) == 1

    output_result = response.output[output_node.id]
    assert output_result["status"] == RunnableStatus.SUCCESS.value

    mock_embedding_executor.assert_called_once_with(
        task_type="RETRIEVAL_QUERY",
        input=[query_input["query"]],
        model=vertexai_model,
        **vertexai_connection.conn_params,
    )


def test_workflow_with_vertexai_document_embedder(
    mock_embedding_executor, vertexai_document_embedder_workflow, document_input, vertexai_model, vertexai_connection
):
    workflow, embedder, output_node = vertexai_document_embedder_workflow

    response = workflow.run(
        input_data=document_input,
        config=RunnableConfig(callbacks=[TracingCallbackHandler()]),
    )

    assert response.status == RunnableStatus.SUCCESS

    embedder_result = response.output[embedder.id]
    assert embedder_result["status"] == RunnableStatus.SUCCESS.value
    assert "documents" in embedder_result["output"]
    assert len(embedder_result["output"]["documents"]) == 1

    assert "meta" in embedder_result["output"]
    assert "model" in embedder_result["output"]["meta"]
    assert embedder_result["output"]["meta"]["model"] == vertexai_model

    output_result = response.output[output_node.id]
    assert output_result["status"] == RunnableStatus.SUCCESS.value

    mock_embedding_executor.assert_called_once_with(
        task_type="RETRIEVAL_DOCUMENT",
        input=[document_input["documents"][0].content],
        model=vertexai_model,
        **vertexai_connection.conn_params,
    )


@pytest.fixture
def empty_query_input():
    return {"query": ""}


@pytest.fixture
def missing_input():
    return {}


@pytest.mark.parametrize(
    "error_class,error_msg,error_args,expected_type",
    [
        (
            AuthenticationError,
            "Invalid credentials",
            ["vertex_ai", "vertex_ai/text-embedding-005"],
            "AuthenticationError",
        ),
        (RateLimitError, "Rate limit exceeded", ["vertex_ai", "vertex_ai/text-embedding-005"], "RateLimitError"),
        (APIError, "Service unavailable", [500, "vertex_ai", "vertex_ai/text-embedding-005"], "APIError"),
        (BadRequestError, "Invalid embedding model", ["non-existent-model", "vertex_ai"], "BadRequestError"),
    ],
)
def test_text_embedder_api_errors(vertexai_text_embedder_workflow, error_class, error_msg, error_args, expected_type):
    workflow, embedder, output_node = vertexai_text_embedder_workflow

    with patch("dynamiq.components.embedders.base.BaseEmbedder._embedding") as mock_embedding:
        if error_class == APIError:
            error = error_class(error_args[0], error_msg, error_args[1], error_args[2])
        else:
            error = error_class(error_msg, *error_args)

        mock_embedding.side_effect = error

        input_data = {"query": "Test query"}
        response = workflow.run(input_data=input_data)

        assert response.status == RunnableStatus.SUCCESS

        embedder_result = response.output[embedder.id]
        assert embedder_result["status"] == RunnableStatus.FAILURE.value
        assert expected_type in embedder_result["error"]["type"]
        assert error_msg in embedder_result["error"]["message"]

        output_result = response.output[output_node.id]
        assert output_result["status"] == RunnableStatus.SKIP.value


def test_text_embedder_missing_input(vertexai_text_embedder_workflow, missing_input):
    workflow, embedder, output_node = vertexai_text_embedder_workflow

    response = workflow.run(input_data=missing_input)

    assert response.status == RunnableStatus.SUCCESS

    embedder_result = response.output[embedder.id]
    assert embedder_result["status"] == RunnableStatus.FAILURE.value

    output_result = response.output[output_node.id]
    assert output_result["status"] == RunnableStatus.SKIP.value


def test_text_embedder_empty_input(vertexai_text_embedder_workflow, empty_query_input):
    workflow, embedder, output_node = vertexai_text_embedder_workflow

    response = workflow.run(input_data=empty_query_input)

    assert response.status == RunnableStatus.SUCCESS

    embedder_result = response.output[embedder.id]
    assert embedder_result["status"] == RunnableStatus.FAILURE.value

    output_result = response.output[output_node.id]
    assert output_result["status"] == RunnableStatus.SKIP.value


@pytest.mark.parametrize(
    "error_class,error_msg,error_args,expected_type",
    [
        (
            AuthenticationError,
            "Invalid credentials",
            ["vertex_ai", "vertex_ai/text-embedding-005"],
            "AuthenticationError",
        ),
        (RateLimitError, "Rate limit exceeded", ["vertex_ai", "vertex_ai/text-embedding-005"], "RateLimitError"),
        (APIError, "Service unavailable", [500, "vertex_ai", "vertex_ai/text-embedding-005"], "APIError"),
        (BadRequestError, "Invalid embedding model", ["non-existent-model", "vertex_ai"], "BadRequestError"),
    ],
)
def test_document_embedder_api_errors(
    vertexai_document_embedder_workflow, document_input, error_class, error_msg, error_args, expected_type
):
    workflow, embedder, output_node = vertexai_document_embedder_workflow

    with patch("dynamiq.components.embedders.base.BaseEmbedder._embedding") as mock_embedding:
        if error_class == APIError:
            error = error_class(error_args[0], error_msg, error_args[1], error_args[2])
        else:
            error = error_class(error_msg, *error_args)

        mock_embedding.side_effect = error

        response = workflow.run(input_data=document_input)

        assert response.status == RunnableStatus.SUCCESS

        embedder_result = response.output[embedder.id]
        assert embedder_result["status"] == RunnableStatus.FAILURE.value
        assert expected_type in embedder_result["error"]["type"]
        assert error_msg in embedder_result["error"]["message"]

        output_result = response.output[output_node.id]
        assert output_result["status"] == RunnableStatus.SKIP.value


def test_document_embedder_missing_input(vertexai_document_embedder_workflow, missing_input):
    workflow, embedder, output_node = vertexai_document_embedder_workflow

    response = workflow.run(input_data=missing_input)

    assert response.status == RunnableStatus.SUCCESS

    embedder_result = response.output[embedder.id]
    assert embedder_result["status"] == RunnableStatus.FAILURE.value

    output_result = response.output[output_node.id]
    assert output_result["status"] == RunnableStatus.SKIP.value


@pytest.fixture
def empty_documents_input():
    return {"documents": []}


def test_document_embedder_empty_document_list(vertexai_document_embedder_workflow, empty_documents_input):
    workflow, embedder, output_node = vertexai_document_embedder_workflow

    response = workflow.run(input_data=empty_documents_input)

    assert response.status == RunnableStatus.SUCCESS

    embedder_result = response.output[embedder.id]
    assert embedder_result["status"] == RunnableStatus.SUCCESS.value

    output_result = response.output[output_node.id]
    assert output_result["status"] == RunnableStatus.SUCCESS.value


@pytest.fixture
def empty_document_content_input():
    return {"documents": [Document(content="")]}


def test_document_embedder_empty_content(vertexai_document_embedder_workflow, empty_document_content_input):
    workflow, embedder, output_node = vertexai_document_embedder_workflow

    response = workflow.run(input_data=empty_document_content_input)

    assert response.status == RunnableStatus.SUCCESS

    embedder_result = response.output[embedder.id]
    assert embedder_result["status"] == RunnableStatus.FAILURE.value

    output_result = response.output[output_node.id]
    assert output_result["status"] == RunnableStatus.SKIP.value


@pytest.fixture
def empty_embedding_response(vertexai_model):
    response = MagicMock()
    response.data = [{"embedding": []}]
    response.model = vertexai_model
    response.usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    return response


def test_text_embedder_api_returns_empty_embedding(
    vertexai_text_embedder_workflow, query_input, empty_embedding_response
):
    workflow, embedder, output_node = vertexai_text_embedder_workflow

    with patch("dynamiq.components.embedders.base.BaseEmbedder._embedding") as mock_embedding:
        mock_embedding.return_value = empty_embedding_response

        response = workflow.run(input_data=query_input)

        assert response.status == RunnableStatus.SUCCESS

        embedder_result = response.output[embedder.id]
        assert embedder_result["status"] == RunnableStatus.SUCCESS.value
        assert "embedding" in embedder_result["output"]
        assert embedder_result["output"]["embedding"] == []
        assert isinstance(embedder_result["output"]["embedding"], list)
        assert len(embedder_result["output"]["embedding"]) == 0

        output_result = response.output[output_node.id]
        assert output_result["status"] == RunnableStatus.SUCCESS.value


def test_document_embedder_api_returns_empty_embedding(
    vertexai_document_embedder_workflow, document_input, empty_embedding_response
):
    workflow, embedder, output_node = vertexai_document_embedder_workflow

    with patch("dynamiq.components.embedders.base.BaseEmbedder._embedding") as mock_embedding:
        mock_embedding.return_value = empty_embedding_response

        response = workflow.run(input_data=document_input)

        assert response.status == RunnableStatus.SUCCESS

        embedder_result = response.output[embedder.id]
        assert embedder_result["status"] == RunnableStatus.SUCCESS.value
        assert "documents" in embedder_result["output"]
        assert len(embedder_result["output"]["documents"]) == 1

        output_result = response.output[output_node.id]
        assert output_result["status"] == RunnableStatus.SUCCESS.value


@pytest.fixture
def long_text():
    return "text " * 5000


@pytest.fixture
def long_query_input(long_text):
    return {"query": long_text}


@pytest.fixture
def long_document_input(long_text):
    return {"documents": [Document(content=long_text)]}


@pytest.fixture
def max_tokens_error_message():
    return "Document exceeds the model's maximum input size of 10000 tokens."


def test_text_embedder_max_tokens_error(
    vertexai_text_embedder_workflow, long_query_input, max_tokens_error_message, vertexai_model
):
    workflow, embedder, output_node = vertexai_text_embedder_workflow

    with patch("dynamiq.components.embedders.base.BaseEmbedder._embedding") as mock_embedding:
        error = BadRequestError(max_tokens_error_message, vertexai_model, "vertex_ai")
        mock_embedding.side_effect = error

        response = workflow.run(input_data=long_query_input)

        assert response.status == RunnableStatus.SUCCESS

        embedder_result = response.output[embedder.id]
        assert embedder_result["status"] == RunnableStatus.FAILURE.value
        assert "BadRequestError" in embedder_result["error"]["type"]
        assert "maximum input size" in embedder_result["error"]["message"]

        output_result = response.output[output_node.id]
        assert output_result["status"] == RunnableStatus.SKIP.value


def test_document_embedder_max_tokens_error(
    vertexai_document_embedder_workflow, long_document_input, max_tokens_error_message, vertexai_model
):
    workflow, embedder, output_node = vertexai_document_embedder_workflow

    with patch("dynamiq.components.embedders.base.BaseEmbedder._embedding") as mock_embedding:
        error = BadRequestError(max_tokens_error_message, vertexai_model, "vertex_ai")
        mock_embedding.side_effect = error

        response = workflow.run(input_data=long_document_input)

        assert response.status == RunnableStatus.SUCCESS

        embedder_result = response.output[embedder.id]
        assert embedder_result["status"] == RunnableStatus.FAILURE.value
        assert "BadRequestError" in embedder_result["error"]["type"]
        assert "maximum input size" in embedder_result["error"]["message"]

        output_result = response.output[output_node.id]
        assert output_result["status"] == RunnableStatus.SKIP.value


@pytest.fixture
def invalid_model_error_message():
    return "Model not found"


def test_text_embedder_invalid_model(
    vertexai_text_embedder_workflow, query_input, invalid_model_error_message, vertexai_model
):
    workflow, embedder, output_node = vertexai_text_embedder_workflow

    with patch("dynamiq.components.embedders.base.BaseEmbedder._embedding") as mock_embedding:
        error = BadRequestError(invalid_model_error_message, vertexai_model, "vertex_ai")
        mock_embedding.side_effect = error

        response = workflow.run(input_data=query_input)

        assert response.status == RunnableStatus.SUCCESS

        embedder_result = response.output[embedder.id]
        assert embedder_result["status"] == RunnableStatus.FAILURE.value
        assert "BadRequestError" in embedder_result["error"]["type"]
        assert "not found" in embedder_result["error"]["message"]

        output_result = response.output[output_node.id]
        assert output_result["status"] == RunnableStatus.SKIP.value
