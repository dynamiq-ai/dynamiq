import uuid
from unittest.mock import patch

import pytest
from litellm import APIError, AuthenticationError, BadRequestError, RateLimitError

from dynamiq import connections
from dynamiq.callbacks import TracingCallbackHandler
from dynamiq.nodes.embedders import VertexAIDocumentEmbedder, VertexAITextEmbedder
from dynamiq.runnables import RunnableConfig, RunnableStatus
from tests.integration.nodes.embedders.conftest import (
    assert_embedder_failure,
    assert_embedder_success,
    create_document_embedder_workflow,
    create_text_embedder_workflow,
)


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
    return create_text_embedder_workflow(vertexai_text_embedder)


@pytest.fixture
def vertexai_document_embedder_workflow(vertexai_document_embedder):
    return create_document_embedder_workflow(vertexai_document_embedder)


def test_workflow_with_vertexai_text_embedder(
    mock_embedding_executor, vertexai_text_embedder_workflow, query_input, vertexai_model, vertexai_connection
):
    workflow, embedder, output_node = vertexai_text_embedder_workflow

    response = workflow.run(
        input_data=query_input,
        config=RunnableConfig(callbacks=[TracingCallbackHandler()]),
    )

    assert_embedder_success(response, embedder, output_node)
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

    assert_embedder_success(response, embedder, output_node)
    mock_embedding_executor.assert_called_once_with(
        task_type="RETRIEVAL_DOCUMENT",
        input=[document_input["documents"][0].content],
        model=vertexai_model,
        **vertexai_connection.conn_params,
    )


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

        assert_embedder_failure(response, embedder, output_node, expected_type, error_msg)
        mock_embedding.assert_called_once_with(
            task_type="RETRIEVAL_QUERY",
            input=["Test query"],
            model=embedder.model,
            **embedder.connection.conn_params,
        )


def test_text_embedder_missing_input(vertexai_text_embedder_workflow, missing_input):
    workflow, embedder, output_node = vertexai_text_embedder_workflow
    response = workflow.run(input_data=missing_input)
    assert_embedder_failure(response, embedder, output_node)


def test_text_embedder_empty_input(vertexai_text_embedder_workflow, empty_query_input):
    workflow, embedder, output_node = vertexai_text_embedder_workflow
    response = workflow.run(input_data=empty_query_input)
    assert_embedder_failure(response, embedder, output_node)


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
        assert_embedder_failure(response, embedder, output_node, expected_type, error_msg)
        mock_embedding.assert_called_once_with(
            task_type="RETRIEVAL_DOCUMENT",
            input=[document_input["documents"][0].content],
            model=embedder.model,
            **embedder.connection.conn_params,
        )


def test_document_embedder_missing_input(vertexai_document_embedder_workflow, missing_input):
    workflow, embedder, output_node = vertexai_document_embedder_workflow
    response = workflow.run(input_data=missing_input)
    assert_embedder_failure(response, embedder, output_node)


def test_document_embedder_empty_document_list(vertexai_document_embedder_workflow, empty_documents_input):
    workflow, embedder, output_node = vertexai_document_embedder_workflow
    response = workflow.run(input_data=empty_documents_input)

    assert response.status == RunnableStatus.SUCCESS

    embedder_result = response.output[embedder.id]
    assert embedder_result["status"] == RunnableStatus.SUCCESS.value
    assert "documents" in embedder_result["output"]
    assert len(embedder_result["output"]["documents"]) == 0

    output_result = response.output[output_node.id]
    assert output_result["status"] == RunnableStatus.SUCCESS.value


def test_document_embedder_empty_content(vertexai_document_embedder_workflow, empty_document_content_input):
    workflow, embedder, output_node = vertexai_document_embedder_workflow
    response = workflow.run(input_data=empty_document_content_input)
    assert_embedder_failure(response, embedder, output_node)


def test_text_embedder_api_returns_empty_embedding(
    vertexai_text_embedder_workflow, query_input, empty_embedding_response_factory, vertexai_model
):
    workflow, embedder, output_node = vertexai_text_embedder_workflow
    empty_response = empty_embedding_response_factory(vertexai_model)

    with patch("dynamiq.components.embedders.base.BaseEmbedder._embedding") as mock_embedding:
        mock_embedding.return_value = empty_response
        response = workflow.run(input_data=query_input)
        assert response.status == RunnableStatus.SUCCESS
        embedder_result = response.output[embedder.id]
        assert embedder_result["status"] == RunnableStatus.FAILURE.value
        assert "Invalid embedding" in embedder_result["error"]["message"]


def test_document_embedder_api_returns_empty_embedding(
    vertexai_document_embedder_workflow, document_input, empty_embedding_response_factory, vertexai_model
):
    workflow, embedder, output_node = vertexai_document_embedder_workflow
    empty_response = empty_embedding_response_factory(vertexai_model)

    with patch("dynamiq.components.embedders.base.BaseEmbedder._embedding") as mock_embedding:
        mock_embedding.return_value = empty_response
        response = workflow.run(input_data=document_input)
        assert response.status == RunnableStatus.SUCCESS
        embedder_result = response.output[embedder.id]
        assert embedder_result["status"] == RunnableStatus.FAILURE.value
        assert "Embedding is empty" in embedder_result["error"]["message"]


def test_text_embedder_api_returns_null_embedding(
    vertexai_text_embedder_workflow, query_input, null_embedding_response_factory, vertexai_model
):
    workflow, embedder, output_node = vertexai_text_embedder_workflow
    null_response = null_embedding_response_factory(vertexai_model)

    with patch("dynamiq.components.embedders.base.BaseEmbedder._embedding") as mock_embedding:
        mock_embedding.return_value = null_response
        response = workflow.run(input_data=query_input)
        assert response.status == RunnableStatus.SUCCESS
        embedder_result = response.output[embedder.id]
        assert embedder_result["status"] == RunnableStatus.FAILURE.value
        assert "Invalid embedding" in embedder_result["error"]["message"]


def test_document_embedder_api_returns_null_embedding(
    vertexai_document_embedder_workflow, document_input, null_embedding_response_factory, vertexai_model
):
    workflow, embedder, output_node = vertexai_document_embedder_workflow
    null_response = null_embedding_response_factory(vertexai_model)

    with patch("dynamiq.components.embedders.base.BaseEmbedder._embedding") as mock_embedding:
        mock_embedding.return_value = null_response
        response = workflow.run(input_data=document_input)
        assert response.status == RunnableStatus.SUCCESS
        embedder_result = response.output[embedder.id]
        assert embedder_result["status"] == RunnableStatus.FAILURE.value
        assert "has no embedding" in embedder_result["error"]["message"]


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
        assert_embedder_failure(response, embedder, output_node, "BadRequestError", "maximum input size")


def test_document_embedder_max_tokens_error(
    vertexai_document_embedder_workflow, long_document_input, max_tokens_error_message, vertexai_model
):
    workflow, embedder, output_node = vertexai_document_embedder_workflow

    with patch("dynamiq.components.embedders.base.BaseEmbedder._embedding") as mock_embedding:
        error = BadRequestError(max_tokens_error_message, vertexai_model, "vertex_ai")
        mock_embedding.side_effect = error
        response = workflow.run(input_data=long_document_input)
        assert_embedder_failure(response, embedder, output_node, "BadRequestError", "maximum input size")


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
        assert_embedder_failure(response, embedder, output_node, "BadRequestError", "not found")
