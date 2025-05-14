import uuid
from unittest.mock import ANY, patch

import pytest
from litellm import APIError, AuthenticationError, BadRequestError, RateLimitError

from dynamiq import connections
from dynamiq.callbacks import TracingCallbackHandler
from dynamiq.nodes.embedders import OpenAIDocumentEmbedder, OpenAITextEmbedder
from dynamiq.runnables import RunnableConfig, RunnableStatus
from tests.integration.nodes.embedders.conftest import (
    assert_embedder_failure,
    assert_embedder_success,
    create_document_embedder_workflow,
    create_text_embedder_workflow,
)


@pytest.fixture
def openai_connection():
    return connections.OpenAI(
        id=str(uuid.uuid4()),
        api_key="api_key",
    )


@pytest.fixture
def openai_model():
    return "text-embedding-3-small"


@pytest.fixture
def openai_text_embedder(openai_connection, openai_model):
    return OpenAITextEmbedder(
        id="text_embedder", name="OpenAITextEmbedder", connection=openai_connection, model=openai_model
    )


@pytest.fixture
def openai_document_embedder(openai_connection, openai_model):
    return OpenAIDocumentEmbedder(
        id="document_embedder", name="OpenAIDocumentEmbedder", connection=openai_connection, model=openai_model
    )


@pytest.fixture
def openai_text_embedder_workflow(openai_text_embedder):
    return create_text_embedder_workflow(openai_text_embedder)


@pytest.fixture
def openai_document_embedder_workflow(openai_document_embedder):
    return create_document_embedder_workflow(openai_document_embedder)


def test_workflow_with_openai_text_embedder(
    mock_embedding_executor, openai_text_embedder_workflow, query_input, openai_model
):
    workflow, embedder, output_node = openai_text_embedder_workflow

    response = workflow.run(
        input_data=query_input,
        config=RunnableConfig(callbacks=[TracingCallbackHandler()]),
    )

    assert_embedder_success(response, embedder, output_node)
    mock_embedding_executor.assert_called_once_with(
        input=[query_input["query"]],
        model=openai_model,
        client=ANY,
    )


def test_workflow_with_openai_document_embedder(
    mock_embedding_executor, openai_document_embedder_workflow, document_input, openai_model
):
    workflow, embedder, output_node = openai_document_embedder_workflow

    response = workflow.run(
        input_data=document_input,
        config=RunnableConfig(callbacks=[TracingCallbackHandler()]),
    )

    assert_embedder_success(response, embedder, output_node)
    mock_embedding_executor.assert_called_once_with(
        input=[document_input["documents"][0].content],
        model=openai_model,
        client=ANY,
    )


@pytest.mark.parametrize(
    "error_class,error_msg,error_args,expected_type",
    [
        (AuthenticationError, "Invalid API key", ["openai", "text-embedding-3-small"], "AuthenticationError"),
        (RateLimitError, "Rate limit exceeded", ["openai", "text-embedding-3-small"], "RateLimitError"),
        (APIError, "Service unavailable", [500, "openai", "text-embedding-3-small"], "APIError"),
        (BadRequestError, "Invalid embedding model", ["non-existent-model", "openai"], "BadRequestError"),
    ],
)
def test_text_embedder_api_errors(openai_text_embedder_workflow, error_class, error_msg, error_args, expected_type):
    workflow, embedder, output_node = openai_text_embedder_workflow

    with patch("dynamiq.components.embedders.base.BaseEmbedder._embedding") as mock_embedding:
        if error_class == APIError:
            error = error_class(error_args[0], error_msg, error_args[1], error_args[2])
        else:
            error = error_class(error_msg, *error_args)

        mock_embedding.side_effect = error

        input_data = {"query": "Test query"}
        response = workflow.run(input_data=input_data)

        assert_embedder_failure(response, embedder, output_node, expected_type, error_msg)


def test_text_embedder_missing_input(openai_text_embedder_workflow, missing_input):
    workflow, embedder, output_node = openai_text_embedder_workflow
    response = workflow.run(input_data=missing_input)
    assert_embedder_failure(response, embedder, output_node)


def test_text_embedder_empty_input(openai_text_embedder_workflow, empty_query_input):
    workflow, embedder, output_node = openai_text_embedder_workflow
    response = workflow.run(input_data=empty_query_input)
    assert_embedder_failure(response, embedder, output_node)


@pytest.mark.parametrize(
    "error_class,error_msg,error_args,expected_type",
    [
        (AuthenticationError, "Invalid API key", ["openai", "text-embedding-3-small"], "AuthenticationError"),
        (RateLimitError, "Rate limit exceeded", ["openai", "text-embedding-3-small"], "RateLimitError"),
        (APIError, "Service unavailable", [500, "openai", "text-embedding-3-small"], "APIError"),
        (BadRequestError, "Invalid embedding model", ["non-existent-model", "openai"], "BadRequestError"),
    ],
)
def test_document_embedder_api_errors(
    openai_document_embedder_workflow, document_input, error_class, error_msg, error_args, expected_type
):
    workflow, embedder, output_node = openai_document_embedder_workflow

    with patch("dynamiq.components.embedders.base.BaseEmbedder._embedding") as mock_embedding:
        if error_class == APIError:
            error = error_class(error_args[0], error_msg, error_args[1], error_args[2])
        else:
            error = error_class(error_msg, *error_args)

        mock_embedding.side_effect = error
        response = workflow.run(input_data=document_input)
        assert_embedder_failure(response, embedder, output_node, expected_type, error_msg)


def test_document_embedder_missing_input(openai_document_embedder_workflow, missing_input):
    workflow, embedder, output_node = openai_document_embedder_workflow
    response = workflow.run(input_data=missing_input)
    assert_embedder_failure(response, embedder, output_node)


def test_document_embedder_empty_document_list(openai_document_embedder_workflow, empty_documents_input):
    workflow, embedder, output_node = openai_document_embedder_workflow
    response = workflow.run(input_data=empty_documents_input)

    assert response.status == RunnableStatus.SUCCESS

    embedder_result = response.output[embedder.id]
    assert embedder_result["status"] == RunnableStatus.SUCCESS.value
    assert "documents" in embedder_result["output"]
    assert len(embedder_result["output"]["documents"]) == 0

    output_result = response.output[output_node.id]
    assert output_result["status"] == RunnableStatus.SUCCESS.value


def test_document_embedder_empty_content(openai_document_embedder_workflow, empty_document_content_input):
    workflow, embedder, output_node = openai_document_embedder_workflow
    response = workflow.run(input_data=empty_document_content_input)
    assert_embedder_failure(response, embedder, output_node)


def test_text_embedder_api_returns_empty_embedding(
    openai_text_embedder_workflow, query_input, empty_embedding_response_factory, openai_model
):
    workflow, embedder, output_node = openai_text_embedder_workflow
    empty_response = empty_embedding_response_factory(openai_model)

    with patch("dynamiq.components.embedders.base.BaseEmbedder._embedding") as mock_embedding:
        mock_embedding.return_value = empty_response
        response = workflow.run(input_data=query_input)
        assert response.status == RunnableStatus.SUCCESS
        embedder_result = response.output[embedder.id]
        assert embedder_result["status"] == RunnableStatus.FAILURE.value
        assert "Invalid embedding" in embedder_result["error"]["message"]


def test_document_embedder_api_returns_empty_embedding(
    openai_document_embedder_workflow, document_input, empty_embedding_response_factory, openai_model
):
    workflow, embedder, output_node = openai_document_embedder_workflow
    empty_response = empty_embedding_response_factory(openai_model)

    with patch("dynamiq.components.embedders.base.BaseEmbedder._embedding") as mock_embedding:
        mock_embedding.return_value = empty_response
        response = workflow.run(input_data=document_input)
        assert response.status == RunnableStatus.SUCCESS
        embedder_result = response.output[embedder.id]
        assert embedder_result["status"] == RunnableStatus.FAILURE.value
        assert "Embedding is empty" in embedder_result["error"]["message"]


def test_text_embedder_api_returns_null_embedding(
    openai_text_embedder_workflow, query_input, null_embedding_response_factory, openai_model
):
    workflow, embedder, output_node = openai_text_embedder_workflow
    null_response = null_embedding_response_factory(openai_model)

    with patch("dynamiq.components.embedders.base.BaseEmbedder._embedding") as mock_embedding:
        mock_embedding.return_value = null_response
        response = workflow.run(input_data=query_input)
        assert response.status == RunnableStatus.SUCCESS
        embedder_result = response.output[embedder.id]
        assert embedder_result["status"] == RunnableStatus.FAILURE.value
        assert "Invalid embedding" in embedder_result["error"]["message"]


def test_document_embedder_api_returns_null_embedding(
    openai_document_embedder_workflow, document_input, null_embedding_response_factory, openai_model
):
    workflow, embedder, output_node = openai_document_embedder_workflow
    null_response = null_embedding_response_factory(openai_model)

    with patch("dynamiq.components.embedders.base.BaseEmbedder._embedding") as mock_embedding:
        mock_embedding.return_value = null_response
        response = workflow.run(input_data=document_input)
        assert response.status == RunnableStatus.SUCCESS
        embedder_result = response.output[embedder.id]
        assert embedder_result["status"] == RunnableStatus.FAILURE.value
        assert "has no embedding" in embedder_result["error"]["message"]


@pytest.fixture
def max_tokens_error_message():
    return "This model's maximum context length is 8191 tokens, but the provided inputs have 10000 tokens"


def test_text_embedder_max_tokens_error(
    openai_text_embedder_workflow, long_query_input, max_tokens_error_message, openai_model
):
    workflow, embedder, output_node = openai_text_embedder_workflow

    with patch("dynamiq.components.embedders.base.BaseEmbedder._embedding") as mock_embedding:
        error = BadRequestError(max_tokens_error_message, openai_model, "openai")
        mock_embedding.side_effect = error
        response = workflow.run(input_data=long_query_input)
        assert_embedder_failure(response, embedder, output_node, "BadRequestError", "maximum context length")


def test_document_embedder_max_tokens_error(
    openai_document_embedder_workflow, long_document_input, max_tokens_error_message, openai_model
):
    workflow, embedder, output_node = openai_document_embedder_workflow

    with patch("dynamiq.components.embedders.base.BaseEmbedder._embedding") as mock_embedding:
        error = BadRequestError(max_tokens_error_message, openai_model, "openai")
        mock_embedding.side_effect = error
        response = workflow.run(input_data=long_document_input)
        assert_embedder_failure(response, embedder, output_node, "BadRequestError", "maximum context length")


@pytest.fixture
def invalid_dimensions_error_message():
    return "Invalid dimensions parameter: must be between 1 and 1536"


def test_text_embedder_invalid_params(
    openai_text_embedder_workflow, query_input, invalid_dimensions_error_message, openai_model
):
    workflow, embedder, output_node = openai_text_embedder_workflow

    with patch("dynamiq.components.embedders.base.BaseEmbedder._embedding") as mock_embedding:
        error = BadRequestError(invalid_dimensions_error_message, openai_model, "openai")
        mock_embedding.side_effect = error
        response = workflow.run(input_data=query_input)
        assert_embedder_failure(response, embedder, output_node, "BadRequestError", "Invalid dimensions")
