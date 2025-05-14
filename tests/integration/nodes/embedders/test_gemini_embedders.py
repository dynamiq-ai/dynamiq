import uuid
from unittest.mock import patch

import pytest
from litellm import APIError, AuthenticationError, BadRequestError, RateLimitError

from dynamiq import connections
from dynamiq.callbacks import TracingCallbackHandler
from dynamiq.nodes.embedders import GeminiDocumentEmbedder, GeminiTextEmbedder
from dynamiq.runnables import RunnableConfig, RunnableStatus
from tests.integration.nodes.embedders.conftest import (
    assert_embedder_failure,
    assert_embedder_success,
    create_document_embedder_workflow,
    create_text_embedder_workflow,
)


@pytest.fixture
def gemini_connection():
    return connections.Gemini(
        id=str(uuid.uuid4()),
        api_key="api_key",
    )


@pytest.fixture
def gemini_model():
    return "gemini/gemini-embedding-exp-03-07"


@pytest.fixture
def gemini_text_embedder(gemini_connection, gemini_model):
    return GeminiTextEmbedder(
        id="text_embedder",
        name="GeminiTextEmbedder",
        connection=gemini_connection,
        model=gemini_model,
        task_type="RETRIEVAL_QUERY",
    )


@pytest.fixture
def gemini_document_embedder(gemini_connection, gemini_model):
    return GeminiDocumentEmbedder(
        id="document_embedder",
        name="GeminiDocumentEmbedder",
        connection=gemini_connection,
        model=gemini_model,
        task_type="RETRIEVAL_DOCUMENT",
    )


@pytest.fixture
def gemini_text_embedder_workflow(gemini_text_embedder):
    return create_text_embedder_workflow(gemini_text_embedder)


@pytest.fixture
def gemini_document_embedder_workflow(gemini_document_embedder):
    return create_document_embedder_workflow(gemini_document_embedder)


def test_workflow_with_gemini_text_embedder(
    mock_embedding_executor, gemini_text_embedder_workflow, query_input, gemini_model
):
    workflow, embedder, output_node = gemini_text_embedder_workflow

    response = workflow.run(
        input_data=query_input,
        config=RunnableConfig(callbacks=[TracingCallbackHandler()]),
    )

    assert_embedder_success(response, embedder, output_node)
    mock_embedding_executor.assert_called_once_with(
        input=[query_input["query"]],
        model=gemini_model,
        task_type="RETRIEVAL_QUERY",
        api_key="api_key",
    )


def test_workflow_with_gemini_document_embedder(
    mock_embedding_executor, gemini_document_embedder_workflow, document_input, gemini_model
):
    workflow, embedder, output_node = gemini_document_embedder_workflow

    response = workflow.run(
        input_data=document_input,
        config=RunnableConfig(callbacks=[TracingCallbackHandler()]),
    )

    assert_embedder_success(response, embedder, output_node)
    mock_embedding_executor.assert_called_once_with(
        input=[document_input["documents"][0].content],
        model=gemini_model,
        task_type="RETRIEVAL_DOCUMENT",
        api_key="api_key",
    )


@pytest.mark.parametrize(
    "error_class,error_msg,error_args,expected_type",
    [
        (
            AuthenticationError,
            "Invalid API key",
            ["gemini", "gemini/gemini-embedding-exp-03-07"],
            "AuthenticationError",
        ),
        (RateLimitError, "Rate limit exceeded", ["gemini", "gemini/gemini-embedding-exp-03-07"], "RateLimitError"),
        (APIError, "Service unavailable", [500, "gemini", "gemini/gemini-embedding-exp-03-07"], "APIError"),
        (BadRequestError, "Invalid embedding model", ["non-existent-model", "gemini"], "BadRequestError"),
    ],
)
def test_text_embedder_api_errors(gemini_text_embedder_workflow, error_class, error_msg, error_args, expected_type):
    workflow, embedder, output_node = gemini_text_embedder_workflow

    with patch("dynamiq.components.embedders.base.BaseEmbedder._embedding") as mock_embedding:
        if error_class == APIError:
            error = error_class(error_args[0], error_msg, error_args[1], error_args[2])
        else:
            error = error_class(error_msg, *error_args)

        mock_embedding.side_effect = error

        input_data = {"query": "Test query"}
        response = workflow.run(input_data=input_data)

        assert_embedder_failure(response, embedder, output_node, expected_type, error_msg)


def test_text_embedder_missing_input(gemini_text_embedder_workflow, missing_input):
    workflow, embedder, output_node = gemini_text_embedder_workflow
    response = workflow.run(input_data=missing_input)
    assert_embedder_failure(response, embedder, output_node)


def test_text_embedder_empty_input(gemini_text_embedder_workflow, empty_query_input):
    workflow, embedder, output_node = gemini_text_embedder_workflow
    response = workflow.run(input_data=empty_query_input)
    assert_embedder_failure(response, embedder, output_node)


@pytest.mark.parametrize(
    "error_class,error_msg,error_args,expected_type",
    [
        (
            AuthenticationError,
            "Invalid API key",
            ["gemini", "gemini/gemini-embedding-exp-03-07"],
            "AuthenticationError",
        ),
        (RateLimitError, "Rate limit exceeded", ["gemini", "gemini/gemini-embedding-exp-03-07"], "RateLimitError"),
        (APIError, "Service unavailable", [500, "gemini", "gemini/gemini-embedding-exp-03-07"], "APIError"),
        (BadRequestError, "Invalid embedding model", ["non-existent-model", "gemini"], "BadRequestError"),
    ],
)
def test_document_embedder_api_errors(
    gemini_document_embedder_workflow, document_input, error_class, error_msg, error_args, expected_type
):
    workflow, embedder, output_node = gemini_document_embedder_workflow

    with patch("dynamiq.components.embedders.base.BaseEmbedder._embedding") as mock_embedding:
        if error_class == APIError:
            error = error_class(error_args[0], error_msg, error_args[1], error_args[2])
        else:
            error = error_class(error_msg, *error_args)

        mock_embedding.side_effect = error
        response = workflow.run(input_data=document_input)
        assert_embedder_failure(response, embedder, output_node, expected_type, error_msg)


def test_document_embedder_missing_input(gemini_document_embedder_workflow, missing_input):
    workflow, embedder, output_node = gemini_document_embedder_workflow
    response = workflow.run(input_data=missing_input)
    assert_embedder_failure(response, embedder, output_node)


def test_document_embedder_empty_document_list(gemini_document_embedder_workflow, empty_documents_input):
    workflow, embedder, output_node = gemini_document_embedder_workflow
    response = workflow.run(input_data=empty_documents_input)

    assert response.status == RunnableStatus.SUCCESS

    embedder_result = response.output[embedder.id]
    assert embedder_result["status"] == RunnableStatus.SUCCESS.value
    assert "documents" in embedder_result["output"]
    assert len(embedder_result["output"]["documents"]) == 0

    output_result = response.output[output_node.id]
    assert output_result["status"] == RunnableStatus.SUCCESS.value


def test_document_embedder_empty_content(gemini_document_embedder_workflow, empty_document_content_input):
    workflow, embedder, output_node = gemini_document_embedder_workflow
    response = workflow.run(input_data=empty_document_content_input)
    assert_embedder_failure(response, embedder, output_node)


def test_text_embedder_api_returns_empty_embedding(
    gemini_text_embedder_workflow, query_input, empty_embedding_response_factory, gemini_model
):
    workflow, embedder, output_node = gemini_text_embedder_workflow
    empty_response = empty_embedding_response_factory(gemini_model)

    with patch("dynamiq.components.embedders.base.BaseEmbedder._embedding") as mock_embedding:
        mock_embedding.return_value = empty_response
        response = workflow.run(input_data=query_input)
        assert response.status == RunnableStatus.SUCCESS
        embedder_result = response.output[embedder.id]
        assert embedder_result["status"] == RunnableStatus.FAILURE.value
        assert "Invalid embedding" in embedder_result["error"]["message"]


def test_document_embedder_api_returns_empty_embedding(
    gemini_document_embedder_workflow, document_input, empty_embedding_response_factory, gemini_model
):
    workflow, embedder, output_node = gemini_document_embedder_workflow
    empty_response = empty_embedding_response_factory(gemini_model)

    with patch("dynamiq.components.embedders.base.BaseEmbedder._embedding") as mock_embedding:
        mock_embedding.return_value = empty_response
        response = workflow.run(input_data=document_input)
        assert response.status == RunnableStatus.SUCCESS
        embedder_result = response.output[embedder.id]
        assert embedder_result["status"] == RunnableStatus.FAILURE.value
        assert "Embedding is empty" in embedder_result["error"]["message"]


def test_text_embedder_api_returns_null_embedding(
    gemini_text_embedder_workflow, query_input, null_embedding_response_factory, gemini_model
):
    workflow, embedder, output_node = gemini_text_embedder_workflow
    null_response = null_embedding_response_factory(gemini_model)

    with patch("dynamiq.components.embedders.base.BaseEmbedder._embedding") as mock_embedding:
        mock_embedding.return_value = null_response
        response = workflow.run(input_data=query_input)
        assert response.status == RunnableStatus.SUCCESS
        embedder_result = response.output[embedder.id]
        assert embedder_result["status"] == RunnableStatus.FAILURE.value
        assert "Invalid embedding" in embedder_result["error"]["message"]


def test_document_embedder_api_returns_null_embedding(
    gemini_document_embedder_workflow, document_input, null_embedding_response_factory, gemini_model
):
    workflow, embedder, output_node = gemini_document_embedder_workflow
    null_response = null_embedding_response_factory(gemini_model)

    with patch("dynamiq.components.embedders.base.BaseEmbedder._embedding") as mock_embedding:
        mock_embedding.return_value = null_response
        response = workflow.run(input_data=document_input)
        assert response.status == RunnableStatus.SUCCESS
        embedder_result = response.output[embedder.id]
        assert embedder_result["status"] == RunnableStatus.FAILURE.value
        assert "has no embedding" in embedder_result["error"]["message"]


@pytest.fixture
def max_tokens_error_message():
    return "Content size exceeds maximum of 3072 tokens. Please reduce the content size."


def test_text_embedder_max_tokens_error(
    gemini_text_embedder_workflow, long_query_input, max_tokens_error_message, gemini_model
):
    workflow, embedder, output_node = gemini_text_embedder_workflow

    with patch("dynamiq.components.embedders.base.BaseEmbedder._embedding") as mock_embedding:
        error = BadRequestError(max_tokens_error_message, gemini_model, "gemini")
        mock_embedding.side_effect = error
        response = workflow.run(input_data=long_query_input)
        assert_embedder_failure(response, embedder, output_node, "BadRequestError", "Content size exceeds maximum")


def test_document_embedder_max_tokens_error(
    gemini_document_embedder_workflow, long_document_input, max_tokens_error_message, gemini_model
):
    workflow, embedder, output_node = gemini_document_embedder_workflow

    with patch("dynamiq.components.embedders.base.BaseEmbedder._embedding") as mock_embedding:
        error = BadRequestError(max_tokens_error_message, gemini_model, "gemini")
        mock_embedding.side_effect = error
        response = workflow.run(input_data=long_document_input)
        assert_embedder_failure(response, embedder, output_node, "BadRequestError", "Content size exceeds maximum")


@pytest.fixture
def invalid_model_error_message():
    return "Model invalid-model not found"


def test_text_embedder_invalid_model(
    gemini_text_embedder_workflow, query_input, invalid_model_error_message, gemini_model
):
    workflow, embedder, output_node = gemini_text_embedder_workflow

    with patch("dynamiq.components.embedders.base.BaseEmbedder._embedding") as mock_embedding:
        error = BadRequestError(invalid_model_error_message, gemini_model, "gemini")
        mock_embedding.side_effect = error
        response = workflow.run(input_data=query_input)
        assert_embedder_failure(response, embedder, output_node, "BadRequestError", "Model invalid-model not found")
