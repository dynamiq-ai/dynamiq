import uuid
from unittest.mock import patch

import pytest
from litellm import APIError, AuthenticationError, BadRequestError, RateLimitError

from dynamiq import connections
from dynamiq.callbacks import TracingCallbackHandler
from dynamiq.nodes.embedders import CohereDocumentEmbedder, CohereTextEmbedder
from dynamiq.runnables import RunnableConfig, RunnableStatus
from tests.integration.nodes.embedders.conftest import (
    assert_embedder_failure,
    assert_embedder_success,
    create_document_embedder_workflow,
    create_text_embedder_workflow,
)


@pytest.fixture
def cohere_connection():
    return connections.Cohere(
        id=str(uuid.uuid4()),
        api_key="api_key",
    )


@pytest.fixture
def cohere_model():
    return "cohere/embed-english-v2.0"


@pytest.fixture
def cohere_text_embedder(cohere_connection, cohere_model):
    return CohereTextEmbedder(
        id="text_embedder", name="CohereTextEmbedder", connection=cohere_connection, model=cohere_model
    )


@pytest.fixture
def cohere_document_embedder(cohere_connection, cohere_model):
    return CohereDocumentEmbedder(
        id="document_embedder", name="CohereDocumentEmbedder", connection=cohere_connection, model=cohere_model
    )


@pytest.fixture
def cohere_text_embedder_workflow(cohere_text_embedder):
    return create_text_embedder_workflow(cohere_text_embedder)


@pytest.fixture
def cohere_document_embedder_workflow(cohere_document_embedder):
    return create_document_embedder_workflow(cohere_document_embedder)


def test_workflow_with_cohere_text_embedder(
    mock_embedding_executor, cohere_text_embedder_workflow, query_input, cohere_model
):
    workflow, embedder, output_node = cohere_text_embedder_workflow

    response = workflow.run(
        input_data=query_input,
        config=RunnableConfig(callbacks=[TracingCallbackHandler()]),
    )

    assert_embedder_success(response, embedder, output_node)
    mock_embedding_executor.assert_called_once_with(
        input_type="search_query",
        input=[query_input["query"]],
        model=cohere_model,
        api_key="api_key",
    )


def test_workflow_with_cohere_document_embedder(
    mock_embedding_executor, cohere_document_embedder_workflow, document_input, cohere_model
):
    workflow, embedder, output_node = cohere_document_embedder_workflow

    response = workflow.run(
        input_data=document_input,
        config=RunnableConfig(callbacks=[TracingCallbackHandler()]),
    )

    assert_embedder_success(response, embedder, output_node)
    mock_embedding_executor.assert_called_once_with(
        input=[document_input["documents"][0].content],
        input_type="search_query",
        model=cohere_model,
        api_key="api_key",
    )


@pytest.mark.parametrize(
    "error_class,error_msg,error_args,expected_type",
    [
        (AuthenticationError, "Invalid API key", ["cohere", "cohere/embed-english-v2.0"], "AuthenticationError"),
        (RateLimitError, "Rate limit exceeded", ["cohere", "cohere/embed-english-v2.0"], "RateLimitError"),
        (APIError, "Service unavailable", [500, "cohere", "cohere/embed-english-v2.0"], "APIError"),
        (BadRequestError, "Invalid embedding model", ["non-existent-model", "cohere"], "BadRequestError"),
    ],
)
def test_text_embedder_api_errors(cohere_text_embedder_workflow, error_class, error_msg, error_args, expected_type):
    workflow, embedder, output_node = cohere_text_embedder_workflow

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
            model=embedder.model, input=["Test query"], input_type="search_query", api_key=embedder.connection.api_key
        )


def test_text_embedder_missing_input(cohere_text_embedder_workflow, missing_input):
    workflow, embedder, output_node = cohere_text_embedder_workflow
    response = workflow.run(input_data=missing_input)
    assert_embedder_failure(response, embedder, output_node)


def test_text_embedder_empty_input(cohere_text_embedder_workflow, empty_query_input):
    workflow, embedder, output_node = cohere_text_embedder_workflow
    response = workflow.run(input_data=empty_query_input)
    assert_embedder_failure(response, embedder, output_node)


@pytest.mark.parametrize(
    "error_class,error_msg,error_args,expected_type",
    [
        (AuthenticationError, "Invalid API key", ["cohere", "cohere/embed-english-v2.0"], "AuthenticationError"),
        (RateLimitError, "Rate limit exceeded", ["cohere", "cohere/embed-english-v2.0"], "RateLimitError"),
        (APIError, "Service unavailable", [500, "cohere", "cohere/embed-english-v2.0"], "APIError"),
        (BadRequestError, "Invalid embedding model", ["non-existent-model", "cohere"], "BadRequestError"),
    ],
)
def test_document_embedder_api_errors(
    cohere_document_embedder_workflow, document_input, error_class, error_msg, error_args, expected_type
):
    workflow, embedder, output_node = cohere_document_embedder_workflow

    with patch("dynamiq.components.embedders.base.BaseEmbedder._embedding") as mock_embedding:
        if error_class == APIError:
            error = error_class(error_args[0], error_msg, error_args[1], error_args[2])
        else:
            error = error_class(error_msg, *error_args)

        mock_embedding.side_effect = error
        response = workflow.run(input_data=document_input)
        assert_embedder_failure(response, embedder, output_node, expected_type, error_msg)
        mock_embedding.assert_called_once_with(
            model=embedder.model,
            input=[document_input["documents"][0].content],
            input_type="search_query",
            api_key=embedder.connection.api_key,
        )


def test_document_embedder_missing_input(cohere_document_embedder_workflow, missing_input):
    workflow, embedder, output_node = cohere_document_embedder_workflow
    response = workflow.run(input_data=missing_input)
    assert_embedder_failure(response, embedder, output_node)


def test_document_embedder_empty_document_list(cohere_document_embedder_workflow, empty_documents_input):
    workflow, embedder, output_node = cohere_document_embedder_workflow
    response = workflow.run(input_data=empty_documents_input)

    assert response.status == RunnableStatus.SUCCESS

    embedder_result = response.output[embedder.id]
    assert embedder_result["status"] == RunnableStatus.SUCCESS.value
    assert "documents" in embedder_result["output"]
    assert len(embedder_result["output"]["documents"]) == 0

    output_result = response.output[output_node.id]
    assert output_result["status"] == RunnableStatus.SUCCESS.value


def test_document_embedder_empty_content(cohere_document_embedder_workflow, empty_document_content_input):
    workflow, embedder, output_node = cohere_document_embedder_workflow
    response = workflow.run(input_data=empty_document_content_input)
    assert_embedder_failure(response, embedder, output_node)


def test_text_embedder_api_returns_empty_embedding(
    cohere_text_embedder_workflow, query_input, empty_embedding_response_factory, cohere_model
):
    workflow, embedder, output_node = cohere_text_embedder_workflow
    empty_response = empty_embedding_response_factory(cohere_model)

    with patch("dynamiq.components.embedders.base.BaseEmbedder._embedding") as mock_embedding:
        mock_embedding.return_value = empty_response
        response = workflow.run(input_data=query_input)
        assert response.status == RunnableStatus.SUCCESS
        embedder_result = response.output[embedder.id]
        assert embedder_result["status"] == RunnableStatus.FAILURE.value
        assert "Invalid embedding" in embedder_result["error"]["message"]


def test_document_embedder_api_returns_empty_embedding(
    cohere_document_embedder_workflow, document_input, empty_embedding_response_factory, cohere_model
):
    workflow, embedder, output_node = cohere_document_embedder_workflow
    empty_response = empty_embedding_response_factory(cohere_model)

    with patch("dynamiq.components.embedders.base.BaseEmbedder._embedding") as mock_embedding:
        mock_embedding.return_value = empty_response
        response = workflow.run(input_data=document_input)
        assert response.status == RunnableStatus.SUCCESS
        embedder_result = response.output[embedder.id]
        assert embedder_result["status"] == RunnableStatus.FAILURE.value
        assert "Embedding is empty" in embedder_result["error"]["message"]


def test_text_embedder_api_returns_null_embedding(
    cohere_text_embedder_workflow, query_input, null_embedding_response_factory, cohere_model
):
    workflow, embedder, output_node = cohere_text_embedder_workflow
    null_response = null_embedding_response_factory(cohere_model)

    with patch("dynamiq.components.embedders.base.BaseEmbedder._embedding") as mock_embedding:
        mock_embedding.return_value = null_response
        response = workflow.run(input_data=query_input)
        assert response.status == RunnableStatus.SUCCESS
        embedder_result = response.output[embedder.id]
        assert embedder_result["status"] == RunnableStatus.FAILURE.value
        assert "Invalid embedding" in embedder_result["error"]["message"]


def test_document_embedder_api_returns_null_embedding(
    cohere_document_embedder_workflow, document_input, null_embedding_response_factory, cohere_model
):
    workflow, embedder, output_node = cohere_document_embedder_workflow
    null_response = null_embedding_response_factory(cohere_model)

    with patch("dynamiq.components.embedders.base.BaseEmbedder._embedding") as mock_embedding:
        mock_embedding.return_value = null_response
        response = workflow.run(input_data=document_input)
        assert response.status == RunnableStatus.SUCCESS
        embedder_result = response.output[embedder.id]
        assert embedder_result["status"] == RunnableStatus.FAILURE.value
        assert "has no embedding" in embedder_result["error"]["message"]


@pytest.fixture
def max_tokens_error_message():
    return "Token limit exceeded. Max limit is 8192 tokens."


def test_text_embedder_max_tokens_error(
    cohere_text_embedder_workflow, long_query_input, max_tokens_error_message, cohere_model
):
    workflow, embedder, output_node = cohere_text_embedder_workflow

    with patch("dynamiq.components.embedders.base.BaseEmbedder._embedding") as mock_embedding:
        error = BadRequestError(max_tokens_error_message, cohere_model, "cohere")
        mock_embedding.side_effect = error
        response = workflow.run(input_data=long_query_input)
        assert_embedder_failure(response, embedder, output_node, "BadRequestError", "Token limit")


def test_document_embedder_max_tokens_error(
    cohere_document_embedder_workflow, long_document_input, max_tokens_error_message, cohere_model
):
    workflow, embedder, output_node = cohere_document_embedder_workflow

    with patch("dynamiq.components.embedders.base.BaseEmbedder._embedding") as mock_embedding:
        error = BadRequestError(max_tokens_error_message, cohere_model, "cohere")
        mock_embedding.side_effect = error
        response = workflow.run(input_data=long_document_input)
        assert_embedder_failure(response, embedder, output_node, "BadRequestError", "Token limit")


@pytest.fixture
def invalid_model_error_message():
    return "Invalid model specified"


def test_text_embedder_invalid_model(
    cohere_text_embedder_workflow, query_input, invalid_model_error_message, cohere_model
):
    workflow, embedder, output_node = cohere_text_embedder_workflow

    with patch("dynamiq.components.embedders.base.BaseEmbedder._embedding") as mock_embedding:
        error = BadRequestError(invalid_model_error_message, cohere_model, "cohere")
        mock_embedding.side_effect = error
        response = workflow.run(input_data=query_input)
        assert_embedder_failure(response, embedder, output_node, "BadRequestError", "Invalid model")
