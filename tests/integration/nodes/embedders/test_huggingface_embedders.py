import uuid
from unittest.mock import patch

import pytest
from litellm import APIError, AuthenticationError, BadRequestError, RateLimitError

from dynamiq import connections
from dynamiq.callbacks import TracingCallbackHandler
from dynamiq.nodes.embedders import HuggingFaceDocumentEmbedder, HuggingFaceTextEmbedder
from dynamiq.runnables import RunnableConfig, RunnableStatus
from tests.integration.nodes.embedders.conftest import (
    assert_embedder_failure,
    assert_embedder_success,
    create_document_embedder_workflow,
    create_text_embedder_workflow,
)


@pytest.fixture
def huggingface_connection():
    return connections.HuggingFace(
        id=str(uuid.uuid4()),
        api_key="api_key",
    )


@pytest.fixture
def huggingface_model():
    return "BAAI/bge-base-en-v1.5"


@pytest.fixture
def huggingface_text_embedder(huggingface_connection, huggingface_model):
    return HuggingFaceTextEmbedder(
        id="text_embedder", name="HuggingFaceTextEmbedder", connection=huggingface_connection, model=huggingface_model
    )


@pytest.fixture
def huggingface_document_embedder(huggingface_connection, huggingface_model):
    return HuggingFaceDocumentEmbedder(
        id="document_embedder",
        name="HuggingFaceDocumentEmbedder",
        connection=huggingface_connection,
        model=huggingface_model,
    )


@pytest.fixture
def huggingface_text_embedder_workflow(huggingface_text_embedder):
    return create_text_embedder_workflow(huggingface_text_embedder)


@pytest.fixture
def huggingface_document_embedder_workflow(huggingface_document_embedder):
    return create_document_embedder_workflow(huggingface_document_embedder)


def test_workflow_with_huggingface_text_embedder(
    mock_embedding_executor, huggingface_text_embedder_workflow, query_input, huggingface_model
):
    workflow, embedder, output_node = huggingface_text_embedder_workflow

    response = workflow.run(
        input_data=query_input,
        config=RunnableConfig(callbacks=[TracingCallbackHandler()]),
    )

    assert_embedder_success(response, embedder, output_node)
    mock_embedding_executor.assert_called_once_with(
        input=[query_input["query"]],
        model=huggingface_model,
        api_key="api_key",
    )


def test_workflow_with_huggingface_document_embedder(
    mock_embedding_executor, huggingface_document_embedder_workflow, document_input, huggingface_model
):
    workflow, embedder, output_node = huggingface_document_embedder_workflow

    response = workflow.run(
        input_data=document_input,
        config=RunnableConfig(callbacks=[TracingCallbackHandler()]),
    )

    assert_embedder_success(response, embedder, output_node)
    mock_embedding_executor.assert_called_once_with(
        input=[document_input["documents"][0].content],
        model=huggingface_model,
        api_key="api_key",
    )


@pytest.mark.parametrize(
    "error_class,error_msg,error_args,expected_type",
    [
        (AuthenticationError, "Invalid API key", ["huggingface", "BAAI/bge-base-en-v1.5"], "AuthenticationError"),
        (RateLimitError, "Rate limit exceeded", ["huggingface", "BAAI/bge-base-en-v1.5"], "RateLimitError"),
        (APIError, "Service unavailable", [500, "huggingface", "BAAI/bge-base-en-v1.5"], "APIError"),
        (BadRequestError, "Invalid embedding model", ["non-existent-model", "huggingface"], "BadRequestError"),
    ],
)
def test_text_embedder_api_errors(
    huggingface_text_embedder_workflow, error_class, error_msg, error_args, expected_type
):
    workflow, embedder, output_node = huggingface_text_embedder_workflow

    with patch("dynamiq.components.embedders.base.BaseEmbedder._embedding") as mock_embedding:
        if error_class == APIError:
            error = error_class(error_args[0], error_msg, error_args[1], error_args[2])
        else:
            error = error_class(error_msg, *error_args)

        mock_embedding.side_effect = error

        input_data = {"query": "Test query"}
        response = workflow.run(input_data=input_data)

        assert_embedder_failure(response, embedder, output_node, expected_type, error_msg)


def test_text_embedder_missing_input(huggingface_text_embedder_workflow, missing_input):
    workflow, embedder, output_node = huggingface_text_embedder_workflow
    response = workflow.run(input_data=missing_input)
    assert_embedder_failure(response, embedder, output_node)


def test_text_embedder_empty_input(huggingface_text_embedder_workflow, empty_query_input):
    workflow, embedder, output_node = huggingface_text_embedder_workflow
    response = workflow.run(input_data=empty_query_input)
    assert_embedder_failure(response, embedder, output_node)


@pytest.mark.parametrize(
    "error_class,error_msg,error_args,expected_type",
    [
        (AuthenticationError, "Invalid API key", ["huggingface", "BAAI/bge-base-en-v1.5"], "AuthenticationError"),
        (RateLimitError, "Rate limit exceeded", ["huggingface", "BAAI/bge-base-en-v1.5"], "RateLimitError"),
        (APIError, "Service unavailable", [500, "huggingface", "BAAI/bge-base-en-v1.5"], "APIError"),
        (BadRequestError, "Invalid embedding model", ["non-existent-model", "huggingface"], "BadRequestError"),
    ],
)
def test_document_embedder_api_errors(
    huggingface_document_embedder_workflow, document_input, error_class, error_msg, error_args, expected_type
):
    workflow, embedder, output_node = huggingface_document_embedder_workflow

    with patch("dynamiq.components.embedders.base.BaseEmbedder._embedding") as mock_embedding:
        if error_class == APIError:
            error = error_class(error_args[0], error_msg, error_args[1], error_args[2])
        else:
            error = error_class(error_msg, *error_args)

        mock_embedding.side_effect = error
        response = workflow.run(input_data=document_input)
        assert_embedder_failure(response, embedder, output_node, expected_type, error_msg)


def test_document_embedder_missing_input(huggingface_document_embedder_workflow, missing_input):
    workflow, embedder, output_node = huggingface_document_embedder_workflow
    response = workflow.run(input_data=missing_input)
    assert_embedder_failure(response, embedder, output_node)


def test_document_embedder_empty_document_list(huggingface_document_embedder_workflow, empty_documents_input):
    workflow, embedder, output_node = huggingface_document_embedder_workflow
    response = workflow.run(input_data=empty_documents_input)

    assert response.status == RunnableStatus.SUCCESS

    embedder_result = response.output[embedder.id]
    assert embedder_result["status"] == RunnableStatus.SUCCESS.value
    assert "documents" in embedder_result["output"]
    assert len(embedder_result["output"]["documents"]) == 0

    output_result = response.output[output_node.id]
    assert output_result["status"] == RunnableStatus.SUCCESS.value


def test_document_embedder_empty_content(huggingface_document_embedder_workflow, empty_document_content_input):
    workflow, embedder, output_node = huggingface_document_embedder_workflow
    response = workflow.run(input_data=empty_document_content_input)
    assert_embedder_failure(response, embedder, output_node)


def test_text_embedder_api_returns_empty_embedding(
    huggingface_text_embedder_workflow, query_input, empty_embedding_response_factory, huggingface_model
):
    workflow, embedder, output_node = huggingface_text_embedder_workflow
    empty_response = empty_embedding_response_factory(huggingface_model)

    with patch("dynamiq.components.embedders.base.BaseEmbedder._embedding") as mock_embedding:
        mock_embedding.return_value = empty_response
        response = workflow.run(input_data=query_input)
        assert response.status == RunnableStatus.SUCCESS
        embedder_result = response.output[embedder.id]
        assert embedder_result["status"] == RunnableStatus.FAILURE.value
        assert "Invalid embedding" in embedder_result["error"]["message"]


def test_document_embedder_api_returns_empty_embedding(
    huggingface_document_embedder_workflow, document_input, empty_embedding_response_factory, huggingface_model
):
    workflow, embedder, output_node = huggingface_document_embedder_workflow
    empty_response = empty_embedding_response_factory(huggingface_model)

    with patch("dynamiq.components.embedders.base.BaseEmbedder._embedding") as mock_embedding:
        mock_embedding.return_value = empty_response
        response = workflow.run(input_data=document_input)
        assert response.status == RunnableStatus.SUCCESS
        embedder_result = response.output[embedder.id]
        assert embedder_result["status"] == RunnableStatus.FAILURE.value
        assert "Embedding is empty" in embedder_result["error"]["message"]


def test_text_embedder_api_returns_null_embedding(
    huggingface_text_embedder_workflow, query_input, null_embedding_response_factory, huggingface_model
):
    workflow, embedder, output_node = huggingface_text_embedder_workflow
    null_response = null_embedding_response_factory(huggingface_model)

    with patch("dynamiq.components.embedders.base.BaseEmbedder._embedding") as mock_embedding:
        mock_embedding.return_value = null_response
        response = workflow.run(input_data=query_input)
        assert response.status == RunnableStatus.SUCCESS
        embedder_result = response.output[embedder.id]
        assert embedder_result["status"] == RunnableStatus.FAILURE.value
        assert "Invalid embedding" in embedder_result["error"]["message"]


def test_document_embedder_api_returns_null_embedding(
    huggingface_document_embedder_workflow, document_input, null_embedding_response_factory, huggingface_model
):
    workflow, embedder, output_node = huggingface_document_embedder_workflow
    null_response = null_embedding_response_factory(huggingface_model)

    with patch("dynamiq.components.embedders.base.BaseEmbedder._embedding") as mock_embedding:
        mock_embedding.return_value = null_response
        response = workflow.run(input_data=document_input)
        assert response.status == RunnableStatus.SUCCESS
        embedder_result = response.output[embedder.id]
        assert embedder_result["status"] == RunnableStatus.FAILURE.value
        assert "has no embedding" in embedder_result["error"]["message"]


@pytest.fixture
def max_tokens_error_message():
    return "Input text exceeds maximum token limit. Please provide a shorter input."


def test_text_embedder_max_tokens_error(
    huggingface_text_embedder_workflow, long_query_input, max_tokens_error_message, huggingface_model
):
    workflow, embedder, output_node = huggingface_text_embedder_workflow

    with patch("dynamiq.components.embedders.base.BaseEmbedder._embedding") as mock_embedding:
        error = BadRequestError(max_tokens_error_message, huggingface_model, "huggingface")
        mock_embedding.side_effect = error
        response = workflow.run(input_data=long_query_input)
        assert_embedder_failure(response, embedder, output_node, "BadRequestError", "token limit")


def test_document_embedder_max_tokens_error(
    huggingface_document_embedder_workflow, long_document_input, max_tokens_error_message, huggingface_model
):
    workflow, embedder, output_node = huggingface_document_embedder_workflow

    with patch("dynamiq.components.embedders.base.BaseEmbedder._embedding") as mock_embedding:
        error = BadRequestError(max_tokens_error_message, huggingface_model, "huggingface")
        mock_embedding.side_effect = error
        response = workflow.run(input_data=long_document_input)
        assert_embedder_failure(response, embedder, output_node, "BadRequestError", "token limit")


@pytest.fixture
def invalid_model_error_message():
    return "The model invalid-model does not exist or you are not authorized to access it."


def test_text_embedder_invalid_model(
    huggingface_text_embedder_workflow, query_input, invalid_model_error_message, huggingface_model
):
    workflow, embedder, output_node = huggingface_text_embedder_workflow

    with patch("dynamiq.components.embedders.base.BaseEmbedder._embedding") as mock_embedding:
        error = BadRequestError(invalid_model_error_message, huggingface_model, "huggingface")
        mock_embedding.side_effect = error
        response = workflow.run(input_data=query_input)
        assert_embedder_failure(response, embedder, output_node, "BadRequestError", "does not exist")
