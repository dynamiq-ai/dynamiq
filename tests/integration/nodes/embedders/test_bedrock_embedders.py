import uuid
from unittest.mock import patch

import pytest
from litellm import APIError, AuthenticationError, BadRequestError, RateLimitError

from dynamiq import connections
from dynamiq.callbacks import TracingCallbackHandler
from dynamiq.nodes.embedders import BedrockDocumentEmbedder, BedrockTextEmbedder
from dynamiq.runnables import RunnableConfig, RunnableStatus
from tests.integration.nodes.embedders.conftest import (
    assert_embedder_failure,
    assert_embedder_success,
    create_document_embedder_workflow,
    create_text_embedder_workflow,
)


@pytest.fixture
def bedrock_connection():
    return connections.AWS(
        id=str(uuid.uuid4()),
        access_key_id="your_access_key_id",
        secret_access_key="your_secret_access_key",
        region="us-east-1",
    )


@pytest.fixture
def bedrock_model():
    return "amazon.titan-embed-text-v1"


@pytest.fixture
def bedrock_text_embedder(bedrock_connection, bedrock_model):
    return BedrockTextEmbedder(
        id="text_embedder", name="BedrockTextEmbedder", connection=bedrock_connection, model=bedrock_model
    )


@pytest.fixture
def bedrock_document_embedder(bedrock_connection, bedrock_model):
    return BedrockDocumentEmbedder(
        id="document_embedder", name="BedrockDocumentEmbedder", connection=bedrock_connection, model=bedrock_model
    )


@pytest.fixture
def bedrock_text_embedder_workflow(bedrock_text_embedder):
    return create_text_embedder_workflow(bedrock_text_embedder)


@pytest.fixture
def bedrock_document_embedder_workflow(bedrock_document_embedder):
    return create_document_embedder_workflow(bedrock_document_embedder)


def test_workflow_with_bedrock_text_embedder(
    mock_embedding_executor, bedrock_text_embedder_workflow, query_input, bedrock_model, bedrock_connection
):
    workflow, embedder, output_node = bedrock_text_embedder_workflow

    response = workflow.run(
        input_data=query_input,
        config=RunnableConfig(callbacks=[TracingCallbackHandler()]),
    )

    assert_embedder_success(response, embedder, output_node)
    mock_embedding_executor.assert_called_once_with(
        input=[query_input["query"]],
        model=bedrock_model,
        aws_secret_access_key=bedrock_connection.secret_access_key,
        aws_region_name=bedrock_connection.region,
        aws_access_key_id=bedrock_connection.access_key_id,
    )


def test_workflow_with_bedrock_document_embedder(
    mock_embedding_executor, bedrock_document_embedder_workflow, document_input, bedrock_model, bedrock_connection
):
    workflow, embedder, output_node = bedrock_document_embedder_workflow

    response = workflow.run(
        input_data=document_input,
        config=RunnableConfig(callbacks=[TracingCallbackHandler()]),
    )

    assert_embedder_success(response, embedder, output_node)
    mock_embedding_executor.assert_called_once_with(
        input=[document_input["documents"][0].content],
        model=bedrock_model,
        aws_secret_access_key=bedrock_connection.secret_access_key,
        aws_region_name=bedrock_connection.region,
        aws_access_key_id=bedrock_connection.access_key_id,
    )


@pytest.mark.parametrize(
    "error_class,error_msg,error_args,expected_type",
    [
        (AuthenticationError, "Invalid credentials", ["bedrock", "amazon.titan-embed-text-v1"], "AuthenticationError"),
        (RateLimitError, "Rate limit exceeded", ["bedrock", "amazon.titan-embed-text-v1"], "RateLimitError"),
        (APIError, "Service unavailable", [500, "bedrock", "amazon.titan-embed-text-v1"], "APIError"),
        (BadRequestError, "Invalid embedding model", ["non-existent-model", "bedrock"], "BadRequestError"),
    ],
)
def test_text_embedder_api_errors(bedrock_text_embedder_workflow, error_class, error_msg, error_args, expected_type):
    workflow, embedder, output_node = bedrock_text_embedder_workflow

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
            model=embedder.model,
            input=["Test query"],
            aws_access_key_id=embedder.connection.access_key_id,
            aws_secret_access_key=embedder.connection.secret_access_key,
            aws_region_name=embedder.connection.region,
        )


def test_text_embedder_missing_input(bedrock_text_embedder_workflow, missing_input):
    workflow, embedder, output_node = bedrock_text_embedder_workflow
    response = workflow.run(input_data=missing_input)
    assert_embedder_failure(response, embedder, output_node)


def test_text_embedder_empty_input(bedrock_text_embedder_workflow, empty_query_input):
    workflow, embedder, output_node = bedrock_text_embedder_workflow
    response = workflow.run(input_data=empty_query_input)
    assert_embedder_failure(response, embedder, output_node)


@pytest.mark.parametrize(
    "error_class,error_msg,error_args,expected_type",
    [
        (AuthenticationError, "Invalid credentials", ["bedrock", "amazon.titan-embed-text-v1"], "AuthenticationError"),
        (RateLimitError, "Rate limit exceeded", ["bedrock", "amazon.titan-embed-text-v1"], "RateLimitError"),
        (APIError, "Service unavailable", [500, "bedrock", "amazon.titan-embed-text-v1"], "APIError"),
        (BadRequestError, "Invalid embedding model", ["non-existent-model", "bedrock"], "BadRequestError"),
    ],
)
def test_document_embedder_api_errors(
    bedrock_document_embedder_workflow, document_input, error_class, error_msg, error_args, expected_type
):
    workflow, embedder, output_node = bedrock_document_embedder_workflow

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
            aws_access_key_id=embedder.connection.access_key_id,
            aws_secret_access_key=embedder.connection.secret_access_key,
            aws_region_name=embedder.connection.region,
        )


def test_document_embedder_missing_input(bedrock_document_embedder_workflow, missing_input):
    workflow, embedder, output_node = bedrock_document_embedder_workflow
    response = workflow.run(input_data=missing_input)
    assert_embedder_failure(response, embedder, output_node)


def test_document_embedder_empty_document_list(bedrock_document_embedder_workflow, empty_documents_input):
    workflow, embedder, output_node = bedrock_document_embedder_workflow
    response = workflow.run(input_data=empty_documents_input)

    assert response.status == RunnableStatus.SUCCESS

    embedder_result = response.output[embedder.id]
    assert embedder_result["status"] == RunnableStatus.SUCCESS.value
    assert "documents" in embedder_result["output"]
    assert len(embedder_result["output"]["documents"]) == 0

    output_result = response.output[output_node.id]
    assert output_result["status"] == RunnableStatus.SUCCESS.value


def test_document_embedder_empty_content(bedrock_document_embedder_workflow, empty_document_content_input):
    workflow, embedder, output_node = bedrock_document_embedder_workflow
    response = workflow.run(input_data=empty_document_content_input)
    assert_embedder_failure(response, embedder, output_node)


def test_text_embedder_api_returns_empty_embedding(
    bedrock_text_embedder_workflow, query_input, empty_embedding_response_factory, bedrock_model
):
    workflow, embedder, output_node = bedrock_text_embedder_workflow
    empty_response = empty_embedding_response_factory(bedrock_model)

    with patch("dynamiq.components.embedders.base.BaseEmbedder._embedding") as mock_embedding:
        mock_embedding.return_value = empty_response
        response = workflow.run(input_data=query_input)
        assert response.status == RunnableStatus.FAILURE
        embedder_result = response.output[embedder.id]
        assert embedder_result["status"] == RunnableStatus.FAILURE.value
        assert "Invalid embedding" in embedder_result["error"]["message"]


def test_document_embedder_api_returns_empty_embedding(
    bedrock_document_embedder_workflow, document_input, empty_embedding_response_factory, bedrock_model
):
    workflow, embedder, output_node = bedrock_document_embedder_workflow
    empty_response = empty_embedding_response_factory(bedrock_model)

    with patch("dynamiq.components.embedders.base.BaseEmbedder._embedding") as mock_embedding:
        mock_embedding.return_value = empty_response
        response = workflow.run(input_data=document_input)
        assert response.status == RunnableStatus.FAILURE
        embedder_result = response.output[embedder.id]
        assert embedder_result["status"] == RunnableStatus.FAILURE.value
        assert "Embedding is empty" in embedder_result["error"]["message"]


def test_text_embedder_api_returns_null_embedding(
    bedrock_text_embedder_workflow, query_input, null_embedding_response_factory, bedrock_model
):
    workflow, embedder, output_node = bedrock_text_embedder_workflow
    null_response = null_embedding_response_factory(bedrock_model)

    with patch("dynamiq.components.embedders.base.BaseEmbedder._embedding") as mock_embedding:
        mock_embedding.return_value = null_response
        response = workflow.run(input_data=query_input)
        assert response.status == RunnableStatus.FAILURE
        embedder_result = response.output[embedder.id]
        assert embedder_result["status"] == RunnableStatus.FAILURE.value
        assert "Invalid embedding" in embedder_result["error"]["message"]


def test_document_embedder_api_returns_null_embedding(
    bedrock_document_embedder_workflow, document_input, null_embedding_response_factory, bedrock_model
):
    workflow, embedder, output_node = bedrock_document_embedder_workflow
    null_response = null_embedding_response_factory(bedrock_model)

    with patch("dynamiq.components.embedders.base.BaseEmbedder._embedding") as mock_embedding:
        mock_embedding.return_value = null_response
        response = workflow.run(input_data=document_input)
        assert response.status == RunnableStatus.FAILURE
        embedder_result = response.output[embedder.id]
        assert embedder_result["status"] == RunnableStatus.FAILURE.value
        assert "has no embedding" in embedder_result["error"]["message"]


@pytest.fixture
def max_tokens_error_message():
    return "Input is too long. Maximum is 8191 tokens."


def test_text_embedder_max_tokens_error(
    bedrock_text_embedder_workflow, long_query_input, max_tokens_error_message, bedrock_model
):
    workflow, embedder, output_node = bedrock_text_embedder_workflow

    with patch("dynamiq.components.embedders.base.BaseEmbedder._embedding") as mock_embedding:
        error = BadRequestError(max_tokens_error_message, bedrock_model, "bedrock")
        mock_embedding.side_effect = error
        response = workflow.run(input_data=long_query_input)
        assert_embedder_failure(response, embedder, output_node, "BadRequestError", "too long")


def test_document_embedder_max_tokens_error(
    bedrock_document_embedder_workflow, long_document_input, max_tokens_error_message, bedrock_model
):
    workflow, embedder, output_node = bedrock_document_embedder_workflow

    with patch("dynamiq.components.embedders.base.BaseEmbedder._embedding") as mock_embedding:
        error = BadRequestError(max_tokens_error_message, bedrock_model, "bedrock")
        mock_embedding.side_effect = error
        response = workflow.run(input_data=long_document_input)
        assert_embedder_failure(response, embedder, output_node, "BadRequestError", "too long")


@pytest.fixture
def invalid_model_error_message():
    return "Model model-id not found"


def test_text_embedder_invalid_model(
    bedrock_text_embedder_workflow, query_input, invalid_model_error_message, bedrock_model
):
    workflow, embedder, output_node = bedrock_text_embedder_workflow

    with patch("dynamiq.components.embedders.base.BaseEmbedder._embedding") as mock_embedding:
        error = BadRequestError(invalid_model_error_message, bedrock_model, "bedrock")
        mock_embedding.side_effect = error
        response = workflow.run(input_data=query_input)
        assert_embedder_failure(response, embedder, output_node, "BadRequestError", "not found")
