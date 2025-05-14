import uuid
from unittest.mock import patch

import pytest
from litellm import APIError, AuthenticationError, BadRequestError, RateLimitError

from dynamiq import connections
from dynamiq.callbacks import TracingCallbackHandler
from dynamiq.nodes.embedders import WatsonXDocumentEmbedder, WatsonXTextEmbedder
from dynamiq.runnables import RunnableConfig, RunnableStatus
from tests.integration.nodes.embedders.conftest import (
    assert_embedder_failure,
    assert_embedder_success,
    create_document_embedder_workflow,
    create_text_embedder_workflow,
)


@pytest.fixture
def watsonx_connection():
    return connections.WatsonX(
        id=str(uuid.uuid4()), api_key="api_key", project_id="project_id", url="https://your-url/"
    )


@pytest.fixture
def watsonx_model():
    return "watsonx/ibm/slate-30m-english-rtrvr"


@pytest.fixture
def watsonx_text_embedder(watsonx_connection, watsonx_model):
    return WatsonXTextEmbedder(
        id="text_embedder", name="WatsonXTextEmbedder", connection=watsonx_connection, model=watsonx_model
    )


@pytest.fixture
def watsonx_document_embedder(watsonx_connection, watsonx_model):
    return WatsonXDocumentEmbedder(
        id="document_embedder", name="WatsonXDocumentEmbedder", connection=watsonx_connection, model=watsonx_model
    )


@pytest.fixture
def watsonx_text_embedder_workflow(watsonx_text_embedder):
    return create_text_embedder_workflow(watsonx_text_embedder)


@pytest.fixture
def watsonx_document_embedder_workflow(watsonx_document_embedder):
    return create_document_embedder_workflow(watsonx_document_embedder)


def test_workflow_with_watsonx_text_embedder(
    mock_embedding_executor, watsonx_text_embedder_workflow, query_input, watsonx_model, watsonx_connection
):
    workflow, embedder, output_node = watsonx_text_embedder_workflow

    response = workflow.run(
        input_data=query_input,
        config=RunnableConfig(callbacks=[TracingCallbackHandler()]),
    )

    assert_embedder_success(response, embedder, output_node)
    mock_embedding_executor.assert_called_once_with(
        input=[query_input["query"]],
        model=watsonx_model,
        apikey=watsonx_connection.api_key,
        project_id=watsonx_connection.project_id,
        url=watsonx_connection.url,
    )


def test_workflow_with_watsonx_document_embedder(
    mock_embedding_executor, watsonx_document_embedder_workflow, document_input, watsonx_model, watsonx_connection
):
    workflow, embedder, output_node = watsonx_document_embedder_workflow

    response = workflow.run(
        input_data=document_input,
        config=RunnableConfig(callbacks=[TracingCallbackHandler()]),
    )

    assert_embedder_success(response, embedder, output_node)
    mock_embedding_executor.assert_called_once_with(
        input=[document_input["documents"][0].content],
        model=watsonx_model,
        apikey=watsonx_connection.api_key,
        project_id=watsonx_connection.project_id,
        url=watsonx_connection.url,
    )


@pytest.mark.parametrize(
    "error_class,error_msg,error_args,expected_type",
    [
        (
            AuthenticationError,
            "Invalid API key",
            ["watsonx", "watsonx/ibm/slate-30m-english-rtrvr"],
            "AuthenticationError",
        ),
        (RateLimitError, "Rate limit exceeded", ["watsonx", "watsonx/ibm/slate-30m-english-rtrvr"], "RateLimitError"),
        (APIError, "Service unavailable", [500, "watsonx", "watsonx/ibm/slate-30m-english-rtrvr"], "APIError"),
        (BadRequestError, "Invalid embedding model", ["non-existent-model", "watsonx"], "BadRequestError"),
    ],
)
def test_text_embedder_api_errors(watsonx_text_embedder_workflow, error_class, error_msg, error_args, expected_type):
    workflow, embedder, output_node = watsonx_text_embedder_workflow

    with patch("dynamiq.components.embedders.base.BaseEmbedder._embedding") as mock_embedding:
        if error_class == APIError:
            error = error_class(error_args[0], error_msg, error_args[1], error_args[2])
        else:
            error = error_class(error_msg, *error_args)

        mock_embedding.side_effect = error

        input_data = {"query": "Test query"}
        response = workflow.run(input_data=input_data)

        assert_embedder_failure(response, embedder, output_node, expected_type, error_msg)


def test_text_embedder_missing_input(watsonx_text_embedder_workflow, missing_input):
    workflow, embedder, output_node = watsonx_text_embedder_workflow
    response = workflow.run(input_data=missing_input)
    assert_embedder_failure(response, embedder, output_node)


def test_text_embedder_empty_input(watsonx_text_embedder_workflow, empty_query_input):
    workflow, embedder, output_node = watsonx_text_embedder_workflow
    response = workflow.run(input_data=empty_query_input)
    assert_embedder_failure(response, embedder, output_node)


@pytest.mark.parametrize(
    "error_class,error_msg,error_args,expected_type",
    [
        (
            AuthenticationError,
            "Invalid API key",
            ["watsonx", "watsonx/ibm/slate-30m-english-rtrvr"],
            "AuthenticationError",
        ),
        (RateLimitError, "Rate limit exceeded", ["watsonx", "watsonx/ibm/slate-30m-english-rtrvr"], "RateLimitError"),
        (APIError, "Service unavailable", [500, "watsonx", "watsonx/ibm/slate-30m-english-rtrvr"], "APIError"),
        (BadRequestError, "Invalid embedding model", ["non-existent-model", "watsonx"], "BadRequestError"),
    ],
)
def test_document_embedder_api_errors(
    watsonx_document_embedder_workflow, document_input, error_class, error_msg, error_args, expected_type
):
    workflow, embedder, output_node = watsonx_document_embedder_workflow

    with patch("dynamiq.components.embedders.base.BaseEmbedder._embedding") as mock_embedding:
        if error_class == APIError:
            error = error_class(error_args[0], error_msg, error_args[1], error_args[2])
        else:
            error = error_class(error_msg, *error_args)

        mock_embedding.side_effect = error
        response = workflow.run(input_data=document_input)
        assert_embedder_failure(response, embedder, output_node, expected_type, error_msg)


def test_document_embedder_missing_input(watsonx_document_embedder_workflow, missing_input):
    workflow, embedder, output_node = watsonx_document_embedder_workflow
    response = workflow.run(input_data=missing_input)
    assert_embedder_failure(response, embedder, output_node)


def test_document_embedder_empty_document_list(watsonx_document_embedder_workflow, empty_documents_input):
    workflow, embedder, output_node = watsonx_document_embedder_workflow
    response = workflow.run(input_data=empty_documents_input)

    assert response.status == RunnableStatus.SUCCESS

    embedder_result = response.output[embedder.id]
    assert embedder_result["status"] == RunnableStatus.SUCCESS.value
    assert "documents" in embedder_result["output"]
    assert len(embedder_result["output"]["documents"]) == 0

    output_result = response.output[output_node.id]
    assert output_result["status"] == RunnableStatus.SUCCESS.value


def test_document_embedder_empty_content(watsonx_document_embedder_workflow, empty_document_content_input):
    workflow, embedder, output_node = watsonx_document_embedder_workflow
    response = workflow.run(input_data=empty_document_content_input)
    assert_embedder_failure(response, embedder, output_node)


def test_text_embedder_api_returns_empty_embedding(
    watsonx_text_embedder_workflow, query_input, empty_embedding_response_factory, watsonx_model
):
    workflow, embedder, output_node = watsonx_text_embedder_workflow
    empty_response = empty_embedding_response_factory(watsonx_model)

    with patch("dynamiq.components.embedders.base.BaseEmbedder._embedding") as mock_embedding:
        mock_embedding.return_value = empty_response
        response = workflow.run(input_data=query_input)
        assert_embedder_success(response, embedder, output_node, expected_embedding_length=0)


def test_document_embedder_api_returns_empty_embedding(
    watsonx_document_embedder_workflow, document_input, empty_embedding_response_factory, watsonx_model
):
    workflow, embedder, output_node = watsonx_document_embedder_workflow
    empty_response = empty_embedding_response_factory(watsonx_model)

    with patch("dynamiq.components.embedders.base.BaseEmbedder._embedding") as mock_embedding:
        mock_embedding.return_value = empty_response
        response = workflow.run(input_data=document_input)
        assert_embedder_success(response, embedder, output_node)


@pytest.fixture
def max_tokens_error_message():
    return "Maximum input size exceeded, limit is 10000 tokens."


def test_text_embedder_max_tokens_error(
    watsonx_text_embedder_workflow, long_query_input, max_tokens_error_message, watsonx_model
):
    workflow, embedder, output_node = watsonx_text_embedder_workflow

    with patch("dynamiq.components.embedders.base.BaseEmbedder._embedding") as mock_embedding:
        error = BadRequestError(max_tokens_error_message, watsonx_model, "watsonx")
        mock_embedding.side_effect = error
        response = workflow.run(input_data=long_query_input)
        assert_embedder_failure(response, embedder, output_node, "BadRequestError", "Maximum input size")


def test_document_embedder_max_tokens_error(
    watsonx_document_embedder_workflow, long_document_input, max_tokens_error_message, watsonx_model
):
    workflow, embedder, output_node = watsonx_document_embedder_workflow

    with patch("dynamiq.components.embedders.base.BaseEmbedder._embedding") as mock_embedding:
        error = BadRequestError(max_tokens_error_message, watsonx_model, "watsonx")
        mock_embedding.side_effect = error
        response = workflow.run(input_data=long_document_input)
        assert_embedder_failure(response, embedder, output_node, "BadRequestError", "Maximum input size")


@pytest.fixture
def invalid_model_error_message():
    return "Model not available"


def test_text_embedder_invalid_model(
    watsonx_text_embedder_workflow, query_input, invalid_model_error_message, watsonx_model
):
    workflow, embedder, output_node = watsonx_text_embedder_workflow

    with patch("dynamiq.components.embedders.base.BaseEmbedder._embedding") as mock_embedding:
        error = BadRequestError(invalid_model_error_message, watsonx_model, "watsonx")
        mock_embedding.side_effect = error
        response = workflow.run(input_data=query_input)
        assert_embedder_failure(response, embedder, output_node, "BadRequestError", "not available")
