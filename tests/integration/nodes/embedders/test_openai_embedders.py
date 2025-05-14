import uuid
from unittest.mock import ANY, MagicMock, patch

import pytest
from litellm import APIError, AuthenticationError, BadRequestError, RateLimitError

from dynamiq import Workflow, connections
from dynamiq.callbacks import TracingCallbackHandler
from dynamiq.flows import Flow
from dynamiq.nodes.embedders import OpenAIDocumentEmbedder, OpenAITextEmbedder
from dynamiq.nodes.node import NodeDependency
from dynamiq.nodes.utils import Output
from dynamiq.runnables import RunnableConfig, RunnableStatus
from dynamiq.types import Document


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
def query_text():
    return "I love pizza!"


@pytest.fixture
def document_content():
    return "Test document content"


@pytest.fixture
def query_input(query_text):
    return {"query": query_text}


@pytest.fixture
def document_input(document_content):
    return {"documents": [Document(content=document_content)]}


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
    output_node = Output(id="output_node", depends=[NodeDependency(openai_text_embedder)])

    workflow = Workflow(
        id=str(uuid.uuid4()),
        flow=Flow(
            nodes=[openai_text_embedder, output_node],
        ),
    )

    return workflow, openai_text_embedder, output_node


@pytest.fixture
def openai_document_embedder_workflow(openai_document_embedder):
    output_node = Output(id="output_node", depends=[NodeDependency(openai_document_embedder)])

    workflow = Workflow(
        id=str(uuid.uuid4()),
        flow=Flow(
            nodes=[openai_document_embedder, output_node],
        ),
    )

    return workflow, openai_document_embedder, output_node


def test_workflow_with_openai_text_embedder(
    mock_embedding_executor, openai_text_embedder_workflow, query_input, openai_model
):
    workflow, embedder, output_node = openai_text_embedder_workflow

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

    output_result = response.output[output_node.id]
    assert output_result["status"] == RunnableStatus.SUCCESS.value

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

    assert response.status == RunnableStatus.SUCCESS

    embedder_result = response.output[embedder.id]
    assert embedder_result["status"] == RunnableStatus.SUCCESS.value
    assert "documents" in embedder_result["output"]
    assert len(embedder_result["output"]["documents"]) == 1

    assert "meta" in embedder_result["output"]
    assert "model" in embedder_result["output"]["meta"]
    assert embedder_result["output"]["meta"]["model"] == openai_model

    output_result = response.output[output_node.id]
    assert output_result["status"] == RunnableStatus.SUCCESS.value

    mock_embedding_executor.assert_called_once_with(
        input=[document_input["documents"][0].content],
        model=openai_model,
        client=ANY,
    )


@pytest.fixture
def empty_query_input():
    return {"query": ""}


@pytest.fixture
def missing_input():
    return {}


@pytest.fixture
def empty_embedding_response(openai_model):
    response = MagicMock()
    response.data = [{"embedding": []}]
    response.model = openai_model
    response.usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    return response


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

        assert response.status == RunnableStatus.SUCCESS

        embedder_result = response.output[embedder.id]
        assert embedder_result["status"] == RunnableStatus.FAILURE.value
        assert expected_type in embedder_result["error"]["type"]
        assert error_msg in embedder_result["error"]["message"]

        output_result = response.output[output_node.id]
        assert output_result["status"] == RunnableStatus.SKIP.value


def test_text_embedder_missing_input(openai_text_embedder_workflow, missing_input):
    workflow, embedder, output_node = openai_text_embedder_workflow

    response = workflow.run(input_data=missing_input)

    assert response.status == RunnableStatus.SUCCESS

    embedder_result = response.output[embedder.id]
    assert embedder_result["status"] == RunnableStatus.FAILURE.value

    output_result = response.output[output_node.id]
    assert output_result["status"] == RunnableStatus.SKIP.value


def test_text_embedder_empty_input(openai_text_embedder_workflow, empty_query_input):
    workflow, embedder, output_node = openai_text_embedder_workflow

    response = workflow.run(input_data=empty_query_input)

    assert response.status == RunnableStatus.SUCCESS

    embedder_result = response.output[embedder.id]
    assert embedder_result["status"] == RunnableStatus.FAILURE.value

    output_result = response.output[output_node.id]
    assert output_result["status"] == RunnableStatus.SKIP.value


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

        assert response.status == RunnableStatus.SUCCESS

        embedder_result = response.output[embedder.id]
        assert embedder_result["status"] == RunnableStatus.FAILURE.value
        assert expected_type in embedder_result["error"]["type"]
        assert error_msg in embedder_result["error"]["message"]

        output_result = response.output[output_node.id]
        assert output_result["status"] == RunnableStatus.SKIP.value


def test_document_embedder_missing_input(openai_document_embedder_workflow, missing_input):
    workflow, embedder, output_node = openai_document_embedder_workflow

    response = workflow.run(input_data=missing_input)

    assert response.status == RunnableStatus.SUCCESS

    embedder_result = response.output[embedder.id]
    assert embedder_result["status"] == RunnableStatus.FAILURE.value

    output_result = response.output[output_node.id]
    assert output_result["status"] == RunnableStatus.SKIP.value


@pytest.fixture
def empty_documents_input():
    return {"documents": []}


def test_document_embedder_empty_document_list(openai_document_embedder_workflow, empty_documents_input):
    workflow, embedder, output_node = openai_document_embedder_workflow

    response = workflow.run(input_data=empty_documents_input)

    assert response.status == RunnableStatus.SUCCESS

    embedder_result = response.output[embedder.id]
    assert embedder_result["status"] == RunnableStatus.SUCCESS.value

    output_result = response.output[output_node.id]
    assert output_result["status"] == RunnableStatus.SUCCESS.value


@pytest.fixture
def empty_document_content_input():
    return {"documents": [Document(content="")]}


def test_document_embedder_empty_content(openai_document_embedder_workflow, empty_document_content_input):
    workflow, embedder, output_node = openai_document_embedder_workflow

    response = workflow.run(input_data=empty_document_content_input)

    assert response.status == RunnableStatus.SUCCESS

    embedder_result = response.output[embedder.id]
    assert embedder_result["status"] == RunnableStatus.FAILURE.value

    output_result = response.output[output_node.id]
    assert output_result["status"] == RunnableStatus.SKIP.value


def test_text_embedder_api_returns_empty_embedding(
    openai_text_embedder_workflow, query_input, empty_embedding_response
):
    workflow, embedder, output_node = openai_text_embedder_workflow

    with patch("dynamiq.components.embedders.base.BaseEmbedder._embedding") as mock_embedding:
        mock_embedding.return_value = empty_embedding_response

        response = workflow.run(input_data=query_input)

        assert response.status == RunnableStatus.SUCCESS

        embedder_result = response.output[embedder.id]
        assert embedder_result["status"] == RunnableStatus.SUCCESS.value
        assert "embedding" in embedder_result["output"]
        assert embedder_result["output"]["embedding"] == []


def test_document_embedder_api_returns_empty_embedding(
    openai_document_embedder_workflow, document_input, empty_embedding_response
):
    workflow, embedder, output_node = openai_document_embedder_workflow

    with patch("dynamiq.components.embedders.base.BaseEmbedder._embedding") as mock_embedding:
        mock_embedding.return_value = empty_embedding_response

        response = workflow.run(input_data=document_input)

        assert response.status == RunnableStatus.SUCCESS

        embedder_result = response.output[embedder.id]
        assert embedder_result["status"] == RunnableStatus.SUCCESS.value
        assert "documents" in embedder_result["output"]
        assert len(embedder_result["output"]["documents"]) == 1


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
    return "This model's maximum context length is 8191 tokens, but the provided inputs have 10000 tokens"


def test_text_embedder_max_tokens_error(
    openai_text_embedder_workflow, long_query_input, max_tokens_error_message, openai_model
):
    workflow, embedder, output_node = openai_text_embedder_workflow

    with patch("dynamiq.components.embedders.base.BaseEmbedder._embedding") as mock_embedding:
        error = BadRequestError(max_tokens_error_message, openai_model, "openai")
        mock_embedding.side_effect = error

        response = workflow.run(input_data=long_query_input)

        assert response.status == RunnableStatus.SUCCESS

        embedder_result = response.output[embedder.id]
        assert embedder_result["status"] == RunnableStatus.FAILURE.value
        assert "BadRequestError" in embedder_result["error"]["type"]
        assert "maximum context length" in embedder_result["error"]["message"]

        output_result = response.output[output_node.id]
        assert output_result["status"] == RunnableStatus.SKIP.value


def test_document_embedder_max_tokens_error(
    openai_document_embedder_workflow, long_document_input, max_tokens_error_message, openai_model
):
    workflow, embedder, output_node = openai_document_embedder_workflow

    with patch("dynamiq.components.embedders.base.BaseEmbedder._embedding") as mock_embedding:
        error = BadRequestError(max_tokens_error_message, openai_model, "openai")
        mock_embedding.side_effect = error

        response = workflow.run(input_data=long_document_input)

        assert response.status == RunnableStatus.SUCCESS

        embedder_result = response.output[embedder.id]
        assert embedder_result["status"] == RunnableStatus.FAILURE.value
        assert "BadRequestError" in embedder_result["error"]["type"]
        assert "maximum context length" in embedder_result["error"]["message"]

        output_result = response.output[output_node.id]
        assert output_result["status"] == RunnableStatus.SKIP.value


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

        assert response.status == RunnableStatus.SUCCESS

        embedder_result = response.output[embedder.id]
        assert embedder_result["status"] == RunnableStatus.FAILURE.value
        assert "BadRequestError" in embedder_result["error"]["type"]
        assert "Invalid dimensions" in embedder_result["error"]["message"]

        output_result = response.output[output_node.id]
        assert output_result["status"] == RunnableStatus.SKIP.value
