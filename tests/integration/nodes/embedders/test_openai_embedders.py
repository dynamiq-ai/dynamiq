import uuid
from unittest.mock import ANY, patch

import pytest
from litellm import APIError, AuthenticationError, BadRequestError, RateLimitError

from dynamiq import Workflow, connections
from dynamiq.callbacks import TracingCallbackHandler
from dynamiq.flows import Flow
from dynamiq.nodes.embedders import OpenAIDocumentEmbedder, OpenAITextEmbedder
from dynamiq.nodes.node import NodeDependency
from dynamiq.nodes.utils import Output
from dynamiq.runnables import RunnableConfig, RunnableResult, RunnableStatus
from dynamiq.types import Document


def test_workflow_with_openai_text_embedder(mock_embedding_executor):
    connection = connections.OpenAI(
        id=str(uuid.uuid4()),
        api_key="api_key",
    )
    model = "text-embedding-3-small"
    wf_openai_ai = Workflow(
        id=str(uuid.uuid4()),
        flow=Flow(
            nodes=[
                OpenAITextEmbedder(
                    name="OpenAITextEmbedder", connection=connection, model=model
                ),
            ],
        ),
    )
    input = {"query": "I love pizza!"}
    response = wf_openai_ai.run(
        input_data=input,
        config=RunnableConfig(callbacks=[TracingCallbackHandler()]),
    )

    expected_result = RunnableResult(
        status=RunnableStatus.SUCCESS,
        input=input,
        output={"query": "I love pizza!", "embedding": [0]},
    ).to_dict()
    expected_output = {wf_openai_ai.flow.nodes[0].id: expected_result}
    assert response == RunnableResult(
        status=RunnableStatus.SUCCESS,
        input=input,
        output=expected_output,
    )
    assert response.output == expected_output
    mock_embedding_executor.assert_called_once_with(
        input=[input["query"]],
        model=model,
        client=ANY,
    )


def test_workflow_with_openai_document_embedder(mock_embedding_executor):
    connection = connections.OpenAI(
        id=str(uuid.uuid4()),
        api_key="api_key",
    )
    model = "text-embedding-3-small"
    wf_openai_ai = Workflow(
        id=str(uuid.uuid4()),
        flow=Flow(
            nodes=[
                OpenAIDocumentEmbedder(
                    name="OpenAIDocumentEmbedder",
                    connection=connection,
                    model=model,
                ),
            ],
        ),
    )
    document = [Document(content="I love pizza!")]
    input = {"documents": document}
    response = wf_openai_ai.run(
        input_data=input,
        config=RunnableConfig(callbacks=[TracingCallbackHandler()]),
    )

    expected_result = RunnableResult(
        status=RunnableStatus.SUCCESS,
        input=input,
        output={
            **input,
            "meta": {
                "model": model,
                "usage": {
                    "usage": {
                        "prompt_tokens": 6,
                        "completion_tokens": 0,
                        "total_tokens": 6,
                    }
                },
            },
        },
    ).to_dict()
    expected_output = {wf_openai_ai.flow.nodes[0].id: expected_result}
    assert response == RunnableResult(
        status=RunnableStatus.SUCCESS,
        input=input,
        output=expected_output,
    )
    assert response.output == expected_output
    mock_embedding_executor.assert_called_once_with(
        input=[document[0].content],
        model=model,
        client=ANY,
    )


@pytest.fixture
def text_embedder_workflow():
    connection = connections.OpenAI(
        id=str(uuid.uuid4()),
        api_key="api_key",
    )
    model = "text-embedding-3-small"
    embedder = OpenAITextEmbedder(id="text_embedder", name="OpenAITextEmbedder", connection=connection, model=model)
    output_node = Output(id="output_node", depends=[NodeDependency(embedder)])

    return (
        Workflow(
            id=str(uuid.uuid4()),
            flow=Flow(
                nodes=[embedder, output_node],
            ),
        ),
        embedder,
        output_node,
    )


@pytest.fixture
def document_embedder_workflow():
    connection = connections.OpenAI(
        id=str(uuid.uuid4()),
        api_key="api_key",
    )
    model = "text-embedding-3-small"
    embedder = OpenAIDocumentEmbedder(
        id="document_embedder",
        name="OpenAIDocumentEmbedder",
        connection=connection,
        model=model,
    )
    output_node = Output(id="output_node", depends=[NodeDependency(embedder)])

    return (
        Workflow(
            id=str(uuid.uuid4()),
            flow=Flow(
                nodes=[embedder, output_node],
            ),
        ),
        embedder,
        output_node,
    )


@pytest.mark.parametrize(
    "error_config,expected_type",
    [
        ((AuthenticationError, "Invalid API key", "openai", "text-embedding-3-small"), "AuthenticationError"),
        ((RateLimitError, "Rate limit exceeded", "openai", "text-embedding-3-small"), "RateLimitError"),
        ((APIError, "Service unavailable", 500, "openai", "text-embedding-3-small"), "APIError"),
        ((BadRequestError, "Invalid embedding model", "non-existent-model", "openai"), "BadRequestError"),
    ],
)
def test_text_embedder_api_errors(text_embedder_workflow, error_config, expected_type):
    workflow, embedder, output_node = text_embedder_workflow

    error_class = error_config[0]
    error_msg = error_config[1]
    error_args = error_config[2:]

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


def test_text_embedder_missing_input(text_embedder_workflow):
    workflow, embedder, output_node = text_embedder_workflow

    input_data = {}
    response = workflow.run(input_data=input_data)

    assert response.status == RunnableStatus.SUCCESS

    embedder_result = response.output[embedder.id]
    assert embedder_result["status"] == RunnableStatus.FAILURE.value

    output_result = response.output[output_node.id]
    assert output_result["status"] == RunnableStatus.SKIP.value


def test_text_embedder_empty_input(text_embedder_workflow):
    workflow, embedder, output_node = text_embedder_workflow

    input_data = {"query": ""}
    response = workflow.run(input_data=input_data)

    assert response.status == RunnableStatus.SUCCESS

    embedder_result = response.output[embedder.id]
    assert embedder_result["status"] == RunnableStatus.FAILURE.value

    output_result = response.output[output_node.id]
    assert output_result["status"] == RunnableStatus.SKIP.value


@pytest.mark.parametrize(
    "error_config,expected_type",
    [
        ((AuthenticationError, "Invalid API key", "openai", "text-embedding-3-small"), "AuthenticationError"),
        ((RateLimitError, "Rate limit exceeded", "openai", "text-embedding-3-small"), "RateLimitError"),
        ((APIError, "Service unavailable", 500, "openai", "text-embedding-3-small"), "APIError"),
        ((BadRequestError, "Invalid embedding model", "non-existent-model", "openai"), "BadRequestError"),
    ],
)
def test_document_embedder_api_errors(document_embedder_workflow, error_config, expected_type):
    workflow, embedder, output_node = document_embedder_workflow

    error_class = error_config[0]
    error_msg = error_config[1]
    error_args = error_config[2:]

    with patch("dynamiq.components.embedders.base.BaseEmbedder._embedding") as mock_embedding:
        if error_class == APIError:
            error = error_class(error_args[0], error_msg, error_args[1], error_args[2])
        else:
            error = error_class(error_msg, *error_args)

        mock_embedding.side_effect = error

        document = [Document(content="Test content")]
        input_data = {"documents": document}
        response = workflow.run(input_data=input_data)

        assert response.status == RunnableStatus.SUCCESS

        embedder_result = response.output[embedder.id]
        assert embedder_result["status"] == RunnableStatus.FAILURE.value
        assert expected_type in embedder_result["error"]["type"]
        assert error_msg in embedder_result["error"]["message"]

        output_result = response.output[output_node.id]
        assert output_result["status"] == RunnableStatus.SKIP.value


def test_document_embedder_missing_input(document_embedder_workflow):
    workflow, embedder, output_node = document_embedder_workflow

    input_data = {}
    response = workflow.run(input_data=input_data)

    assert response.status == RunnableStatus.SUCCESS

    embedder_result = response.output[embedder.id]
    assert embedder_result["status"] == RunnableStatus.FAILURE.value

    output_result = response.output[output_node.id]
    assert output_result["status"] == RunnableStatus.SKIP.value


def test_document_embedder_empty_document_list(document_embedder_workflow):
    workflow, embedder, output_node = document_embedder_workflow

    input_data = {"documents": []}
    response = workflow.run(input_data=input_data)

    assert response.status == RunnableStatus.SUCCESS

    embedder_result = response.output[embedder.id]
    assert embedder_result["status"] == RunnableStatus.SUCCESS.value

    output_result = response.output[output_node.id]
    assert output_result["status"] == RunnableStatus.SUCCESS.value


def test_document_embedder_empty_content(document_embedder_workflow):
    workflow, embedder, output_node = document_embedder_workflow

    input_data = {"documents": [Document(content="")]}
    response = workflow.run(input_data=input_data)

    assert response.status == RunnableStatus.SUCCESS

    embedder_result = response.output[embedder.id]
    assert embedder_result["status"] == RunnableStatus.FAILURE.value

    output_result = response.output[output_node.id]
    assert output_result["status"] == RunnableStatus.SKIP.value
