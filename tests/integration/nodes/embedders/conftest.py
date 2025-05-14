import uuid
from unittest.mock import MagicMock

import pytest
from litellm import APIError, AuthenticationError, BadRequestError, RateLimitError

from dynamiq import Workflow
from dynamiq.flows import Flow
from dynamiq.nodes.node import NodeDependency
from dynamiq.nodes.utils import Output
from dynamiq.runnables import RunnableStatus
from dynamiq.types import Document


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


@pytest.fixture
def empty_query_input():
    return {"query": ""}


@pytest.fixture
def missing_input():
    return {}


@pytest.fixture
def empty_documents_input():
    return {"documents": []}


@pytest.fixture
def empty_document_content_input():
    return {"documents": [Document(content="")]}


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
def empty_embedding_response_factory():
    def _factory(model):
        response = MagicMock()
        response.data = [{"embedding": []}]
        response.model = model
        response.usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        return response

    return _factory


@pytest.fixture(
    params=[
        (AuthenticationError, "Invalid credentials", ["invalid", "model"], "AuthenticationError"),
        (RateLimitError, "Rate limit exceeded", ["provider", "model"], "RateLimitError"),
        (APIError, "Service unavailable", [500, "provider", "model"], "APIError"),
        (BadRequestError, "Invalid embedding model", ["non-existent-model", "provider"], "BadRequestError"),
    ]
)
def api_error_params(request):
    return request.param


def create_text_embedder_workflow(embedder):
    output_node = Output(id="output_node", depends=[NodeDependency(embedder)])

    workflow = Workflow(
        id=str(uuid.uuid4()),
        flow=Flow(
            nodes=[embedder, output_node],
        ),
    )

    return workflow, embedder, output_node


def create_document_embedder_workflow(embedder):
    output_node = Output(id="output_node", depends=[NodeDependency(embedder)])

    workflow = Workflow(
        id=str(uuid.uuid4()),
        flow=Flow(
            nodes=[embedder, output_node],
        ),
    )

    return workflow, embedder, output_node


def assert_embedder_success(response, embedder, output_node, expected_embedding_length=1):
    assert response.status == RunnableStatus.SUCCESS

    embedder_result = response.output[embedder.id]
    assert embedder_result["status"] == RunnableStatus.SUCCESS.value

    if "query" in embedder_result["output"]:
        assert "embedding" in embedder_result["output"]
        assert isinstance(embedder_result["output"]["embedding"], list)
        assert len(embedder_result["output"]["embedding"]) == expected_embedding_length
    elif "documents" in embedder_result["output"]:
        assert "meta" in embedder_result["output"]
        assert "model" in embedder_result["output"]["meta"]

    output_result = response.output[output_node.id]
    assert output_result["status"] == RunnableStatus.SUCCESS.value


def assert_embedder_failure(response, embedder, output_node, expected_error_type=None, expected_error_message=None):
    assert response.status == RunnableStatus.SUCCESS

    embedder_result = response.output[embedder.id]
    assert embedder_result["status"] == RunnableStatus.FAILURE.value

    if expected_error_type:
        assert expected_error_type in embedder_result["error"]["type"]

    if expected_error_message:
        assert expected_error_message in embedder_result["error"]["message"]

    output_result = response.output[output_node.id]
    assert output_result["status"] == RunnableStatus.SKIP.value
