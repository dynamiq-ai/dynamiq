import uuid
from unittest.mock import Mock, patch

import pytest

from dynamiq import Workflow, connections
from dynamiq.callbacks import TracingCallbackHandler
from dynamiq.flows import Flow
from dynamiq.nodes.rankers import CohereReranker
from dynamiq.runnables import RunnableConfig, RunnableResult, RunnableStatus
from dynamiq.types import Document


@pytest.fixture
def mock_rerank_response():
    return Mock(
        results=[
            {"index": 0, "relevance_score": 0.8},
            {"index": 1, "relevance_score": 0.6},
        ]
    )


@pytest.fixture
def mock_rerank_executor(mock_rerank_response):
    with patch("litellm.rerank") as mock:
        mock.return_value = mock_rerank_response
        yield mock


def test_workflow_with_cohere_reranker(mock_rerank_executor):
    connection = connections.Cohere(
        id=str(uuid.uuid4()),
        api_key="api_key",
    )
    model = "cohere/rerank-v3.5"

    documents = [
        Document(content="Pizza is a delicious Italian dish"),
        Document(content="Hamburgers are popular fast food"),
    ]

    wf_cohere_ai = Workflow(
        id=str(uuid.uuid4()),
        flow=Flow(
            nodes=[
                CohereReranker(
                    name="CohereReranker",
                    connection=connection,
                    model=model,
                    top_k=2,
                ),
            ],
        ),
    )

    input_data = {"query": "Tell me about pizza", "documents": documents}

    response = wf_cohere_ai.run(
        input_data=input_data,
        config=RunnableConfig(callbacks=[TracingCallbackHandler()]),
    )

    expected_documents = documents.copy()
    expected_documents[0].score = 0.8
    expected_documents[1].score = 0.6

    expected_result = RunnableResult(
        status=RunnableStatus.SUCCESS,
        input=input_data,
        output={"documents": expected_documents},
    ).to_dict()

    expected_output = {wf_cohere_ai.flow.nodes[0].id: expected_result}

    assert response == RunnableResult(
        status=RunnableStatus.SUCCESS,
        input=input_data,
        output=expected_output,
    )
    assert response.output == expected_output

    mock_rerank_executor.assert_called_once_with(
        model=model,
        query=input_data["query"],
        documents=[doc.content for doc in documents],
        top_n=2,
    )


def test_workflow_with_cohere_reranker_empty_documents(mock_rerank_executor):
    connection = connections.Cohere(
        id=str(uuid.uuid4()),
        api_key="api_key",
    )
    model = "cohere/rerank-v3.5"

    wf_cohere_ai = Workflow(
        id=str(uuid.uuid4()),
        flow=Flow(
            nodes=[
                CohereReranker(
                    name="CohereReranker",
                    connection=connection,
                    model=model,
                ),
            ],
        ),
    )

    input_data = {"query": "Tell me about pizza", "documents": []}

    response = wf_cohere_ai.run(
        input_data=input_data,
        config=RunnableConfig(callbacks=[TracingCallbackHandler()]),
    )

    expected_result = RunnableResult(
        status=RunnableStatus.SUCCESS,
        input=input_data,
        output={"documents": []},
    ).to_dict()

    expected_output = {wf_cohere_ai.flow.nodes[0].id: expected_result}

    assert response == RunnableResult(
        status=RunnableStatus.SUCCESS,
        input=input_data,
        output=expected_output,
    )
    assert response.output == expected_output

    mock_rerank_executor.assert_not_called()
