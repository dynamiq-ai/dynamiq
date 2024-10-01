import uuid

from dynamiq import Workflow, connections
from dynamiq.callbacks import TracingCallbackHandler
from dynamiq.flows import Flow
from dynamiq.nodes.embedders import CohereDocumentEmbedder, CohereTextEmbedder
from dynamiq.runnables import RunnableConfig, RunnableResult, RunnableStatus
from dynamiq.types import Document


def test_workflow_with_cohere_text_embedder(mock_embedding_executor):
    connection = connections.Cohere(
        id=str(uuid.uuid4()),
        api_key="api_key",
    )
    model = "cohere/embed-english-v2.0"
    wf_cohere_ai = Workflow(
        id=str(uuid.uuid4()),
        flow=Flow(
            nodes=[
                CohereTextEmbedder(
                    name="CohereTextEmbedder", connection=connection, model=model
                ),
            ],
        ),
    )
    input = {"query": "I love pizza!"}
    response = wf_cohere_ai.run(
        input_data=input,
        config=RunnableConfig(callbacks=[TracingCallbackHandler()]),
    )

    expected_result = RunnableResult(
        status=RunnableStatus.SUCCESS,
        input=input,
        output={"query": "I love pizza!", "embedding": [0]},
    ).to_dict()
    expected_output = {wf_cohere_ai.flow.nodes[0].id: expected_result}
    assert response == RunnableResult(
        status=RunnableStatus.SUCCESS,
        input=input,
        output=expected_output,
    )
    assert response.output == expected_output
    mock_embedding_executor.assert_called_once_with(
        input_type="search_query",
        input=[input["query"]],
        model=model,
        api_key=connection.api_key,
    )


def test_workflow_with_cohere_document_embedder(
    mock_llm_response_text, mock_embedding_executor
):
    connection = connections.Cohere(
        id=str(uuid.uuid4()),
        api_key="api_key",
    )
    model = "cohere/embed-english-v2.0"
    wf_cohere_ai = Workflow(
        id=str(uuid.uuid4()),
        flow=Flow(
            nodes=[
                CohereDocumentEmbedder(
                    name="CohereDocumentEmbedder",
                    connection=connection,
                    model=model,
                ),
            ],
        ),
    )
    document = [Document(content="I love pizza!")]
    input = {"documents": document}
    response = wf_cohere_ai.run(
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
    expected_output = {wf_cohere_ai.flow.nodes[0].id: expected_result}
    assert response == RunnableResult(
        status=RunnableStatus.SUCCESS,
        input=input,
        output=expected_output,
    )
    assert response.output == expected_output
    mock_embedding_executor.assert_called_once_with(
        input=[document[0].content],
        input_type="search_query",
        model=model,
        api_key=connection.api_key,
    )
