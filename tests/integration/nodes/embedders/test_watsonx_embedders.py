import uuid

from dynamiq import Workflow, connections
from dynamiq.callbacks import TracingCallbackHandler
from dynamiq.flows import Flow
from dynamiq.nodes.embedders import WatsonXDocumentEmbedder, WatsonXTextEmbedder
from dynamiq.runnables import RunnableConfig, RunnableResult, RunnableStatus
from dynamiq.types import Document


def test_workflow_with_watsonx_text_embedder(mock_embedding_executor):
    connection = connections.WatsonX(
        id=str(uuid.uuid4()), api_key="api_key", project_id="project_id", url="https://your-url/"
    )
    model = "watsonx/ibm/slate-30m-english-rtrvr"
    wf_watsonx_ai = Workflow(
        id=str(uuid.uuid4()),
        flow=Flow(
            nodes=[
                WatsonXTextEmbedder(name="WatsonXTextEmbedder", connection=connection, model=model),
            ],
        ),
    )
    input = {"query": "I love pizza!"}
    response = wf_watsonx_ai.run(
        input_data=input,
        config=RunnableConfig(callbacks=[TracingCallbackHandler()]),
    )

    expected_result = RunnableResult(
        status=RunnableStatus.SUCCESS,
        input=input,
        output={"query": "I love pizza!", "embedding": [0]},
    ).to_dict()
    expected_output = {wf_watsonx_ai.flow.nodes[0].id: expected_result}
    assert response == RunnableResult(
        status=RunnableStatus.SUCCESS,
        input=input,
        output=expected_output,
    )
    assert response.output == expected_output
    mock_embedding_executor.assert_called_once_with(
        input=[input["query"]],
        model=model,
        apikey=connection.api_key,
        project_id=connection.project_id,
        url=connection.url,
    )


def test_workflow_with_watsonx_document_embedder(mock_embedding_executor):
    connection = connections.WatsonX(
        id=str(uuid.uuid4()), api_key="api_key", project_id="project_id", url="https://your-url/"
    )
    model = "watsonx/ibm/slate-30m-english-rtrvr"
    wf_watsonx_ai = Workflow(
        id=str(uuid.uuid4()),
        flow=Flow(
            nodes=[
                WatsonXDocumentEmbedder(
                    name="WatsonXDocumentEmbedder",
                    connection=connection,
                    model=model,
                ),
            ],
        ),
    )
    document = [Document(content="I love pizza!")]
    input = {"documents": document}
    response = wf_watsonx_ai.run(
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
    expected_output = {wf_watsonx_ai.flow.nodes[0].id: expected_result}
    assert response == RunnableResult(
        status=RunnableStatus.SUCCESS,
        input=input,
        output=expected_output,
    )
    assert response.output == expected_output
    mock_embedding_executor.assert_called_once_with(
        input=[document[0].content],
        model=model,
        apikey=connection.api_key,
        project_id=connection.project_id,
        url=connection.url,
    )
