import uuid

from dynamiq import Workflow, connections
from dynamiq.callbacks import TracingCallbackHandler
from dynamiq.flows import Flow
from dynamiq.nodes.embedders import BedrockDocumentEmbedder, BedrockTextEmbedder
from dynamiq.runnables import RunnableConfig, RunnableResult, RunnableStatus
from dynamiq.types import Document


def test_workflow_with_bedrock_text_embedder(mock_embedding_executor):
    connection = connections.AWS(
        id=str(uuid.uuid4()),
        access_key_id="your_access_key_id",
        secret_access_key="your_secret_access_key",
        region="us-east-1",
    )
    model = "amazon.titan-embed-text-v1"
    wf_bedrock_ai = Workflow(
        id=str(uuid.uuid4()),
        flow=Flow(
            nodes=[
                BedrockTextEmbedder(
                    name="BedrockTextEmbedder", connection=connection, model=model
                ),
            ],
        ),
    )
    input = {"query": "I love pizza!"}
    response = wf_bedrock_ai.run(
        input_data=input,
        config=RunnableConfig(callbacks=[TracingCallbackHandler()]),
    )

    expected_result = RunnableResult(
        status=RunnableStatus.SUCCESS,
        input=input,
        output={"query": "I love pizza!", "embedding": [0]},
    ).to_dict()
    expected_output = {wf_bedrock_ai.flow.nodes[0].id: expected_result}
    assert response == RunnableResult(
        status=RunnableStatus.SUCCESS,
        input=input,
        output=expected_output,
    )
    assert response.output == expected_output
    mock_embedding_executor.assert_called_once_with(
        input=[input["query"]],
        model=model,
        aws_secret_access_key=connection.secret_access_key,
        aws_region_name=connection.region,
        aws_access_key_id=connection.access_key_id,
    )


def test_workflow_with_bedrock_document_embedder(mock_embedding_executor):
    connection = connections.AWS(
        id=str(uuid.uuid4()),
        access_key_id="your_access_key_id",
        secret_access_key="your_secret_access_key",
        region="us-east-1",
    )
    model = "amazon.titan-embed-text-v1"
    wf_bedrock_ai = Workflow(
        id=str(uuid.uuid4()),
        flow=Flow(
            nodes=[
                BedrockDocumentEmbedder(
                    name="BedrockDocumentEmbedder",
                    connection=connection,
                    model=model,
                ),
            ],
        ),
    )
    document = [Document(content="I love pizza!")]
    input = {"documents": document}
    response = wf_bedrock_ai.run(
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
    expected_output = {wf_bedrock_ai.flow.nodes[0].id: expected_result}
    assert response == RunnableResult(
        status=RunnableStatus.SUCCESS,
        input=input,
        output=expected_output,
    )
    assert response.output == expected_output
    mock_embedding_executor.assert_called_once_with(
        input=[document[0].content],
        model=model,
        aws_secret_access_key=connection.secret_access_key,
        aws_region_name=connection.region,
        aws_access_key_id=connection.access_key_id,
    )
