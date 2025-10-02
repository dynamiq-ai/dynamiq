import uuid
from unittest.mock import ANY

from dynamiq import Workflow, connections
from dynamiq.callbacks import TracingCallbackHandler
from dynamiq.callbacks.tracing import RunStatus
from dynamiq.flows import Flow
from dynamiq.nodes.embedders import OpenAIDocumentEmbedder, OpenAITextEmbedder
from dynamiq.runnables import RunnableConfig, RunnableResult, RunnableStatus
from dynamiq.types import Document
from dynamiq.utils import format_value


def test_workflow_with_openai_text_embedder(mock_embedding_executor_truncate_tracing, mock_embedding_tracing_output):
    connection = connections.OpenAI(
        id=str(uuid.uuid4()),
        api_key="api_key",
    )
    model = "text-embedding-3-small"
    embedder = OpenAITextEmbedder(name="OpenAITextEmbedder", connection=connection, model=model)
    wf_openai_ai = Workflow(
        id=str(uuid.uuid4()),
        flow=Flow(
            nodes=[embedder],
        ),
    )
    tracing = TracingCallbackHandler()
    input = {"query": "I love pizza!"}
    response = wf_openai_ai.run(
        input_data=input,
        config=RunnableConfig(callbacks=[tracing]),
    )

    expected_result = RunnableResult(
        status=RunnableStatus.SUCCESS,
        input=input,
        output={"query": "I love pizza!", **mock_embedding_tracing_output},
    ).to_dict()
    expected_output = {wf_openai_ai.flow.nodes[0].id: expected_result}
    assert response == RunnableResult(
        status=RunnableStatus.SUCCESS,
        input=input,
        output=expected_output,
    )
    assert response.output == expected_output
    tracing_runs = list(tracing.runs.values())
    assert len(tracing_runs) == 3
    wf_run = tracing_runs[0]
    assert wf_run.metadata["workflow"]["id"] == wf_openai_ai.id
    assert wf_run.metadata["workflow"]["version"] == wf_openai_ai.version
    assert wf_run.status == RunStatus.SUCCEEDED
    flow_run = tracing_runs[1]
    assert flow_run.metadata["flow"]["id"] == wf_openai_ai.flow.id
    assert flow_run.parent_run_id == wf_run.id
    assert flow_run.status == RunStatus.SUCCEEDED
    openai_run = tracing_runs[2]
    assert openai_run.parent_run_id == flow_run.id
    assert openai_run.input == input
    assert (
        openai_run.output
        == format_value({"query": "I love pizza!", **mock_embedding_tracing_output}, truncate_enabled=True)[0]
    )
    assert openai_run.status == RunStatus.SUCCEEDED

    mock_embedding_executor_truncate_tracing.assert_called_once_with(
        input=[input["query"]],
        model=model,
        client=ANY,
    )


def test_workflow_with_openai_document_embedder(mock_embedding_executor_truncate_tracing):
    connection = connections.OpenAI(
        id=str(uuid.uuid4()),
        api_key="api_key",
    )
    model = "text-embedding-3-small"
    embedder = OpenAIDocumentEmbedder(
        name="OpenAIDocumentEmbedder",
        connection=connection,
        model=model,
    )
    wf_openai_ai = Workflow(
        id=str(uuid.uuid4()),
        flow=Flow(
            nodes=[embedder],
        ),
    )
    tracing = TracingCallbackHandler()
    document = [Document(content="I love pizza!")]
    input = {"documents": document}
    response = wf_openai_ai.run(
        input_data=input,
        config=RunnableConfig(callbacks=[tracing]),
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
    tracing_runs = list(tracing.runs.values())
    assert len(tracing_runs) == 3
    wf_run = tracing_runs[0]
    assert wf_run.metadata["workflow"]["id"] == wf_openai_ai.id
    assert wf_run.metadata["workflow"]["version"] == wf_openai_ai.version
    assert wf_run.status == RunStatus.SUCCEEDED
    flow_run = tracing_runs[1]
    assert flow_run.metadata["flow"]["id"] == wf_openai_ai.flow.id
    assert flow_run.parent_run_id == wf_run.id
    assert flow_run.status == RunStatus.SUCCEEDED
    openai_run = tracing_runs[2]
    assert openai_run.parent_run_id == flow_run.id
    embedder_tracing_output = {
        **input,
        "meta": {
            "model": "text-embedding-3-small",
            "usage": {"usage": {"completion_tokens": 0, "prompt_tokens": 6, "total_tokens": 6}},
        },
    }
    assert openai_run.output == format_value(embedder_tracing_output, truncate_enabled=True)[0]
    assert openai_run.status == RunStatus.SUCCEEDED
    mock_embedding_executor_truncate_tracing.assert_called_once_with(
        input=[document[0].content],
        model=model,
        client=ANY,
    )
