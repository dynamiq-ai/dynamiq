import json
import uuid
from io import BytesIO
from unittest import mock
from unittest.mock import ANY

import pytest

from dynamiq import Workflow, flows
from dynamiq.callbacks import TracingCallbackHandler
from dynamiq.callbacks.tracing import RunStatus
from dynamiq.nodes.node import NodeDependency
from dynamiq.nodes.utils import Output
from dynamiq.runnables import RunnableConfig, RunnableResult, RunnableStatus
from dynamiq.utils import format_value
from dynamiq.utils.utils import JsonWorkflowEncoder, encode_bytes


@pytest.fixture()
def anthropic_node_with_dependency(openai_node, anthropic_node):
    anthropic_node.depends = [NodeDependency(openai_node)]
    return anthropic_node


@pytest.fixture()
def output_node(openai_node, anthropic_node_with_dependency):
    return Output(depends=[NodeDependency(node=openai_node), NodeDependency(node=anthropic_node_with_dependency)])


@pytest.fixture()
def wf(openai_node, anthropic_node_with_dependency, output_node):
    return Workflow(
        id=str(uuid.uuid4()),
        flow=flows.Flow(
            nodes=[openai_node, anthropic_node_with_dependency, output_node],
        ),
        version="1",
    )


def test_workflow_with_depend_nodes_with_tracing(
    wf,
    openai_node,
    anthropic_node_with_dependency,
    output_node,
    mock_llm_response_text,
    mock_llm_executor,
):
    file_name = "file.txt"
    bytes_content = b"file content"
    bytes_content_non_utf8 = b"\xff\xfb\x90\xc4\x00\x00\n\xddu\x15\xe1\x84Z\xe1\xb9"
    file = BytesIO(bytes_content)
    file.name = file_name
    file_without_name = BytesIO(bytes_content)
    files = [file, file_without_name]
    input_data = {"a": 1, "b": {"files": files}, "c": bytes_content, "d": bytes_content_non_utf8}

    tags = ["test1", "test2"]
    tracing = TracingCallbackHandler(tags=tags)

    response = wf.run(
        input_data=input_data,
        config=RunnableConfig(callbacks=[tracing]),
    )
    expected_input_openai = input_data
    expected_tracing_input = {
        "a": 1,
        "b": {"files": [file_name, bytes_content.decode()]},
        "c": encode_bytes(bytes_content),
        "d": encode_bytes(bytes_content_non_utf8),
    }
    expected_output_openai = {"content": mock_llm_response_text, "tool_calls": None}
    expected_result_openai = RunnableResult(
        status=RunnableStatus.SUCCESS,
        input=expected_input_openai,
        output=expected_output_openai,
    )
    expected_input_anthropic = input_data | {openai_node.id: expected_result_openai.to_tracing_depend_dict()}
    expected_output_anthropic = {"content": mock_llm_response_text, "tool_calls": None}
    expected_result_anthropic = RunnableResult(
        status=RunnableStatus.SUCCESS,
        input=expected_input_anthropic,
        output=expected_output_anthropic,
    )

    expected_input_output = expected_input_anthropic | {
        anthropic_node_with_dependency.id: expected_result_anthropic.to_tracing_depend_dict()
    }
    expected_result_output = RunnableResult(
        status=RunnableStatus.SUCCESS,
        input=expected_input_output,
        output=expected_input_output,
    )

    expected_output = {
        openai_node.id: expected_result_openai.to_dict(skip_format_types={BytesIO, bytes}),
        anthropic_node_with_dependency.id: expected_result_anthropic.to_dict(skip_format_types={BytesIO, bytes}),
        output_node.id: expected_result_output.to_dict(skip_format_types={BytesIO, bytes}),
    }
    expected_openai_messages = openai_node.prompt.format_messages(**expected_input_openai)
    expected_anthropic_messages = anthropic_node_with_dependency.prompt.format_messages(**expected_input_anthropic)

    assert response == RunnableResult(
        status=RunnableStatus.SUCCESS, input=input_data, output=expected_output
    )
    assert mock_llm_executor.call_count == 2
    assert mock_llm_executor.call_args_list == [
        mock.call(
            tools=None,
            tool_choice=None,
            model=openai_node.model,
            messages=expected_openai_messages,
            stream=False,
            temperature=openai_node.temperature,
            max_tokens=1000,
            stop=None,
            seed=None,
            presence_penalty=None,
            frequency_penalty=None,
            top_p=None,
            client=ANY,
            response_format=None,
            drop_params=True,
        ),
        mock.call(
            tools=None,
            tool_choice=None,
            model=anthropic_node_with_dependency.model,
            messages=expected_anthropic_messages,
            stream=False,
            temperature=anthropic_node_with_dependency.temperature,
            max_tokens=1000,
            stop=None,
            seed=None,
            presence_penalty=None,
            frequency_penalty=None,
            top_p=None,
            api_key=anthropic_node_with_dependency.connection.api_key,
            response_format=None,
            drop_params=True,
        ),
    ]

    tracing_runs = list(tracing.runs.values())
    assert len(tracing_runs) == 5
    wf_run = tracing_runs[0]
    assert wf_run.metadata["workflow"]["id"] == wf.id
    assert wf_run.metadata["workflow"]["version"] == wf.version
    assert wf_run.metadata["host"]
    assert wf_run.output == format_value(expected_output)
    assert wf_run.status == RunStatus.SUCCEEDED
    assert wf_run.tags == tags
    flow_run = tracing_runs[1]
    assert flow_run.metadata["flow"]["id"] == wf.flow.id
    assert flow_run.metadata["host"]
    assert flow_run.parent_run_id == wf_run.id
    assert flow_run.output == format_value(expected_output)
    assert flow_run.status == RunStatus.SUCCEEDED
    assert flow_run.tags == tags
    openai_run = tracing_runs[2]
    openai_node = openai_node.to_dict()
    openai_node["prompt"]["messages"] = expected_openai_messages
    assert openai_run.metadata["node"] == openai_node
    assert openai_run.metadata["host"]
    assert openai_run.metadata.get("usage")
    assert openai_run.parent_run_id == flow_run.id
    assert openai_run.input == expected_tracing_input
    assert openai_run.output == format_value(expected_output_openai)
    assert openai_run.status == RunStatus.SUCCEEDED
    assert openai_run.tags == tags
    anthropic_run = tracing_runs[3]
    anthropic_node = anthropic_node_with_dependency.to_dict()
    anthropic_node["prompt"]["messages"] = expected_anthropic_messages
    assert anthropic_run.metadata["node"] == anthropic_node
    assert anthropic_run.metadata["host"]
    assert anthropic_run.metadata.get("usage")
    assert anthropic_run.parent_run_id == flow_run.id
    assert anthropic_run.input == format_value(expected_input_anthropic)
    assert anthropic_run.output == format_value(expected_output_anthropic)
    assert anthropic_run.status == RunStatus.SUCCEEDED
    assert anthropic_run.tags == tags
    output_node_run = tracing_runs[4]
    assert output_node_run.metadata["node"] == output_node.to_dict()
    assert output_node_run.metadata["host"]
    assert output_node_run.metadata.get("usage") is None
    assert output_node_run.parent_run_id == flow_run.id
    assert output_node_run.input == format_value(expected_input_output)
    assert output_node_run.output == format_value(expected_input_output)
    assert output_node_run.status == RunStatus.SUCCEEDED
    assert output_node_run.tags == tags


def test_workflow_with_depend_nodes_and_depend_fail(
    wf,
    openai_node,
    anthropic_node_with_dependency,
    output_node,
    mock_llm_response_text,
    mock_llm_executor,
):
    input_data = {"a": 1}
    tracing = TracingCallbackHandler()
    error = ValueError("Error")
    mock_llm_executor.side_effect = error

    response = wf.run(
        input_data=input_data,
        config=RunnableConfig(callbacks=[tracing]),
    )
    expected_input_openai = input_data
    expected_output_openai = None
    expected_result_openai = RunnableResult(
        status=RunnableStatus.FAILURE,
        input=expected_input_openai,
        output={"content": "Error", "error_type": type(error).__name__, "recoverable": False},
    )
    expected_input_anthropic = input_data | {openai_node.id: expected_result_openai.to_tracing_depend_dict()}
    expected_output_anthropic = None
    expected_result_anthropic = RunnableResult(
        status=RunnableStatus.SKIP,
        input=expected_input_anthropic,
        output={
            "content": f"Dependency {openai_node.id}: failed",
            "error_type": "NodeFailedException",
            "recoverable": False,
        },
    )

    expected_input_output_node = expected_input_anthropic | {
        anthropic_node_with_dependency.id: expected_result_anthropic.to_tracing_depend_dict()
    }
    expected_output_output_node = None
    expected_result_output_node = RunnableResult(
        status=RunnableStatus.SKIP,
        input=expected_input_output_node,
        output={
            "content": f"Dependency {openai_node.id}: failed",
            "error_type": "NodeFailedException",
            "recoverable": False,
        },
    )

    expected_output = {
        openai_node.id: expected_result_openai.to_dict(skip_format_types={BytesIO}),
        anthropic_node_with_dependency.id: expected_result_anthropic.to_dict(skip_format_types={BytesIO}),
        output_node.id: expected_result_output_node.to_dict(skip_format_types={BytesIO}),
    }
    expected_openai_messages = openai_node.prompt.format_messages(**expected_input_openai)

    assert response == RunnableResult(
        status=RunnableStatus.SUCCESS, input=input_data, output=expected_output
    )
    assert mock_llm_executor.call_count == 1
    assert mock_llm_executor.call_args_list == [
        mock.call(
            tools=None,
            tool_choice=None,
            model=openai_node.model,
            messages=expected_openai_messages,
            stream=False,
            temperature=openai_node.temperature,
            client=ANY,
            max_tokens=1000,
            stop=None,
            seed=None,
            presence_penalty=None,
            frequency_penalty=None,
            top_p=None,
            response_format=None,
            drop_params=True,
        )
    ]

    tracing_runs = list(tracing.runs.values())
    assert len(tracing_runs) == 5
    wf_run = tracing_runs[0]
    assert wf_run.metadata["workflow"]["id"] == wf.id
    assert wf_run.metadata["workflow"]["version"] == wf.version
    assert wf_run.metadata["host"]
    assert wf_run.output == expected_output
    assert wf_run.status == RunStatus.SUCCEEDED
    assert wf_run.tags == []
    flow_run = tracing_runs[1]
    assert flow_run.metadata["flow"]["id"] == wf.flow.id
    assert flow_run.parent_run_id == wf_run.id
    assert flow_run.output == expected_output
    assert flow_run.status == RunStatus.SUCCEEDED
    assert flow_run.tags == []
    openai_run = tracing_runs[2]
    openai_node = openai_node.to_dict()
    openai_node["prompt"]["messages"] = expected_openai_messages
    assert openai_run.metadata["node"] == openai_node
    assert openai_run.metadata["host"]
    assert openai_run.metadata.get("usage") is None
    assert openai_run.parent_run_id == flow_run.id
    assert openai_run.input == expected_input_openai
    assert openai_run.output == expected_output_openai
    assert openai_run.error == {
        "message": str(error),
        "traceback": ANY,
    }
    assert openai_run.status == RunStatus.FAILED
    assert openai_run.tags == []
    anthropic_run = tracing_runs[3]
    assert anthropic_run.metadata["node"] == anthropic_node_with_dependency.to_dict()
    assert anthropic_run.metadata["host"]
    assert anthropic_run.metadata.get("usage") is None
    assert anthropic_run.metadata["skip"] == {
        "failed_dependency": anthropic_node_with_dependency.depends[0].to_dict(),
    }
    assert anthropic_run.parent_run_id == flow_run.id
    assert anthropic_run.input == expected_input_anthropic
    assert anthropic_run.output == expected_output_anthropic
    assert anthropic_run.status == RunStatus.SKIPPED
    assert anthropic_run.tags == []
    output_node_run = tracing_runs[4]
    assert output_node_run.metadata["node"] == output_node.to_dict()
    assert output_node_run.metadata["host"]
    assert output_node_run.metadata.get("usage") is None
    assert output_node_run.metadata["skip"] == {
        "failed_dependency": output_node.depends[0].to_dict(),
    }
    assert output_node_run.parent_run_id == flow_run.id
    assert output_node_run.input == expected_input_output_node
    assert output_node_run.output == expected_output_output_node
    assert output_node_run.status == RunStatus.SKIPPED
    assert output_node_run.tags == []


def test_workflow_with_failed_flow(
    openai_node,
    mock_llm_response_text,
    mock_llm_executor,
    mocker,
):
    wf = Workflow(flow=flows.Flow(nodes=[openai_node]))
    input_data = {"a": 1, "b": 2}
    tracing = TracingCallbackHandler()

    error = ValueError("Error")
    mocker.patch("dynamiq.flows.flow.Flow._get_nodes_ready_to_run", side_effect=error)
    response = wf.run(
        input_data=input_data,
        config=RunnableConfig(callbacks=[tracing]),
    )

    assert response == RunnableResult(status=RunnableStatus.FAILURE, input=input_data)
    tracing_runs = list(tracing.runs.values())
    assert len(tracing_runs) == 2
    wf_run = tracing_runs[0]
    assert wf_run.metadata["workflow"]["id"] == wf.id
    assert wf_run.output is None
    assert wf_run.status == RunStatus.FAILED
    flow_run = tracing_runs[1]
    assert flow_run.metadata["flow"]["id"] == wf.flow.id
    assert flow_run.parent_run_id == wf_run.id
    assert flow_run.output is None
    assert flow_run.status == RunStatus.FAILED


def test_workflow_with_input_mappings(
    openai_node,
    mock_llm_response_text,
    mock_llm_executor,
):
    def get_multiplied_value_a(inputs, outputs):
        return inputs["a"] * 10

    output_node = (
        Output()
        .inputs(
            test="a",
            openai=openai_node.outputs.content,
            is_openai_output=lambda inputs, outputs: bool(outputs[openai_node.id]["content"]),
            multiplied_value_a=get_multiplied_value_a,
        )
        .depends_on(openai_node)
    )
    wf = Workflow(flow=flows.Flow(nodes=[openai_node, output_node]))
    input_data = {"a": 1, "b": 2}
    tracing = TracingCallbackHandler()

    response = wf.run(
        input_data=input_data,
        config=RunnableConfig(callbacks=[tracing]),
    )

    expected_input_openai = input_data
    expected_output_openai = {"content": mock_llm_response_text, "tool_calls": None}
    expected_result_openai = RunnableResult(
        status=RunnableStatus.SUCCESS,
        input=expected_input_openai,
        output=expected_output_openai,
    )
    expected_input_output_node = (
        input_data
        | {openai_node.id: expected_result_openai.to_tracing_depend_dict()}
        | dict(test="a", openai=mock_llm_response_text, is_openai_output=True, multiplied_value_a=input_data["a"] * 10)
    )
    expected_output_output_node = expected_input_output_node
    expected_result_output_node = RunnableResult(
        status=RunnableStatus.SUCCESS,
        input=expected_input_output_node,
        output=expected_output_output_node,
    )

    expected_output = {
        openai_node.id: expected_result_openai.to_dict(),
        output_node.id: expected_result_output_node.to_dict(),
    }

    assert response == RunnableResult(status=RunnableStatus.SUCCESS, input=input_data, output=expected_output)
    assert json.dumps({"runs": [run.to_dict() for run in tracing.runs.values()]}, cls=JsonWorkflowEncoder)
