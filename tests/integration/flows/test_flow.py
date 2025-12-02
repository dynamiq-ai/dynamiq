import json
import uuid
from io import BytesIO
from unittest import mock
from unittest.mock import ANY

import pytest

from dynamiq import Workflow, flows
from dynamiq.callbacks import TracingCallbackHandler
from dynamiq.callbacks.tracing import RunStatus
from dynamiq.nodes.exceptions import NodeConditionFailedException, NodeFailedException
from dynamiq.nodes.llms.mistral import Mistral, MistralConnection
from dynamiq.nodes.node import ErrorHandling, InputTransformer, NodeDependency
from dynamiq.nodes.operators import Choice, ChoiceOption
from dynamiq.nodes.tools import Python
from dynamiq.nodes.types import Behavior, ChoiceCondition, ConditionOperator
from dynamiq.nodes.utils import Output
from dynamiq.runnables import RunnableConfig, RunnableResult, RunnableStatus
from dynamiq.runnables.base import RunnableResultError
from dynamiq.utils import format_value
from dynamiq.utils.utils import JsonWorkflowEncoder, encode_bytes


@pytest.fixture()
def openai_node_with_return_behavior(openai_node):
    openai_node.id = "openai"
    openai_node.error_handling.behavior = Behavior.RETURN
    return openai_node


@pytest.fixture()
def anthropic_node_with_dependency(openai_node, anthropic_node):
    anthropic_node.depends = [NodeDependency(openai_node)]
    return anthropic_node


@pytest.fixture()
def anthropic_node_with_success_status_conditional_depend(openai_node_with_return_behavior, anthropic_node):
    anthropic_node.id = "anthropic"
    anthropic_node.error_handling = ErrorHandling(behavior=Behavior.RETURN)
    anthropic_node.depends = [
        NodeDependency(
            node=openai_node_with_return_behavior,
            condition=ChoiceCondition(
                variable="$.status", operator=ConditionOperator.STRING_EQUALS, value=RunnableStatus.SUCCESS.value
            ),
        )
    ]
    return anthropic_node


@pytest.fixture()
def mistral_node_with_failed_status_conditional_depend(openai_node_with_return_behavior, ds_prompt):
    return Mistral(
        id="mistral",
        name="Mistral",
        model="mistral/mistral-large-latest",
        connection=MistralConnection(api_key="test-api-key"),
        prompt=ds_prompt,
        is_postponed_component_init=True,
        error_handling=ErrorHandling(behavior=Behavior.RETURN),
        depends=[
            NodeDependency(
                openai_node_with_return_behavior,
                condition=ChoiceCondition(
                    variable="$.status", operator=ConditionOperator.STRING_EQUALS, value=RunnableStatus.FAILURE.value
                ),
            )
        ],
    )


def choice_with_failed_conditional_depend(openai_node_with_return_behavior):
    success_openai = ChoiceCondition(
        variable=f"$.{openai_node_with_return_behavior.id}.status",
        operator=ConditionOperator.STRING_EQUALS,
        value=RunnableStatus.SUCCESS.value,
    )
    failed_or_skipped_openai = ChoiceCondition(
        variable=f"$.{openai_node_with_return_behavior.id}.status",
        operator=ConditionOperator.STRING_EQUALS,
        value=RunnableStatus.SUCCESS.value,
        is_not=True,
    )

    return Choice(
        options=[
            ChoiceOption(id="success", condition=success_openai),
            ChoiceOption(id="failed_or_skipped", condition=failed_or_skipped_openai),
        ]
    )


@pytest.fixture()
def output_node(openai_node, anthropic_node_with_dependency):
    return Output(depends=[NodeDependency(node=openai_node), NodeDependency(node=anthropic_node_with_dependency)])


@pytest.fixture()
def output_node_with_conditional_dependencies(
    openai_node,
    anthropic_node_with_success_status_conditional_depend,
    mistral_node_with_failed_status_conditional_depend,
):
    return Output(
        depends=[
            NodeDependency(node=openai_node),
            NodeDependency(node=anthropic_node_with_success_status_conditional_depend),
            NodeDependency(node=mistral_node_with_failed_status_conditional_depend),
        ]
    )


@pytest.fixture()
def wf_with_conditional_depend(
    openai_node_with_return_behavior,
    anthropic_node_with_success_status_conditional_depend,
    mistral_node_with_failed_status_conditional_depend,
    output_node_with_conditional_dependencies,
):
    return Workflow(
        id=str(uuid.uuid4()),
        flow=flows.Flow(
            nodes=[
                openai_node_with_return_behavior,
                anthropic_node_with_success_status_conditional_depend,
                mistral_node_with_failed_status_conditional_depend,
                output_node_with_conditional_dependencies,
            ],
        ),
        version="1",
    )


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
    metadata = {"app_id": "0.0.1", "runtime_id": "0.0.1"}
    tracing = TracingCallbackHandler(tags=tags, metadata=metadata)

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
    expected_output_openai = {"content": mock_llm_response_text}
    expected_result_openai = RunnableResult(
        status=RunnableStatus.SUCCESS,
        input=expected_input_openai,
        output=expected_output_openai,
    )
    expected_input_anthropic = input_data | {openai_node.id: expected_result_openai.to_tracing_depend_dict()}
    expected_output_anthropic = {"content": mock_llm_response_text}
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
        openai_node.id: expected_result_openai.to_dict(skip_format_types={BytesIO, bytes}, for_tracing=True),
        anthropic_node_with_dependency.id: expected_result_anthropic.to_dict(
            skip_format_types={BytesIO, bytes}, for_tracing=True
        ),
        output_node.id: expected_result_output.to_dict(skip_format_types={BytesIO, bytes}, for_tracing=True),
    }
    expected_openai_messages = openai_node.prompt.format_messages(**expected_input_openai)
    expected_anthropic_messages = anthropic_node_with_dependency.prompt.format_messages(**expected_input_anthropic)

    assert response == RunnableResult(status=RunnableStatus.SUCCESS, input=input_data, output=expected_output)
    assert mock_llm_executor.call_count == 2
    assert mock_llm_executor.call_args_list == [
        mock.call(
            tools=None,
            tool_choice=None,
            model=openai_node.model,
            messages=expected_openai_messages,
            stream=False,
            temperature=openai_node.temperature,
            max_tokens=None,
            stop=None,
            seed=None,
            presence_penalty=None,
            frequency_penalty=None,
            top_p=None,
            api_key=openai_node.connection.api_key,
            client=ANY,
            response_format=None,
            drop_params=True,
            api_base="https://api.openai.com/v1",
        ),
        mock.call(
            tools=None,
            tool_choice=None,
            model=anthropic_node_with_dependency.model,
            messages=expected_anthropic_messages,
            stream=False,
            temperature=anthropic_node_with_dependency.temperature,
            max_tokens=None,
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
    assert wf_run.output == format_value(expected_output)
    assert wf_run.status == RunStatus.SUCCEEDED
    assert wf_run.tags == tags
    assert metadata.items() <= wf_run.metadata.items()
    flow_run = tracing_runs[1]
    assert flow_run.metadata["flow"]["id"] == wf.flow.id
    assert flow_run.parent_run_id == wf_run.id
    assert flow_run.output == format_value(expected_output)
    assert flow_run.status == RunStatus.SUCCEEDED
    assert flow_run.tags == tags
    assert metadata.items() <= flow_run.metadata.items()
    openai_run = tracing_runs[2]
    openai_node = openai_node.to_dict(for_tracing=True)
    openai_node["prompt"]["messages"] = expected_openai_messages
    assert openai_run.metadata["node"] == openai_node
    assert openai_run.metadata.get("usage")
    assert openai_run.parent_run_id == flow_run.id
    assert openai_run.input == expected_tracing_input
    assert openai_run.output == format_value(expected_output_openai)
    assert openai_run.status == RunStatus.SUCCEEDED
    assert openai_run.tags == tags
    assert metadata.items() <= openai_run.metadata.items()
    anthropic_run = tracing_runs[3]
    anthropic_node = anthropic_node_with_dependency.to_dict(for_tracing=True)
    anthropic_node["prompt"]["messages"] = expected_anthropic_messages
    assert anthropic_run.metadata["node"] == anthropic_node
    assert anthropic_run.metadata.get("usage")
    assert anthropic_run.parent_run_id == flow_run.id
    assert anthropic_run.input == format_value(expected_input_anthropic)
    assert anthropic_run.output == format_value(expected_output_anthropic)
    assert anthropic_run.status == RunStatus.SUCCEEDED
    assert anthropic_run.tags == tags
    assert metadata.items() <= anthropic_run.metadata.items()
    output_node_run = tracing_runs[4]
    assert output_node_run.metadata["node"] == output_node.to_dict(for_tracing=True)
    assert output_node_run.metadata.get("usage") is None
    assert output_node_run.parent_run_id == flow_run.id
    assert output_node_run.input == format_value(expected_input_output)
    assert output_node_run.output == format_value(expected_input_output)
    assert output_node_run.status == RunStatus.SUCCEEDED
    assert output_node_run.tags == tags
    assert output_node_run.executions
    assert metadata.items() <= output_node_run.metadata.items()


@pytest.mark.asyncio
async def test_workflow_with_depend_nodes_with_tracing_async(
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
    metadata = {"app_id": "0.0.1", "runtime_id": "0.0.1"}
    tracing = TracingCallbackHandler(tags=tags, metadata=metadata)

    response = await wf.run(
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
    expected_output_openai = {"content": mock_llm_response_text}
    expected_result_openai = RunnableResult(
        status=RunnableStatus.SUCCESS,
        input=expected_input_openai,
        output=expected_output_openai,
    )
    expected_input_anthropic = input_data | {openai_node.id: expected_result_openai.to_tracing_depend_dict()}
    expected_output_anthropic = {"content": mock_llm_response_text}
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
        openai_node.id: expected_result_openai.to_dict(skip_format_types={BytesIO, bytes}, for_tracing=True),
        anthropic_node_with_dependency.id: expected_result_anthropic.to_dict(
            skip_format_types={BytesIO, bytes}, for_tracing=True
        ),
        output_node.id: expected_result_output.to_dict(skip_format_types={BytesIO, bytes}, for_tracing=True),
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
            max_tokens=None,
            stop=None,
            seed=None,
            presence_penalty=None,
            frequency_penalty=None,
            top_p=None,
            api_key=openai_node.connection.api_key,
            client=ANY,
            response_format=None,
            drop_params=True,
            api_base="https://api.openai.com/v1",
        ),
        mock.call(
            tools=None,
            tool_choice=None,
            model=anthropic_node_with_dependency.model,
            messages=expected_anthropic_messages,
            stream=False,
            temperature=anthropic_node_with_dependency.temperature,
            max_tokens=None,
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
    assert wf_run.status == RunStatus.SUCCEEDED
    assert wf_run.output == format_value(expected_output)
    assert wf_run.tags == tags
    assert metadata.items() <= wf_run.metadata.items()
    flow_run = tracing_runs[1]
    assert flow_run.metadata["flow"]["id"] == wf.flow.id
    assert flow_run.parent_run_id == wf_run.id
    assert flow_run.output == format_value(expected_output)
    assert flow_run.status == RunStatus.SUCCEEDED
    assert flow_run.tags == tags
    assert metadata.items() <= flow_run.metadata.items()
    openai_run = tracing_runs[2]
    openai_node = openai_node.to_dict(for_tracing=True)
    openai_node["prompt"]["messages"] = expected_openai_messages
    assert openai_run.metadata["node"] == openai_node
    assert openai_run.metadata.get("usage")
    assert openai_run.parent_run_id == flow_run.id
    assert openai_run.input == expected_tracing_input
    assert openai_run.output == format_value(expected_output_openai)
    assert openai_run.status == RunStatus.SUCCEEDED
    assert openai_run.tags == tags
    assert metadata.items() <= openai_run.metadata.items()
    anthropic_run = tracing_runs[3]
    anthropic_node = anthropic_node_with_dependency.to_dict(for_tracing=True)
    anthropic_node["prompt"]["messages"] = expected_anthropic_messages
    assert anthropic_run.metadata["node"] == anthropic_node
    assert anthropic_run.metadata.get("usage")
    assert anthropic_run.parent_run_id == flow_run.id
    assert anthropic_run.input == format_value(expected_input_anthropic)
    assert anthropic_run.output == format_value(expected_output_anthropic)
    assert anthropic_run.status == RunStatus.SUCCEEDED
    assert anthropic_run.tags == tags
    assert metadata.items() <= anthropic_run.metadata.items()
    output_node_run = tracing_runs[4]
    assert output_node_run.metadata["node"] == output_node.to_dict(for_tracing=True)
    assert output_node_run.metadata.get("usage") is None
    assert output_node_run.parent_run_id == flow_run.id
    assert output_node_run.input == format_value(expected_input_output)
    assert output_node_run.output == format_value(expected_input_output)
    assert output_node_run.status == RunStatus.SUCCEEDED
    assert output_node_run.tags == tags
    assert output_node_run.executions
    assert metadata.items() <= output_node_run.metadata.items()


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
        output=expected_output_openai,
        error=RunnableResultError(type=type(error), message="Error"),
    )
    expected_input_anthropic = input_data | {openai_node.id: expected_result_openai.to_tracing_depend_dict()}
    expected_output_anthropic = None

    expected_result_anthropic = RunnableResult(
        status=RunnableStatus.SKIP,
        input=expected_input_anthropic,
        output=expected_output_anthropic,
        error=RunnableResultError(
            type=NodeFailedException,
            message=f"Dependency {openai_node.id}: failed",
        ),
    )

    expected_input_output_node = expected_input_anthropic | {
        anthropic_node_with_dependency.id: expected_result_anthropic.to_tracing_depend_dict()
    }
    expected_output_output_node = None
    expected_result_output_node = RunnableResult(
        status=RunnableStatus.SKIP,
        input=expected_input_output_node,
        output=expected_output_output_node,
        error=RunnableResultError(
            type=NodeFailedException,
            message=f"Dependency {openai_node.id}: failed",
        ),
    )

    expected_output = {
        openai_node.id: expected_result_openai.to_dict(skip_format_types={BytesIO}, for_tracing=True),
        anthropic_node_with_dependency.id: expected_result_anthropic.to_dict(
            skip_format_types={BytesIO}, for_tracing=True
        ),
        output_node.id: expected_result_output_node.to_dict(skip_format_types={BytesIO}, for_tracing=True),
    }
    expected_openai_messages = openai_node.prompt.format_messages(**expected_input_openai)

    assert response.status == RunnableStatus.FAILURE
    assert response.input == input_data
    assert response.output == expected_output
    assert response.error is not None
    assert len(response.error.failed_nodes) == 1
    assert response.error.failed_nodes[0].name == openai_node.name

    assert mock_llm_executor.call_count == 1
    assert mock_llm_executor.call_args_list == [
        mock.call(
            tools=None,
            tool_choice=None,
            model=openai_node.model,
            messages=expected_openai_messages,
            stream=False,
            temperature=openai_node.temperature,
            api_key=openai_node.connection.api_key,
            client=ANY,
            max_tokens=None,
            stop=None,
            seed=None,
            presence_penalty=None,
            frequency_penalty=None,
            top_p=None,
            response_format=None,
            drop_params=True,
            api_base="https://api.openai.com/v1",
        )
    ]

    tracing_runs = list(tracing.runs.values())
    assert len(tracing_runs) == 5
    wf_run = tracing_runs[0]
    assert wf_run.metadata["workflow"]["id"] == wf.id
    assert wf_run.metadata["workflow"]["version"] == wf.version
    assert wf_run.output is None
    assert wf_run.status == RunStatus.FAILED
    assert "failed_nodes" in wf_run.metadata
    assert len(wf_run.metadata["failed_nodes"]) == 1
    assert wf_run.tags == []
    flow_run = tracing_runs[1]
    assert flow_run.metadata["flow"]["id"] == wf.flow.id
    assert flow_run.parent_run_id == wf_run.id
    assert flow_run.output is None
    assert flow_run.status == RunStatus.FAILED
    assert "failed_nodes" in flow_run.metadata
    assert len(flow_run.metadata["failed_nodes"]) == 1
    assert flow_run.tags == []
    openai_run = tracing_runs[2]
    openai_node = openai_node.to_dict(for_tracing=True)
    openai_node["prompt"]["messages"] = expected_openai_messages
    assert openai_run.metadata["node"] == openai_node
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
    assert anthropic_run.metadata["node"] == anthropic_node_with_dependency.to_dict(for_tracing=True)
    assert anthropic_run.metadata.get("usage") is None
    assert anthropic_run.metadata["skip"] == {
        "failed_dependency": anthropic_node_with_dependency.depends[0].to_dict(for_tracing=True),
    }
    assert anthropic_run.parent_run_id == flow_run.id
    assert anthropic_run.input == expected_input_anthropic
    assert anthropic_run.output == expected_output_anthropic
    assert anthropic_run.status == RunStatus.SKIPPED
    assert anthropic_run.tags == []
    output_node_run = tracing_runs[4]
    assert output_node_run.metadata["node"] == output_node.to_dict(for_tracing=True)
    assert output_node_run.metadata.get("usage") is None
    assert output_node_run.metadata["skip"] == {
        "failed_dependency": output_node.depends[0].to_dict(for_tracing=True),
    }
    assert output_node_run.parent_run_id == flow_run.id
    assert output_node_run.input == expected_input_output_node
    assert output_node_run.output == expected_output_output_node
    assert output_node_run.status == RunStatus.SKIPPED
    assert output_node_run.tags == []
    assert not output_node_run.executions


@pytest.mark.asyncio
async def test_workflow_with_depend_nodes_and_depend_fail_async(
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

    response = await wf.run(
        input_data=input_data,
        config=RunnableConfig(callbacks=[tracing]),
    )
    expected_input_openai = input_data
    expected_output_openai = None
    expected_result_openai = RunnableResult(
        status=RunnableStatus.FAILURE,
        input=expected_input_openai,
        output=expected_output_openai,
        error=RunnableResultError(type=type(error), message="Error"),
    )
    expected_input_anthropic = input_data | {openai_node.id: expected_result_openai.to_tracing_depend_dict()}
    expected_output_anthropic = None

    expected_result_anthropic = RunnableResult(
        status=RunnableStatus.SKIP,
        input=expected_input_anthropic,
        output=expected_output_anthropic,
        error=RunnableResultError(
            type=NodeFailedException,
            message=f"Dependency {openai_node.id}: failed",
        ),
    )

    expected_input_output_node = expected_input_anthropic | {
        anthropic_node_with_dependency.id: expected_result_anthropic.to_tracing_depend_dict()
    }
    expected_output_output_node = None
    expected_result_output_node = RunnableResult(
        status=RunnableStatus.SKIP,
        input=expected_input_output_node,
        output=expected_output_output_node,
        error=RunnableResultError(
            type=NodeFailedException,
            message=f"Dependency {openai_node.id}: failed",
        ),
    )

    expected_output = {
        openai_node.id: expected_result_openai.to_dict(skip_format_types={BytesIO}, for_tracing=True),
        anthropic_node_with_dependency.id: expected_result_anthropic.to_dict(
            skip_format_types={BytesIO}, for_tracing=True
        ),
        output_node.id: expected_result_output_node.to_dict(skip_format_types={BytesIO}, for_tracing=True),
    }
    expected_openai_messages = openai_node.prompt.format_messages(**expected_input_openai)

    assert response.status == RunnableStatus.FAILURE
    assert response.input == input_data
    assert response.output == expected_output
    assert response.error is not None
    assert len(response.error.failed_nodes) == 1
    assert response.error.failed_nodes[0].name == openai_node.name

    assert mock_llm_executor.call_count == 1
    assert mock_llm_executor.call_args_list == [
        mock.call(
            tools=None,
            tool_choice=None,
            model=openai_node.model,
            messages=expected_openai_messages,
            stream=False,
            temperature=openai_node.temperature,
            api_key=openai_node.connection.api_key,
            client=ANY,
            max_tokens=None,
            stop=None,
            seed=None,
            presence_penalty=None,
            frequency_penalty=None,
            top_p=None,
            response_format=None,
            drop_params=True,
            api_base="https://api.openai.com/v1",
        )
    ]

    tracing_runs = list(tracing.runs.values())
    assert len(tracing_runs) == 5
    wf_run = tracing_runs[0]
    assert wf_run.metadata["workflow"]["id"] == wf.id
    assert wf_run.metadata["workflow"]["version"] == wf.version
    assert wf_run.output is None
    assert wf_run.status == RunStatus.FAILED
    assert "failed_nodes" in wf_run.metadata
    assert len(wf_run.metadata["failed_nodes"]) == 1
    assert wf_run.tags == []
    flow_run = tracing_runs[1]
    assert flow_run.metadata["flow"]["id"] == wf.flow.id
    assert flow_run.parent_run_id == wf_run.id
    assert flow_run.output is None
    assert flow_run.status == RunStatus.FAILED
    assert "failed_nodes" in flow_run.metadata
    assert len(flow_run.metadata["failed_nodes"]) == 1
    assert flow_run.tags == []
    openai_run = tracing_runs[2]
    openai_node = openai_node.to_dict(for_tracing=True)
    openai_node["prompt"]["messages"] = expected_openai_messages
    assert openai_run.metadata["node"] == openai_node
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
    assert anthropic_run.metadata["node"] == anthropic_node_with_dependency.to_dict(for_tracing=True)
    assert anthropic_run.metadata.get("usage") is None
    assert anthropic_run.metadata["skip"] == {
        "failed_dependency": anthropic_node_with_dependency.depends[0].to_dict(for_tracing=True),
    }
    assert anthropic_run.parent_run_id == flow_run.id
    assert anthropic_run.input == expected_input_anthropic
    assert anthropic_run.output == expected_output_anthropic
    assert anthropic_run.status == RunStatus.SKIPPED
    assert anthropic_run.tags == []
    output_node_run = tracing_runs[4]
    assert output_node_run.metadata["node"] == output_node.to_dict(for_tracing=True)
    assert output_node_run.metadata.get("usage") is None
    assert output_node_run.metadata["skip"] == {
        "failed_dependency": output_node.depends[0].to_dict(for_tracing=True),
    }
    assert output_node_run.parent_run_id == flow_run.id
    assert output_node_run.input == expected_input_output_node
    assert output_node_run.output == expected_output_output_node
    assert output_node_run.status == RunStatus.SKIPPED
    assert output_node_run.tags == []
    assert not output_node_run.executions


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

    assert response == RunnableResult(
        status=RunnableStatus.FAILURE,
        input=input_data,
        error=RunnableResultError(
            type=type(error),
            message=str(error),
        ),
    )
    tracing_runs = list(tracing.runs.values())
    assert len(tracing_runs) == 2
    wf_run = tracing_runs[0]
    assert wf_run.metadata["workflow"]["id"] == wf.id
    assert wf_run.output is None
    assert wf_run.error
    assert wf_run.status == RunStatus.FAILED
    flow_run = tracing_runs[1]
    assert flow_run.metadata["flow"]["id"] == wf.flow.id
    assert flow_run.parent_run_id == wf_run.id
    assert flow_run.output is None
    assert flow_run.error
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
    expected_output_openai = {"content": mock_llm_response_text}
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
        openai_node.id: expected_result_openai.to_dict(for_tracing=True),
        output_node.id: expected_result_output_node.to_dict(for_tracing=True),
    }

    assert response == RunnableResult(status=RunnableStatus.SUCCESS, input=input_data, output=expected_output)
    assert json.dumps({"runs": [run.to_dict() for run in tracing.runs.values()]}, cls=JsonWorkflowEncoder)


@pytest.mark.asyncio
async def test_workflow_with_conditional_depend_nodes_with_tracing_async(
    wf_with_conditional_depend,
    openai_node_with_return_behavior,
    mistral_node_with_failed_status_conditional_depend,
    anthropic_node_with_success_status_conditional_depend,
    output_node_with_conditional_dependencies,
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
    metadata = {"app_id": "0.0.1", "runtime_id": "0.0.1"}
    tracing = TracingCallbackHandler(tags=tags, metadata=metadata)

    response = await wf_with_conditional_depend.run(
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
    expected_output_openai = {"content": mock_llm_response_text}
    expected_result_openai = RunnableResult(
        status=RunnableStatus.SUCCESS,
        input=expected_input_openai,
        output=expected_output_openai,
    )
    expected_input_anthropic = expected_input_mistral = input_data | {
        openai_node_with_return_behavior.id: expected_result_openai.to_tracing_depend_dict(),
    }
    expected_output_anthropic = {"content": mock_llm_response_text}
    expected_result_anthropic = RunnableResult(
        status=RunnableStatus.SUCCESS,
        input=expected_input_anthropic,
        output=expected_output_anthropic,
    )
    mistral_error_msg = (
        f"Dependency {openai_node_with_return_behavior.id} result condition "
        f"`{mistral_node_with_failed_status_conditional_depend.depends[0].condition}`: result is false"
    )
    expected_output_mistral = None
    expected_result_mistral = RunnableResult(
        status=RunnableStatus.SKIP,
        input=expected_input_mistral,
        output=expected_output_mistral,
        error=RunnableResultError(type=NodeConditionFailedException, message=mistral_error_msg),
    )

    expected_input_output = input_data | {
        openai_node_with_return_behavior.id: expected_result_openai.to_tracing_depend_dict(for_tracing=True),
        anthropic_node_with_success_status_conditional_depend.id: expected_result_anthropic.to_tracing_depend_dict(
            for_tracing=True
        ),
        mistral_node_with_failed_status_conditional_depend.id: expected_result_mistral.to_tracing_depend_dict(
            for_tracing=True
        ),
    }
    expected_result_output = RunnableResult(
        status=RunnableStatus.SUCCESS,
        input=expected_input_output,
        output=expected_input_output,
    )

    expected_output = {
        openai_node_with_return_behavior.id: expected_result_openai.to_dict(
            skip_format_types={BytesIO, bytes}, for_tracing=True
        ),
        anthropic_node_with_success_status_conditional_depend.id: expected_result_anthropic.to_dict(
            skip_format_types={BytesIO, bytes}, for_tracing=True
        ),
        mistral_node_with_failed_status_conditional_depend.id: expected_result_mistral.to_dict(
            skip_format_types={BytesIO, bytes}, for_tracing=True
        ),
        output_node_with_conditional_dependencies.id: expected_result_output.to_dict(
            skip_format_types={BytesIO, bytes}, for_tracing=True
        ),
    }
    expected_openai_messages = openai_node_with_return_behavior.prompt.format_messages(**expected_input_openai)
    expected_anthropic_messages = anthropic_node_with_success_status_conditional_depend.prompt.format_messages(
        **expected_input_anthropic
    )

    assert response == RunnableResult(status=RunnableStatus.SUCCESS, input=input_data, output=expected_output)
    assert mock_llm_executor.call_count == 2
    assert mock_llm_executor.call_args_list == [
        mock.call(
            tools=None,
            tool_choice=None,
            model=openai_node_with_return_behavior.model,
            messages=expected_openai_messages,
            stream=False,
            temperature=openai_node_with_return_behavior.temperature,
            max_tokens=None,
            stop=None,
            seed=None,
            presence_penalty=None,
            frequency_penalty=None,
            top_p=None,
            api_key=openai_node_with_return_behavior.connection.api_key,
            client=ANY,
            response_format=None,
            drop_params=True,
            api_base="https://api.openai.com/v1",
        ),
        mock.call(
            tools=None,
            tool_choice=None,
            model=anthropic_node_with_success_status_conditional_depend.model,
            messages=expected_anthropic_messages,
            stream=False,
            temperature=anthropic_node_with_success_status_conditional_depend.temperature,
            max_tokens=None,
            stop=None,
            seed=None,
            presence_penalty=None,
            frequency_penalty=None,
            top_p=None,
            api_key=anthropic_node_with_success_status_conditional_depend.connection.api_key,
            response_format=None,
            drop_params=True,
        ),
    ]

    tracing_runs = list(tracing.runs.values())
    assert len(tracing_runs) == 6
    wf_run = tracing_runs[0]
    assert wf_run.metadata["workflow"]["id"] == wf_with_conditional_depend.id
    assert wf_run.metadata["workflow"]["version"] == wf_with_conditional_depend.version
    assert wf_run.output == format_value(expected_output)
    assert wf_run.status == RunStatus.SUCCEEDED
    assert wf_run.tags == tags
    assert metadata.items() <= wf_run.metadata.items()
    flow_run = tracing_runs[1]
    assert flow_run.metadata["flow"]["id"] == wf_with_conditional_depend.flow.id
    assert flow_run.parent_run_id == wf_run.id
    assert flow_run.output == format_value(expected_output)
    assert flow_run.status == RunStatus.SUCCEEDED
    assert flow_run.tags == tags
    assert metadata.items() <= flow_run.metadata.items()
    openai_run = tracing_runs[2]
    openai_node = openai_node_with_return_behavior.to_dict(for_tracing=True)
    openai_node["prompt"]["messages"] = expected_openai_messages
    assert openai_run.metadata["node"] == openai_node
    assert openai_run.metadata.get("usage")
    assert openai_run.parent_run_id == flow_run.id
    assert openai_run.input == expected_tracing_input
    assert openai_run.output == format_value(expected_output_openai)
    assert openai_run.status == RunStatus.SUCCEEDED
    assert openai_run.tags == tags
    assert metadata.items() <= openai_run.metadata.items()
    anthropic_run = (
        tracing_runs[3]
        if tracing_runs[3].metadata["node"]["id"] == anthropic_node_with_success_status_conditional_depend.id
        else tracing_runs[4]
    )
    mistral_run = tracing_runs[3] if anthropic_run != tracing_runs[3] else tracing_runs[4]
    anthropic_node = anthropic_node_with_success_status_conditional_depend.to_dict(for_tracing=True)
    anthropic_node["prompt"]["messages"] = expected_anthropic_messages
    assert anthropic_run.metadata["node"] == anthropic_node
    assert anthropic_run.metadata.get("usage")
    assert anthropic_run.parent_run_id == flow_run.id
    assert anthropic_run.input == format_value(expected_input_anthropic)
    assert anthropic_run.output == format_value(expected_output_anthropic)
    assert anthropic_run.status == RunStatus.SUCCEEDED
    assert anthropic_run.tags == tags
    assert metadata.items() <= anthropic_run.metadata.items()
    mistral_node = mistral_node_with_failed_status_conditional_depend.to_dict(for_tracing=True)
    assert mistral_run.metadata["node"] == mistral_node
    assert mistral_run.metadata.get("usage") is None
    assert mistral_run.metadata["skip"] == {
        "failed_dependency": mistral_node_with_failed_status_conditional_depend.depends[0].to_dict(for_tracing=True),
    }
    assert mistral_run.parent_run_id == flow_run.id
    assert mistral_run.input == format_value(expected_input_mistral)
    assert mistral_run.output == expected_output_mistral
    assert mistral_run.status == RunStatus.SKIPPED
    assert mistral_run.tags == tags
    assert metadata.items() <= mistral_run.metadata.items()
    output_node_run = tracing_runs[5]
    assert output_node_run.metadata["node"] == output_node_with_conditional_dependencies.to_dict(for_tracing=True)
    assert output_node_run.metadata.get("usage") is None
    assert output_node_run.parent_run_id == flow_run.id
    assert output_node_run.input == format_value(expected_input_output)
    assert output_node_run.output == format_value(expected_input_output)
    assert output_node_run.status == RunStatus.SUCCEEDED
    assert output_node_run.tags == tags
    assert output_node_run.executions
    assert metadata.items() <= output_node_run.metadata.items()


@pytest.mark.asyncio
async def test_workflow_with_conditional_depend_nodes_with_return_behavior_and_depend_fail_async(
    openai_node_with_return_behavior,
    mock_llm_executor,
):
    error_cls, error_msg = ValueError, "Error"
    mock_llm_executor.side_effect = error_cls(error_msg)
    success_openai = ChoiceCondition(
        variable=f"$.{openai_node_with_return_behavior.id}.status",
        operator=ConditionOperator.STRING_EQUALS,
        value=RunnableStatus.SUCCESS.value,
    )
    failed_or_skipped_openai = ChoiceCondition(
        variable=f"$.{openai_node_with_return_behavior.id}.status",
        operator=ConditionOperator.STRING_EQUALS,
        value=RunnableStatus.SUCCESS.value,
        is_not=True,
    )
    choice_node = Choice(
        id="choice",
        options=[
            ChoiceOption(id="success", condition=success_openai),
            ChoiceOption(id="failed_or_skipped", condition=failed_or_skipped_openai),
        ],
        depends=[NodeDependency(openai_node_with_return_behavior)],
    )
    python_return_if_openai_success = "Success"
    python_return_if_openai_failed_or_skipped = "Failed"
    python_code = """
def run(input_data):
    choice_output = input_data["choice"]["output"]
    success_option = choice_output["success"]
    failed_or_skipped_option = choice_output["failed_or_skipped"]
    if success_option["status"] == "success":
        return "{python_return_if_openai_success}"
    if failed_or_skipped_option["status"] == "success":
        return "{python_return_if_openai_failed_or_skipped}"
"""
    python_code = python_code.format(
        python_return_if_openai_success=python_return_if_openai_success,
        python_return_if_openai_failed_or_skipped=python_return_if_openai_failed_or_skipped,
    )
    python_node = Python(
        id="python",
        code=python_code,
        depends=[NodeDependency(choice_node)],
    )
    output_node = Output(
        id="output",
        input_transformer=InputTransformer(selector={"content": f"$.{python_node.id}.output.content"}),
        depends=[NodeDependency(python_node)],
    )
    input_data = {}
    wf = Workflow(flow=flows.Flow(nodes=[openai_node_with_return_behavior, choice_node, python_node, output_node]))

    response = await wf.run(input_data=input_data)

    expected_input_openai = input_data
    expected_result_openai = RunnableResult(
        status=RunnableStatus.FAILURE,
        input=expected_input_openai,
        output=None,
        error=RunnableResultError(type=error_cls, message=error_msg),
    )
    expected_input_choice = input_data | {
        openai_node_with_return_behavior.id: expected_result_openai.to_tracing_depend_dict(),
    }
    expected_output_choice = {
        "success": RunnableResult(status=RunnableStatus.FAILURE, input=expected_input_choice, output=False).to_dict(
            for_tracing=True
        ),
        "failed_or_skipped": RunnableResult(
            status=RunnableStatus.SUCCESS, input=expected_input_choice, output=True
        ).to_dict(),
    }
    expected_result_choice = RunnableResult(
        status=RunnableStatus.SUCCESS,
        input=expected_input_choice,
        output=expected_output_choice,
    )
    expected_result_python = RunnableResult(
        status=RunnableStatus.SUCCESS,
        input={
            openai_node_with_return_behavior.id: expected_result_openai.to_tracing_depend_dict(),
            choice_node.id: expected_result_choice.to_tracing_depend_dict(),
        },
        output={"content": python_return_if_openai_failed_or_skipped},
    )

    expected_input_output = input_data | {"content": python_return_if_openai_failed_or_skipped}
    expected_result_output = RunnableResult(
        status=RunnableStatus.SUCCESS,
        input=expected_input_output,
        output=expected_input_output,
    )

    expected_output = {
        openai_node_with_return_behavior.id: expected_result_openai.to_dict(for_tracing=True),
        choice_node.id: expected_result_choice.to_dict(for_tracing=True),
        python_node.id: expected_result_python.to_dict(for_tracing=True),
        output_node.id: expected_result_output.to_dict(for_tracing=True),
    }

    assert response == RunnableResult(status=RunnableStatus.SUCCESS, input=input_data, output=expected_output)
