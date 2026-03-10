import uuid

import pytest

from dynamiq import Workflow, connections
from dynamiq.callbacks import TracingCallbackHandler
from dynamiq.connections.managers import get_connection_manager
from dynamiq.flows import Flow
from dynamiq.nodes.llms import Anthropic
from dynamiq.prompts import Message, Prompt
from dynamiq.runnables import RunnableConfig, RunnableResult, RunnableStatus
from dynamiq.serializers.dumpers.yaml import WorkflowYAMLDumper
from dynamiq.serializers.loaders.yaml import WorkflowYAMLLoader


def get_anthropic_workflow(
    model: str,
    connection: connections.Anthropic,
    **anthropic_kwargs,
):
    wf_anthropic = Workflow(
        id=str(uuid.uuid4()),
        flow=Flow(
            nodes=[
                Anthropic(
                    name="Anthropic",
                    model=model,
                    connection=connection,
                    prompt=Prompt(
                        messages=[
                            Message(
                                role="user",
                                content="What is LLM?",
                            ),
                        ],
                    ),
                    temperature=0.1,
                    **anthropic_kwargs,
                ),
            ],
        ),
    )

    return wf_anthropic


@pytest.mark.parametrize(
    ("model", "expected_model"),
    [
        ("anthropic/claude-opus-4-20250514", "anthropic/claude-opus-4-20250514"),
        ("claude-opus-4-20250514", "anthropic/claude-opus-4-20250514"),
    ],
)
def test_workflow_with_anthropic_llm(mock_llm_response_text, mock_llm_executor, model, expected_model):
    model = model
    connection = connections.Anthropic(
        id=str(uuid.uuid4()),
        api_key="api_key",
    )
    wf_anthropic = get_anthropic_workflow(model=model, connection=connection)

    response = wf_anthropic.run(
        input_data={},
        config=RunnableConfig(callbacks=[TracingCallbackHandler()]),
    )

    expected_result = RunnableResult(
        status=RunnableStatus.SUCCESS,
        input={},
        output={"content": mock_llm_response_text},
    ).to_dict()
    expected_output = {wf_anthropic.flow.nodes[0].id: expected_result}
    assert response == RunnableResult(
        status=RunnableStatus.SUCCESS,
        input={},
        output=expected_output,
    )
    assert response.output == expected_output
    mock_llm_executor.assert_called_once_with(
        tools=None,
        tool_choice=None,
        model=expected_model,
        messages=wf_anthropic.flow.nodes[0].prompt.format_messages(),
        stream=False,
        temperature=wf_anthropic.flow.nodes[0].temperature,
        max_tokens=None,
        stop=None,
        seed=None,
        frequency_penalty=None,
        presence_penalty=None,
        top_p=None,
        api_key=connection.api_key,
        response_format=None,
        drop_params=True,
    )


def test_workflow_with_anthropic_prompt_caching_injection_points(mock_llm_response_text, mock_llm_executor):
    connection = connections.Anthropic(
        id=str(uuid.uuid4()),
        api_key="api_key",
    )
    wf_anthropic = get_anthropic_workflow(
        model="claude-opus-4-20250514",
        connection=connection,
        cache_control={"type": "ephemeral"},
    )

    response = wf_anthropic.run(
        input_data={},
        config=RunnableConfig(callbacks=[TracingCallbackHandler()]),
    )
    assert response.status == RunnableStatus.SUCCESS

    call_kwargs = mock_llm_executor.call_args.kwargs
    assert call_kwargs["cache_control_injection_points"] == [
        {
            "location": "message",
            "index": -1,
            "control": {"type": "ephemeral"},
        }
    ]
    assert call_kwargs["model"] == "anthropic/claude-opus-4-20250514"


def test_workflow_with_anthropic_prompt_caching_injection_points_ttl(mock_llm_response_text, mock_llm_executor):
    connection = connections.Anthropic(
        id=str(uuid.uuid4()),
        api_key="api_key",
    )
    wf_anthropic = get_anthropic_workflow(
        model="claude-opus-4-20250514",
        connection=connection,
        cache_control={"type": "ephemeral", "ttl": "1h"},
    )

    response = wf_anthropic.run(
        input_data={},
        config=RunnableConfig(callbacks=[TracingCallbackHandler()]),
    )
    assert response.status == RunnableStatus.SUCCESS

    call_kwargs = mock_llm_executor.call_args.kwargs
    assert call_kwargs["cache_control_injection_points"] == [
        {
            "location": "message",
            "index": -1,
            "control": {"type": "ephemeral", "ttl": "1h"},
        }
    ]


def test_anthropic_cache_control_yaml_roundtrip(tmp_path):
    connection = connections.Anthropic(
        id="anthropic-conn",
        api_key="api_key",
    )
    wf_anthropic = get_anthropic_workflow(
        model="claude-opus-4-20250514",
        connection=connection,
        cache_control={"type": "ephemeral", "ttl": "1h"},
    )

    yaml_path = tmp_path / "anthropic_cache_control.yaml"
    WorkflowYAMLDumper.dump(yaml_path, wf_anthropic.to_yaml_file_data())

    with get_connection_manager() as cm:
        wf_data = WorkflowYAMLLoader.load(file_path=yaml_path, connection_manager=cm, init_components=True)
        loaded_workflow = Workflow.from_yaml_file_data(file_data=wf_data)

    loaded_node = loaded_workflow.flow.nodes[0]
    assert isinstance(loaded_node, Anthropic)
    assert loaded_node.cache_control is not None
    assert loaded_node.cache_control.model_dump(exclude_none=True) == {"type": "ephemeral", "ttl": "1h"}
