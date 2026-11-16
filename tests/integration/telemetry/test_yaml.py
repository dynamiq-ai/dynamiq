"""A workflow loaded from YAML with ``init_components=True`` emits the same span tree as its Python twin."""

import yaml

from dynamiq import Workflow
from dynamiq.connections import OpenAI as OpenAIConnection
from dynamiq.connections.managers import get_connection_manager
from dynamiq.flows import Flow
from dynamiq.nodes.agents import Agent
from dynamiq.nodes.llms import OpenAI
from dynamiq.nodes.tools.python import Python
from dynamiq.nodes.types import InferenceMode
from dynamiq.runnables import RunnableStatus
from dynamiq.serializers.loaders.yaml import WorkflowYAMLLoader
from tests.helpers.agent_hooks import final_step, tool_step
from tests.integration.telemetry.conftest import OPERATION
from tests.integration.telemetry.test_llm_workflow import build_workflow


def load(path) -> Workflow:
    with get_connection_manager() as connection_manager:
        data = WorkflowYAMLLoader.load(file_path=str(path), connection_manager=connection_manager, init_components=True)
    return Workflow.from_yaml_file_data(file_data=data)


def round_trip(workflow: Workflow, tmp_path) -> Workflow:
    path = tmp_path / "workflow.yaml"
    workflow.to_yaml_file(str(path))
    return load(path)


def identity(spans) -> list[tuple]:
    """What distinguishes a span in the tree, apart from ids that are generated per run."""
    return sorted(
        (
            s.name,
            s.kind.name,
            s.attributes.get(OPERATION),
            s.attributes["dynamiq.run.type"],
            s.attributes.get("dynamiq.node.type"),
            s.attributes.get("dynamiq.node.model"),
            s.attributes["dynamiq.run.status"],
        )
        for s in spans.spans
    )


def test_llm_workflow_round_trip_has_the_same_tree(mock_llm_executor, make_config, exporter, trace, tmp_path):
    python_twin = build_workflow()
    loaded = round_trip(python_twin, tmp_path)

    assert python_twin.run(input_data={"topic": "hello"}, config=make_config()).status == RunnableStatus.SUCCESS
    expected = trace()
    exporter.clear()
    assert loaded.run(input_data={"topic": "hello"}, config=make_config()).status == RunnableStatus.SUCCESS
    actual = trace()

    assert len(expected.spans) == 4
    assert actual.shape() == expected.shape()
    assert identity(actual) == identity(expected)
    # node ids and workflow identity are kept by the YAML, so they appear unchanged
    assert {s.attributes.get("dynamiq.node.id") for s in actual.spans} == {
        s.attributes.get("dynamiq.node.id") for s in expected.spans
    }
    root = actual.one("invoke_workflow Workflow")
    assert root.attributes["dynamiq.workflow.id"] == "wf-1"
    assert root.attributes["dynamiq.workflow.version"] == "3"
    assert root.attributes["dynamiq.flow.id"] == "flow-1"
    assert (
        actual.one("chat gpt-4o").attributes["gen_ai.input.messages"]
        == expected.one("chat gpt-4o").attributes["gen_ai.input.messages"]
    )


def test_handwritten_yaml_matches_the_python_twin(mock_llm_executor, make_config, exporter, trace, tmp_path):
    document = {
        "connections": {"openai-conn": {"type": "dynamiq.connections.OpenAI", "api_key": "test-key"}},
        "nodes": {
            "in": {"type": "dynamiq.nodes.utils.Input", "name": "start"},
            "llm": {
                "type": "dynamiq.nodes.llms.OpenAI",
                "name": "answerer",
                "model": "gpt-4o",
                "connection": "openai-conn",
                "prompt": {
                    "messages": [
                        {"role": "system", "content": "Be brief."},
                        {"role": "user", "content": "Say {{ topic }}"},
                    ]
                },
                "depends": [{"node": "in"}],
            },
            "out": {"type": "dynamiq.nodes.utils.Output", "name": "end", "depends": [{"node": "llm"}]},
        },
        "flows": {"flow-1": {"name": "flow", "nodes": ["in", "llm", "out"]}},
        "workflows": {"wf-1": {"version": "3", "flow": "flow-1"}},
    }
    path = tmp_path / "handwritten.yaml"
    path.write_text(yaml.safe_dump(document))
    loaded = load(path)

    build_workflow().run(input_data={"topic": "hello"}, config=make_config())
    expected = trace()
    exporter.clear()
    assert loaded.run(input_data={"topic": "hello"}, config=make_config()).status == RunnableStatus.SUCCESS
    actual = trace()

    assert actual.shape() == expected.shape()
    assert identity(actual) == identity(expected)


def agent_workflow() -> Workflow:
    tool = Python(
        id="py", name="python-tool", description="Runs code.", code="def run(input_data): return {'content': 'ok'}"
    )
    llm = OpenAI(id="llm", connection=OpenAIConnection(id="conn", api_key="test-key"), model="gpt-4o")
    agent = Agent(
        id="agent",
        name="assistant",
        llm=llm,
        role="help the user",
        tools=[tool],
        inference_mode=InferenceMode.FUNCTION_CALLING,
    )
    return Workflow(id="wf-agent", flow=Flow(id="flow-agent", nodes=[agent]))


def test_agent_workflow_round_trip_has_the_same_tree(script_llm, make_config, exporter, trace, tmp_path):
    steps = [tool_step("python-tool"), final_step("done")]
    script_llm(InferenceMode.FUNCTION_CALLING, steps)
    twin = agent_workflow()
    loaded = round_trip(agent_workflow(), tmp_path)

    assert (
        twin.run(input_data={"input": "hi", "session_id": "S"}, config=make_config()).status == RunnableStatus.SUCCESS
    )
    expected = trace()
    exporter.clear()
    script_llm(InferenceMode.FUNCTION_CALLING, steps)
    assert (
        loaded.run(input_data={"input": "hi", "session_id": "S"}, config=make_config()).status == RunnableStatus.SUCCESS
    )
    actual = trace()

    assert actual.shape() == expected.shape()
    assert actual.shape() == [
        "invoke_workflow Workflow",
        [
            [
                "invoke_agent assistant",
                sorted([["chat gpt-4o", []], ["execute_tool python-tool", []], ["chat gpt-4o", []]], key=repr),
            ]
        ],
    ]
    assert identity(actual) == identity(expected)
    # the session id comes from the nearest enclosing agent, so the workflow span above it has none
    for span in actual.spans:
        if span.name != "invoke_workflow Workflow":
            assert span.attributes["gen_ai.conversation.id"] == "S"
