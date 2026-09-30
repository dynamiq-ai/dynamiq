"""Agent ``tool_hooks`` survive a YAML dump/load round-trip."""

import os

import yaml

from dynamiq import Workflow
from dynamiq.connections import OpenAI as OpenAIConnection
from dynamiq.connections.managers import get_connection_manager
from dynamiq.flows import Flow
from dynamiq.nodes.agents import Agent
from dynamiq.nodes.agents.hooks import ToolHook
from dynamiq.nodes.llms import OpenAI
from dynamiq.nodes.tools.python import Python
from dynamiq.serializers.loaders.yaml import WorkflowYAMLLoader

HOOKS = [
    ToolHook(
        tools=["python-tool"],
        input_transformer={"selector": {"input": "$.input", "mode": "fast"}},
        output_transformer={"selector": {"content": "$.content"}},
    ),
    ToolHook(tools=["python-tool"], block=True, block_message="Blocked.", stop_agent=True),
]


def _workflow() -> Workflow:
    llm = OpenAI(
        id="llm",
        connection=OpenAIConnection(id="conn", api_key="test-api-key"),
        model="gpt-4o",
        max_tokens=50,
    )
    tool = Python(
        id="py",
        name="python-tool",
        code="def run(input_data): return {'content': 'ok'}",
    )
    agent = Agent(
        id="agent",
        name="hooked",
        llm=llm,
        role="helper",
        tools=[tool],
        tool_hooks=HOOKS,
    )
    return Workflow(id="wf", flow=Flow(id="flow", nodes=[agent]))


def test_tool_hooks_are_dumped_to_yaml(tmp_path):
    path = os.path.join(tmp_path, "wf.yaml")
    _workflow().to_yaml_file(path)

    with open(path) as f:
        dumped = yaml.safe_load(f)

    hooks = dumped["nodes"]["agent"]["tool_hooks"]
    assert hooks[0]["tools"] == ["python-tool"]
    assert hooks[0]["input_transformer"]["selector"] == {
        "input": "$.input",
        "mode": "fast",
    }
    assert hooks[1]["block"] is True and hooks[1]["stop_agent"] is True


def test_tool_hooks_round_trip_through_yaml(tmp_path):
    path = os.path.join(tmp_path, "wf.yaml")
    _workflow().to_yaml_file(path)

    with get_connection_manager() as cm:
        wf_data = WorkflowYAMLLoader.load(file_path=path, connection_manager=cm, init_components=True)
        loaded = Workflow.from_yaml_file_data(file_data=wf_data)

    agent = loaded.flow.nodes[0]
    assert all(isinstance(h, ToolHook) for h in agent.tool_hooks)
    assert agent.tool_hooks == HOOKS
    # and a second dump is stable
    assert agent.to_dict()["tool_hooks"] == _workflow().flow.nodes[0].to_dict()["tool_hooks"]


class DenyForbidden(ToolHook):
    forbidden: str = "DROP TABLE"

    def should_block(self, tool_name, tool_input):
        return self.forbidden in str(tool_input)


def test_python_hook_subclass_round_trips_through_yaml(tmp_path):
    hooks = [DenyForbidden(tools=["python-tool"], forbidden="rm -rf", block_message="No.", on_error="skip")]
    wf = _workflow()
    wf.flow.nodes[0].tool_hooks = hooks
    path = os.path.join(tmp_path, "wf.yaml")
    wf.to_yaml_file(path)

    with open(path) as f:
        dumped = yaml.safe_load(f)
    assert dumped["nodes"]["agent"]["tool_hooks"][0]["type"] == f"{DenyForbidden.__module__}.DenyForbidden"

    with get_connection_manager() as cm:
        wf_data = WorkflowYAMLLoader.load(file_path=path, connection_manager=cm, init_components=True)
        loaded = Workflow.from_yaml_file_data(file_data=wf_data)

    loaded_hook = loaded.flow.nodes[0].tool_hooks[0]
    assert isinstance(loaded_hook, DenyForbidden)
    assert loaded_hook == hooks[0]
    assert loaded_hook.should_block("python-tool", {"cmd": "rm -rf /"}) is True


def test_model_hooks_round_trip_through_yaml(tmp_path):
    from dynamiq.nodes.agents.model_hooks import ModelHook, RegexGuardModelHook, RegexRedactModelHook

    hooks = [
        RegexRedactModelHook(patterns=[r"\S+@\S+"], replacement="[EMAIL]", apply_to="input"),
        RegexGuardModelHook(block_if_matches=["(?i)ignore previous"], on_block="answer", block_message="Refused."),
        ModelHook(block_message="plain"),
    ]
    wf = _workflow()
    wf.flow.nodes[0].model_hooks = hooks
    path = os.path.join(tmp_path, "wf.yaml")
    wf.to_yaml_file(path)

    with open(path) as f:
        dumped = yaml.safe_load(f)["nodes"]["agent"]["model_hooks"]
    assert dumped[0]["type"] == "dynamiq.nodes.agents.model_hooks.RegexRedactModelHook"
    assert "type" not in dumped[2]

    with get_connection_manager() as cm:
        wf_data = WorkflowYAMLLoader.load(file_path=path, connection_manager=cm, init_components=True)
        loaded = Workflow.from_yaml_file_data(file_data=wf_data)

    assert loaded.flow.nodes[0].model_hooks == hooks
    assert [type(h) for h in loaded.flow.nodes[0].model_hooks] == [type(h) for h in hooks]
