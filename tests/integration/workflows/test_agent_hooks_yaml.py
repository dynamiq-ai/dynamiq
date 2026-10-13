"""Agent ``hooks`` survive YAML and ``model_dump`` round trips: top level and nested (agent-as-tool, Map)."""

import os

import pytest
import yaml

from dynamiq import Workflow
from dynamiq.connections import HuggingFace
from dynamiq.connections import OpenAI as OpenAIConnection
from dynamiq.connections.managers import get_connection_manager
from dynamiq.flows import Flow
from dynamiq.nodes.agents import Agent
from dynamiq.nodes.agents.hooks import (
    ALLOW,
    Block,
    CallLimitHook,
    Hook,
    PIIHook,
    PromptInjectionHook,
    RegexHook,
    ToolPolicyHook,
    TransformHook,
)
from dynamiq.nodes.llms import OpenAI
from dynamiq.nodes.operators.operators import Map
from dynamiq.nodes.tools.agent_tool import SubAgentTool
from dynamiq.nodes.tools.python import Python
from dynamiq.serializers.loaders.yaml import WorkflowYAMLLoader, WorkflowYAMLLoaderException
from dynamiq.types.feedback import FeedbackMethod


class DenyForbidden(Hook):
    """A Python hook referenced from YAML by its dotted path."""

    forbidden: str = "DROP TABLE"

    def before_tool(self, ctx, call):
        return Block("No.") if self.forbidden in str(call.input) else ALLOW


def make_hooks() -> list[Hook]:
    return [
        PIIHook(
            entities=["email", "phone"],
            on=["input", "tool_result"],
            restore_in_tools=["python-tool"],
            scope="request",
            live_stream=True,
            inherit=True,
        ),
        RegexHook(presets=["injection_basic"], on=["input"], on_violation="answer", name="no-injection"),
        ToolPolicyHook(
            tools=["python-tool"],
            allow_if={"metadata.role": ["admin", "ops"]},
            approval=True,
            approval_unless={"metadata.role": "admin"},
            feedback_method=FeedbackMethod.STREAM,
            editable_params=["query"],
        ),
        CallLimitHook(tools=["python-tool"], max_per_run=3),
        TransformHook(
            tools=["python-tool"], input_transformer={"selector": {"mode": "fast"}}, from_context={"tenant": "user_id"}
        ),
        PromptInjectionHook(
            detector={"connection": HuggingFace(id="hf-conn", api_key="k")}, on=["tool_result"], on_violation="fail"
        ),
        DenyForbidden(tools=["python-tool"], forbidden="rm -rf", on_error="skip", timeout_seconds=2),
    ]


def make_agent(agent_id="agent", hooks=None) -> Agent:
    return Agent(
        id=agent_id,
        name=f"agent {agent_id}",
        llm=OpenAI(id=f"llm-{agent_id}", connection=OpenAIConnection(id="conn", api_key="k"), model="gpt-4o"),
        role="helper",
        tools=[Python(id=f"py-{agent_id}", name="python-tool", code="def run(input_data): return {'content': 'ok'}")],
        hooks=make_hooks() if hooks is None else hooks,
    )


def round_trip(workflow: Workflow, tmp_path) -> Workflow:
    path = os.path.join(tmp_path, "wf.yaml")
    workflow.to_yaml_file(path)
    with get_connection_manager() as cm:
        data = WorkflowYAMLLoader.load(file_path=path, connection_manager=cm, init_components=True)
        return Workflow.from_yaml_file_data(file_data=data)


def dump_hooks(hooks):
    return [h.model_dump(mode="json") for h in hooks]


def test_every_builtin_and_python_hook_round_trips_through_yaml(tmp_path):
    original = make_agent()
    workflow = Workflow(id="wf", flow=Flow(id="flow", nodes=[original]))
    path = os.path.join(tmp_path, "wf.yaml")
    workflow.to_yaml_file(path)
    with open(path) as f:
        dumped = yaml.safe_load(f)["nodes"]["agent"]["hooks"]
    assert [h["type"] for h in dumped][:6] == [
        "pii",
        "regex",
        "tool_policy",
        "call_limit",
        "transform",
        "prompt_injection",
    ]
    assert dumped[6]["type"].endswith("DenyForbidden") and dumped[6]["forbidden"] == "rm -rf"

    loaded = round_trip(workflow, tmp_path).flow.nodes[0]
    assert [type(h) for h in loaded.hooks] == [type(h) for h in original.hooks]
    assert dump_hooks(loaded.hooks)[:5] == dump_hooks(original.hooks)[:5]
    assert loaded.hooks[1].name == "no-injection" and loaded.hooks[6].timeout_seconds == 2


def test_model_dump_keeps_every_hook_field():
    agent = make_agent()
    dumped = agent.model_dump()["hooks"]
    assert dumped[0]["entities"] == ["email", "phone"] and dumped[0]["restore_in_tools"] == ["python-tool"]
    assert dumped[3]["max_per_run"] == 3 and dumped[6]["forbidden"] == "rm -rf"
    rebuilt = make_agent(hooks=dumped).hooks
    assert [type(h) for h in rebuilt] == [type(h) for h in agent.hooks]
    assert isinstance(rebuilt[5].detector.connection, HuggingFace)


def test_hooks_survive_when_the_agent_is_nested_as_a_tool_and_inside_map(tmp_path):
    child = make_agent("child")
    parent = make_agent("parent", hooks=[RegexHook(patterns=["x"], on=["input"])])
    parent.tools.append(SubAgentTool(id="sub", agent=child, name="Researcher", description="researches"))
    mapper = Map(id="mapper", node=make_agent("mapped"))
    workflow = Workflow(id="wf", flow=Flow(id="flow", nodes=[parent, mapper]))

    loaded = round_trip(workflow, tmp_path)
    loaded_parent = next(n for n in loaded.flow.nodes if n.id == "parent")
    loaded_map = next(n for n in loaded.flow.nodes if n.id == "mapper")
    sub_tool = next(t for t in loaded_parent.tools if isinstance(t, SubAgentTool))

    assert [type(h) for h in loaded_parent.hooks] == [RegexHook]
    assert [type(h) for h in sub_tool.agent.hooks] == [type(h) for h in child.hooks]
    assert [type(h) for h in loaded_map.node.hooks] == [type(h) for h in child.hooks]
    assert dump_hooks(sub_tool.agent.hooks)[:5] == dump_hooks(child.hooks)[:5]
    assert isinstance(sub_tool.agent.hooks[5].detector.connection, HuggingFace)


YAML = """
connections:
  hf-conn:
    type: dynamiq.connections.HuggingFace
    api_key: key
  openai-conn:
    type: dynamiq.connections.OpenAI
    api_key: key
nodes:
  agent:
    type: dynamiq.nodes.agents.Agent
    name: a
    role: r
    llm:
      type: dynamiq.nodes.llms.OpenAI
      connection: openai-conn
      model: gpt-4o
    hooks:
      - type: pii
        entities: [email]
      - type: prompt_injection
        on: [input, tool_result]
        detector: {connection: hf-conn}
        on_violation: answer
{extra}
flows:
  f:
    nodes: [agent]
workflows:
  w:
    flow: f
"""


def load_yaml(tmp_path, extra=""):
    path = os.path.join(tmp_path, "w.yaml")
    with open(path, "w") as f:
        f.write(YAML.replace("{extra}", extra))
    with get_connection_manager() as cm:
        data = WorkflowYAMLLoader.load(file_path=path, connection_manager=cm, init_components=True)
        return Workflow.from_yaml_file_data(file_data=data)


def test_a_hand_written_yaml_resolves_the_detector_connection_by_reference(tmp_path):
    agent = load_yaml(tmp_path).flow.nodes[0]
    assert isinstance(agent.hooks[0], PIIHook)
    detector = agent.hooks[1].detector
    assert isinstance(detector.connection, HuggingFace) and detector.connection.api_key == "key"


@pytest.mark.parametrize(
    "extra, message",
    [
        ("      - type: regex\n        patern: ['x']", "patern"),  # a typo in a field
        ("      - patterns: ['x']", "missing `type`"),  # no type
        ("      - type: piii", "Unknown hook type 'piii'"),
        ("      - type: regex\n        patterns: []", "at least one pattern or preset"),
        (
            "      - type: tool_policy\n        tools: [x]\n        approval: true\n        aproval_unless: {a: b}",
            "aproval_unless",
        ),
    ],
)
def test_wrong_hook_config_fails_at_load_time(tmp_path, extra, message):
    with pytest.raises(Exception, match=message):
        load_yaml(tmp_path, extra)


def test_a_missing_detector_connection_names_the_connection(tmp_path):
    bad = YAML.replace("{extra}", "").replace("detector: {connection: hf-conn}", "detector: {connection: nope}")
    path = os.path.join(tmp_path, "bad.yaml")
    with open(path, "w") as f:
        f.write(bad)
    with get_connection_manager() as cm:
        with pytest.raises(WorkflowYAMLLoaderException, match="Connection 'nope'"):
            WorkflowYAMLLoader.load(file_path=path, connection_manager=cm, init_components=True)
