"""Branch convergence: a Choice splits the flow and both branches rejoin on the Output node.

Exactly one branch runs per execution, so the other is SKIPPED. The workflow Output node
must still produce a result, while conditional branching itself stays strict.
"""

import asyncio

import pytest

from dynamiq import Workflow
from dynamiq.flows import Flow
from dynamiq.nodes.node import InputTransformer, NodeDependency
from dynamiq.nodes.operators import Choice, ChoiceOption, Pass
from dynamiq.nodes.types import ChoiceCondition, ConditionOperator
from dynamiq.nodes.utils import Input, Output
from dynamiq.runnables import RunnableConfig, RunnableStatus

BLOCKED_OPTION = "option-1"
COALESCE = "$.agent.output.content || $.error.output.output"


def _boolean_condition(variable: str, value: bool = True) -> ChoiceCondition:
    return ChoiceCondition(variable=variable, operator=ConditionOperator.BOOLEAN_EQUALS, value=value)


def build_workflow(output_selector: dict, agent_also_depends_on_input: bool = False) -> Workflow:
    """input -> detector -> choice -{option-1: error, default: agent}-> output."""
    input_node = Input(id="input", name="input")
    detector = Pass(
        id="detector",
        name="detector",
        input_transformer=InputTransformer(selector={"prompt_detected": "$.input.output.detected"}),
        depends=[NodeDependency(input_node)],
    )
    choice = Choice(
        id="choice",
        name="choice",
        options=[
            ChoiceOption(id=BLOCKED_OPTION, name=BLOCKED_OPTION, condition=_boolean_condition("$.prompt_detected")),
            ChoiceOption(id="default", name="default", condition=None),
        ],
        input_transformer=InputTransformer(selector={"prompt_detected": "$.detector.output.prompt_detected"}),
        depends=[NodeDependency(detector)],
    )

    agent_depends = [NodeDependency(choice, option="default")]
    if agent_also_depends_on_input:
        # Mirrors the shipped prompt-injection template, where the agent is gated on a
        # Choice option AND additionally depends on the ungated input node.
        agent_depends.append(NodeDependency(input_node))

    agent = Pass(
        id="agent",
        name="agent",
        input_transformer=InputTransformer(selector={"content": "AGENT ANSWER"}),
        depends=agent_depends,
    )
    error = Pass(
        id="error",
        name="error",
        input_transformer=InputTransformer(selector={"output": "BLOCKED"}),
        depends=[NodeDependency(choice, option=BLOCKED_OPTION)],
    )
    output = Output(
        id="output",
        name="output",
        input_transformer=InputTransformer(selector=output_selector),
        depends=[NodeDependency(agent), NodeDependency(error)],
    )
    return Workflow(flow=Flow(nodes=[input_node, detector, choice, agent, error, output]))


def run(workflow: Workflow, detected: bool) -> dict:
    result = workflow.run(input_data={"detected": detected}, config=RunnableConfig())
    return result, result.output or {}


@pytest.mark.parametrize(
    ("detected", "expected"),
    [(False, "AGENT ANSWER"), (True, "BLOCKED")],
)
def test_output_runs_on_either_branch(detected, expected):
    """The Output node must produce the value of whichever branch executed."""
    _, per_node = run(build_workflow({"output": COALESCE}), detected)

    assert per_node["output"]["status"] == RunnableStatus.SUCCESS.value
    assert per_node["output"]["output"] == {"output": expected}


@pytest.mark.parametrize("detected", [False, True])
def test_untaken_branch_is_still_skipped(detected):
    """Relaxing the join must not make both branches run."""
    _, per_node = run(build_workflow({"output": COALESCE}), detected)

    taken, untaken = ("error", "agent") if detected else ("agent", "error")
    assert per_node[taken]["status"] == RunnableStatus.SUCCESS.value
    assert per_node[untaken]["status"] == RunnableStatus.SKIP.value


def test_choice_option_gate_is_still_enforced_with_extra_dependency():
    """Regression: a branch node gated on a Choice option AND depending on an ungated node
    must still be skipped when its option is not selected."""
    _, per_node = run(build_workflow({"output": COALESCE}, agent_also_depends_on_input=True), detected=True)

    assert per_node["agent"]["status"] == RunnableStatus.SKIP.value
    assert per_node["output"]["output"] == {"output": "BLOCKED"}


@pytest.mark.parametrize(
    ("detected", "expected"),
    [(False, "AGENT ANSWER"), (True, "BLOCKED")],
)
def test_output_runs_on_either_branch_async(detected, expected):
    """The async executor validates dependencies on its own path; cover it too."""
    workflow = build_workflow({"output": COALESCE})
    result = asyncio.run(workflow.run_async(input_data={"detected": detected}, config=RunnableConfig()))
    per_node = result.output or {}

    assert per_node["output"]["status"] == RunnableStatus.SUCCESS.value
    assert per_node["output"]["output"] == {"output": expected}


def test_single_dependency_skip_still_propagates():
    """With no successful sibling there is nothing to tolerate: skip still propagates."""
    input_node = Input(id="input", name="input")
    choice = Choice(
        id="choice",
        name="choice",
        options=[
            ChoiceOption(id="go", name="go", condition=_boolean_condition("$.go")),
            ChoiceOption(id="default", name="default", condition=None),
        ],
        input_transformer=InputTransformer(selector={"go": "$.input.output.go"}),
        depends=[NodeDependency(input_node)],
    )
    branch = Pass(
        id="branch",
        name="branch",
        input_transformer=InputTransformer(selector={"v": "B"}),
        depends=[NodeDependency(choice, option="go")],
    )
    output = Output(
        id="output",
        name="output",
        input_transformer=InputTransformer(selector={"output": "$.branch.output.v"}),
        depends=[NodeDependency(branch)],
    )
    workflow = Workflow(flow=Flow(nodes=[input_node, choice, branch, output]))

    result = workflow.run(input_data={"go": False}, config=RunnableConfig())
    assert (result.output or {})["output"]["status"] == RunnableStatus.SKIP.value


def test_output_skips_when_every_dependency_is_skipped():
    input_node = Input(id="input", name="input")
    choice = Choice(
        id="choice",
        name="choice",
        options=[
            ChoiceOption(id="go", name="go", condition=_boolean_condition("$.go")),
            ChoiceOption(id="default", name="default", condition=None),
        ],
        input_transformer=InputTransformer(selector={"go": "$.input.output.go"}),
        depends=[NodeDependency(input_node)],
    )
    first = Pass(
        id="first",
        name="first",
        input_transformer=InputTransformer(selector={"v": "1"}),
        depends=[NodeDependency(choice, option="go")],
    )
    second = Pass(
        id="second",
        name="second",
        input_transformer=InputTransformer(selector={"v": "2"}),
        depends=[NodeDependency(first)],
    )
    output = Output(
        id="output",
        name="output",
        input_transformer=InputTransformer(selector={"output": "$.first.output.v || $.second.output.v"}),
        depends=[NodeDependency(first), NodeDependency(second)],
    )
    workflow = Workflow(flow=Flow(nodes=[input_node, choice, first, second, output]))

    result = workflow.run(input_data={"go": False}, config=RunnableConfig())
    assert (result.output or {})["output"]["status"] == RunnableStatus.SKIP.value


def test_upstream_failure_still_fails_the_flow():
    """Tolerating skips must not hide a real failure."""

    class FailingNode(Pass):
        def execute(self, input_data, config=None, **kwargs):
            raise RuntimeError("node failed")

    input_node = Input(id="input", name="input")
    failing = FailingNode(id="failing", name="failing", depends=[NodeDependency(input_node)])
    healthy = Pass(
        id="healthy",
        name="healthy",
        input_transformer=InputTransformer(selector={"v": "OK"}),
        depends=[NodeDependency(input_node)],
    )
    downstream = Pass(
        id="downstream",
        name="downstream",
        input_transformer=InputTransformer(selector={"v": "D"}),
        depends=[NodeDependency(failing)],
    )
    output = Output(
        id="output",
        name="output",
        input_transformer=InputTransformer(selector={"output": "$.downstream.output.v || $.healthy.output.v"}),
        depends=[NodeDependency(downstream), NodeDependency(healthy)],
    )
    workflow = Workflow(flow=Flow(nodes=[input_node, failing, healthy, downstream, output]))

    result = workflow.run(input_data={}, config=RunnableConfig())

    assert result.status == RunnableStatus.FAILURE
    assert [node.name for node in result.error.failed_nodes] == ["failing"]


def test_parallel_fan_in_is_unaffected():
    """Two dependencies that both succeed keep their existing behaviour."""
    input_node = Input(id="input", name="input")
    left = Pass(
        id="left",
        name="left",
        input_transformer=InputTransformer(selector={"v": "L"}),
        depends=[NodeDependency(input_node)],
    )
    right = Pass(
        id="right",
        name="right",
        input_transformer=InputTransformer(selector={"v": "R"}),
        depends=[NodeDependency(input_node)],
    )
    output = Output(
        id="output",
        name="output",
        input_transformer=InputTransformer(selector={"a": "$.left.output.v", "b": "$.right.output.v"}),
        depends=[NodeDependency(left), NodeDependency(right)],
    )
    workflow = Workflow(flow=Flow(nodes=[input_node, left, right, output]))

    result = workflow.run(input_data={}, config=RunnableConfig())
    assert (result.output or {})["output"]["output"] == {"a": "L", "b": "R"}
