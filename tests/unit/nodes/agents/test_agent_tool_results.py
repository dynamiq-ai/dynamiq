"""What the model gets back from a tool call, in every inference mode."""

from typing import Any, ClassVar, Literal
from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel, Field

from dynamiq.nodes import NodeGroup
from dynamiq.nodes.node import Node
from dynamiq.nodes.types import InferenceMode
from dynamiq.runnables import RunnableConfig, RunnableResult, RunnableStatus
from tests.helpers.agent_hooks import (
    ALL_MODES,
    TOOL_CALLS,
    EmailTool,
    build_agent,
    encode,
    final_step,
    run_scripted,
    tool_step,
)


class CommandInput(BaseModel):
    command: str = Field(..., description="Shell command to run.")


class StatefulShell(Node):
    """Answers like a shell whose state changes between calls: every run reports a new output."""

    group: Literal[NodeGroup.TOOLS] = NodeGroup.TOOLS
    name: str = "shell"
    description: str = "Runs a shell command."
    input_schema: ClassVar[type[CommandInput]] = CommandInput
    output_template: str = "run {n}: {command}"
    runs: int = 0

    def execute(self, input_data: CommandInput, config: RunnableConfig | None = None, **kwargs) -> dict[str, Any]:
        self.runs += 1
        return {"content": self.output_template.format(n=self.runs, command=input_data.command)}


@pytest.mark.parametrize("mode", ALL_MODES)
def test_a_repeated_identical_call_runs_the_tool_again(mode):
    # The second `pytest` follows a fix the agent made, so a stale first result would mislead it.
    shell = StatefulShell()
    steps = [tool_step("shell", command="pytest"), tool_step("shell", command="pytest"), final_step("done")]

    result, recorder = run_scripted(build_agent(tools=[shell]), mode, steps)

    assert result.status == RunnableStatus.SUCCESS
    assert shell.runs == 2
    assert "run 2: pytest" in recorder.sent_text(2)


@pytest.mark.parametrize("mode", ALL_MODES)
def test_template_braces_in_a_tool_result_reach_the_model_verbatim(mode):
    # A Helm template the agent may edit and write back; `.Values.image` alone is not valid Jinja either.
    shell = StatefulShell(output_template="image: {{{{ .Values.image }}}} ({{{{is ok}}}})")
    steps = [tool_step("shell", command="cat chart/templates/deployment.yaml"), final_step("done")]

    result, recorder = run_scripted(build_agent(tools=[shell]), mode, steps)

    assert result.status == RunnableStatus.SUCCESS
    assert "image: {{ .Values.image }} ({{is ok}})" in recorder.sent_text(1)


@pytest.mark.parametrize("mode", [InferenceMode.DEFAULT, InferenceMode.XML])
def test_template_braces_in_a_tool_result_are_sent_to_the_provider_verbatim(mode):
    # Checked on the request the provider receives, after the prompt is rendered, not on the messages before it.
    from litellm import ModelResponse

    shell = StatefulShell(output_template="image: {{{{ .Values.image }}}} ({{{{is ok}}}})")
    agent = build_agent(tools=[shell], inference_mode=mode)
    steps = [tool_step("shell", command="cat chart/templates/deployment.yaml"), final_step("done")]
    sent: list[list[dict]] = []

    def completion(*args, **kwargs):
        sent.append(kwargs["messages"])
        response = ModelResponse()
        response["choices"][0]["message"]["content"] = encode(mode, steps[min(len(sent) - 1, 1)], 0)["content"]
        return response

    with patch("dynamiq.nodes.llms.base.BaseLLM._completion", side_effect=completion):
        result = agent.run(input_data={"input": "show the chart"})

    assert result.status == RunnableStatus.SUCCESS
    assert "image: {{ .Values.image }} ({{is ok}})" in str(sent[1])


def run_with_first_reply(agent, mode, steps, first_content=None):
    """Run ``steps``; the first reply ends at the output limit, optionally with ``first_content`` as its text."""
    calls = {"n": 0}

    def run(**kwargs):
        index = calls["n"]
        calls["n"] += 1
        result = MagicMock(spec=RunnableResult)
        result.status = RunnableStatus.SUCCESS
        result.output = encode(mode, steps[min(index, len(steps) - 1)], index)
        if index == 0:
            result.output["finish_reason"] = "length"
            if first_content is not None:
                result.output["content"] = first_content
        return result

    with patch.object(agent.llm, "run", side_effect=run) as llm_run:
        result = agent.run(input_data={"input": "write the notes"})
    sent = ["\n".join(str(m.content) for m in call.kwargs["prompt"].messages) for call in llm_run.call_args_list]
    return result, sent


def test_a_function_call_cut_off_at_the_output_limit_is_not_run():
    # Partial arguments still parse, so a cut-off write would replace a file or a todo list with a fragment.
    shell = StatefulShell()
    mode = InferenceMode.FUNCTION_CALLING
    steps = [tool_step("shell", command="cat <<EOF > notes.md"), tool_step("shell", command="ls"), final_step("done")]

    result, sent = run_with_first_reply(build_agent(tools=[shell], inference_mode=mode), mode, steps)

    assert result.status == RunnableStatus.SUCCESS
    assert shell.runs == 1  # only the complete call
    assert "cut off" in sent[1]


@pytest.mark.parametrize("mode", [InferenceMode.DEFAULT, InferenceMode.XML, InferenceMode.STRUCTURED_OUTPUT])
def test_a_reply_cut_off_inside_its_action_says_why_it_was_not_run(mode):
    shell = StatefulShell()
    steps = [tool_step("shell", command="cat <<EOF > notes.md"), tool_step("shell", command="ls"), final_step("done")]
    complete = encode(mode, steps[0], 0)["content"]

    agent = build_agent(tools=[shell], inference_mode=mode)

    result, sent = run_with_first_reply(agent, mode, steps, complete[: len(complete) - 12])

    assert result.status == RunnableStatus.SUCCESS
    assert shell.runs == 1
    assert "cut off" in sent[1]


@pytest.mark.parametrize("mode", ALL_MODES)
def test_a_reply_without_text_is_retried(mode):
    # Reasoning models can spend the whole output budget thinking and return no text at all.
    shell = StatefulShell()
    steps = [tool_step("shell", command="ls"), tool_step("shell", command="ls"), final_step("done")]
    calls = {"n": 0}

    def run(**kwargs):
        index = calls["n"]
        calls["n"] += 1
        result = MagicMock(spec=RunnableResult)
        result.status = RunnableStatus.SUCCESS
        result.output = {"content": None} if index == 0 else encode(mode, steps[index], index)
        return result

    agent = build_agent(tools=[shell], inference_mode=mode)
    with patch.object(agent.llm, "run", side_effect=run):
        result = agent.run(input_data={"input": "list the files"})

    assert result.status == RunnableStatus.SUCCESS, result.error
    assert shell.runs == 1


XML_ACTION = "<output><thought>Send the numbers.</thought><action>send-email</action>"


@pytest.mark.parametrize(
    "cut", [XML_ACTION, XML_ACTION + "<action_in", XML_ACTION + "<action_input>"], ids=["no-input", "tag", "open"]
)
def test_an_xml_action_cut_off_before_its_input_is_not_run(cut):
    # XML accepts an action without <action_input> as one that takes none, so a cut-off reply would run the
    # tool with every parameter at its default.
    TOOL_CALLS.clear()
    mode = InferenceMode.XML
    send = tool_step("send-email", to="ceo@example.com", body="Q3")
    steps = [send, send, final_step("done")]

    result, sent = run_with_first_reply(build_agent(tools=[EmailTool()], inference_mode=mode), mode, steps, cut)

    assert result.status == RunnableStatus.SUCCESS
    assert TOOL_CALLS == [("send-email", {"to": "ceo@example.com", "body": "Q3"})]
    assert "cut off" in sent[1]


@pytest.mark.parametrize(
    "nested",
    [
        '<action><send-email>{"to": "ceo@example.com", "body": "Q3"}</send-email></action>',
        "<action><send-email><to>ceo@example.com</to><body>Q3</body></send-email></action>",
        '<action><action>send-email</action><action_input>{"to": "ceo@example.com", "body": "Q3"}</action_input>'
        "</action>",
    ],
    ids=["json", "elements", "wrapped"],
)
def test_an_xml_action_wrapped_in_extra_elements_runs(nested):
    TOOL_CALLS.clear()
    mode = InferenceMode.XML
    steps = [final_step("unused"), final_step("sent")]
    first = f"<output><thought>Send the numbers.</thought>{nested}</output>"

    agent = build_agent(tools=[EmailTool()], inference_mode=mode)
    calls = {"n": 0}

    def run(**kwargs):
        index = calls["n"]
        calls["n"] += 1
        result = MagicMock(spec=RunnableResult)
        result.status = RunnableStatus.SUCCESS
        result.output = encode(mode, steps[min(index, len(steps) - 1)], index)
        if index == 0:
            result.output["content"] = first
        return result

    with patch.object(agent.llm, "run", side_effect=run):
        result = agent.run(input_data={"input": "send the numbers"})

    assert result.status == RunnableStatus.SUCCESS
    assert TOOL_CALLS == [("send-email", {"to": "ceo@example.com", "body": "Q3"})]


RUN_ON = {
    InferenceMode.DEFAULT: "\nObservation: notes.md written\nThought: done\nAnswer: written",
    InferenceMode.XML: "<output><thought>done</thought><answer>written</answer></output>",
    InferenceMode.STRUCTURED_OUTPUT: '{"thought": "done", "action": "finish", "action_input": "written"}',
}


@pytest.mark.parametrize("mode", list(RUN_ON))
def test_a_complete_action_runs_when_the_reply_ran_on_to_the_output_limit(mode):
    # Reasoning models ignore stop sequences and keep writing (a made-up Observation, a second block) until
    # the limit; the action itself is complete and is the reply's only real part.
    shell = StatefulShell()
    steps = [tool_step("shell", command="ls"), final_step("done")]

    agent = build_agent(tools=[shell], inference_mode=mode)

    result, _ = run_with_first_reply(agent, mode, steps, encode(mode, steps[0], 0)["content"] + RUN_ON[mode])

    assert result.status == RunnableStatus.SUCCESS
    assert shell.runs == 1
