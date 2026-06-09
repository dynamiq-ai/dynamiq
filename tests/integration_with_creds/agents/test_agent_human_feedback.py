"""Live tests that an agent passes a NON-EMPTY question to HumanFeedbackTool.

Regression coverage for the bug where the tool's ``input`` template variable
lived only in the description + ``extra="allow"`` and was therefore never
advertised in the model-facing schema. In FUNCTION_CALLING mode the schema set
``additionalProperties: false``, so the model emitted ``{"action": "ask"}`` with
no ``input``, and the ``{{input}}`` template rendered to an empty prompt.

These run a real agent across every inference mode, force it to ask the user a
clarifying question via the tool, and assert the rendered prompt the tool
receives is a non-empty string.
"""

import pytest

from dynamiq import Workflow
from dynamiq.connections import OpenAI as OpenAIConnection
from dynamiq.flows import Flow
from dynamiq.nodes.agents import Agent
from dynamiq.nodes.llms import OpenAI
from dynamiq.nodes.tools.human_feedback import HumanFeedbackTool, InputMethodCallable
from dynamiq.nodes.types import InferenceMode
from dynamiq.runnables import RunnableConfig, RunnableStatus
from dynamiq.utils.logger import logger


class _RecordingInputMethod(InputMethodCallable):
    """Captures every rendered prompt the tool hands to the input method and
    returns a canned answer so the blocking ``ask`` action proceeds."""

    def __init__(self, response: str = "Yes, go ahead with a science-fiction novel."):
        self.response = response
        self.prompts: list[str] = []

    def get_input(self, prompt: str, **kwargs) -> str:
        self.prompts.append(prompt)
        return self.response


@pytest.fixture(scope="module")
def llm_instance():
    return OpenAI(
        connection=OpenAIConnection(),
        model="gpt-5.4-mini",
        max_tokens=5000,
        temperature=0,
    )


@pytest.fixture(scope="module")
def agent_role():
    return (
        "You are an assistant that must always confirm intent before answering. "
        "Before giving any final answer you MUST use the message-sender tool with "
        "action 'ask' to ask the user a clear, specific clarifying question phrased "
        "as a full sentence. Only after the user replies may you produce a final answer."
    )


@pytest.mark.integration
@pytest.mark.flaky(reruns=3)
@pytest.mark.parametrize(
    "inference_mode",
    [
        InferenceMode.DEFAULT,
        InferenceMode.XML,
        InferenceMode.STRUCTURED_OUTPUT,
        InferenceMode.FUNCTION_CALLING,
    ],
    ids=["default", "xml", "structured_output", "function_calling"],
)
def test_human_feedback_tool_receives_nonempty_input(llm_instance, agent_role, inference_mode):
    """Across every inference mode, the agent must hand the tool a non-empty prompt."""
    recorder = _RecordingInputMethod()
    hf_tool = HumanFeedbackTool(input_method=recorder)

    agent = Agent(
        name=f"Human Feedback Agent {inference_mode.value}",
        llm=llm_instance,
        tools=[hf_tool],
        role=agent_role,
        inference_mode=inference_mode,
        max_loops=6,
        verbose=True,
    )

    wf = Workflow(flow=Flow(nodes=[agent]))
    result = wf.run(
        input_data={"input": "I'd like a recommendation for a good book to read."},
        config=RunnableConfig(request_timeout=120),
    )

    assert (
        result.status == RunnableStatus.SUCCESS
    ), f"[{inference_mode.value}] agent run failed: {result.output}"

    logger.info(f"[{inference_mode.value}] recorded prompts: {recorder.prompts}")

    assert recorder.prompts, (
        f"[{inference_mode.value}] agent never asked the user via HumanFeedbackTool "
        f"(input_method was not invoked)."
    )
    for prompt in recorder.prompts:
        assert isinstance(prompt, str) and prompt.strip(), (
            f"[{inference_mode.value}] tool received an empty prompt: {prompt!r}. "
            f"The '{{{{input}}}}' template rendered empty — `input` was not passed by the model."
        )
