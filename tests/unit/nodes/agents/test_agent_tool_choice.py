"""Which tool_choice a FUNCTION_CALLING agent sends, and what happens when the model answers in plain text."""

import json
from unittest.mock import MagicMock, patch

import pytest

from dynamiq.connections import Anthropic as AnthropicConnection
from dynamiq.nodes.agents import Agent
from dynamiq.nodes.llms import Anthropic
from dynamiq.nodes.types import InferenceMode
from dynamiq.prompts import MessageRole
from dynamiq.runnables import RunnableResult, RunnableStatus
from tests.helpers.agent_hooks import SearchTool, make_llm


def final_answer_call(answer: str) -> dict:
    arguments = json.dumps({"thought": "done", "answer": answer})
    call = {"id": "call_1", "type": "function", "function": {"name": "provide_final_answer", "arguments": arguments}}
    return {"content": "", "tool_calls": [call]}


def run_agent(llm, outputs: list[dict]):
    agent = Agent(name="assistant", llm=llm, tools=[SearchTool()], inference_mode=InferenceMode.FUNCTION_CALLING)
    requests: list[dict] = []

    def run(**kwargs):
        requests.append(kwargs)
        result = MagicMock(spec=RunnableResult)
        result.status = RunnableStatus.SUCCESS
        result.output = outputs[min(len(requests) - 1, len(outputs) - 1)]
        return result

    with patch.object(agent.llm, "run", side_effect=run):
        result = agent.run(input_data={"input": "What is the capital of France?"})
    return result, requests


def claude(model: str, **kwargs) -> Anthropic:
    return Anthropic(model=model, connection=AnthropicConnection(api_key="test-api-key"), max_tokens=4096, **kwargs)


@pytest.mark.parametrize(
    "llm",
    [
        pytest.param(claude("claude-opus-5-5"), id="opus-5-5"),
        pytest.param(claude("claude-sonnet-5-5"), id="sonnet-5-5"),
        pytest.param(claude("claude-fable-5-1"), id="fable-5-1"),
        pytest.param(claude("openrouter/anthropic/claude-sonnet-5.5"), id="openrouter-dotted-sonnet-5.5"),
        pytest.param(claude("claude-sonnet-4-5", thinking_enabled=True, budget_tokens=2048), id="sonnet-4-5-thinking"),
    ],
)
def test_a_tool_call_is_not_forced_on_claude_models_that_reject_it(llm):
    # These models answer a forced tool_choice with a 400, or with thinking switched off.
    result, requests = run_agent(llm, [final_answer_call("Paris")])

    assert result.status == RunnableStatus.SUCCESS
    assert requests[0].get("tool_choice") is None


@pytest.mark.parametrize(
    "llm",
    [
        pytest.param(make_llm(), id="gpt-4o"),
        pytest.param(claude("claude-sonnet-4-5"), id="sonnet-4-5"),
        pytest.param(claude("openrouter/anthropic/claude-sonnet-4.5"), id="openrouter-dotted-sonnet-4.5"),
    ],
)
def test_a_tool_call_is_still_forced_where_the_model_accepts_it(llm):
    _, requests = run_agent(llm, [final_answer_call("Paris")])

    assert requests[0].get("tool_choice") == "required"


def test_a_plain_text_reply_stays_in_history_when_the_model_is_asked_to_call_a_function():
    # Without its own reply in the history, the correction refers to text the model cannot see.
    result, requests = run_agent(make_llm(), [{"content": "Paris."}, final_answer_call("Paris")])

    assert result.status == RunnableStatus.SUCCESS
    retry = requests[1]["prompt"].messages
    assert (retry[-2].role, retry[-2].content) == (MessageRole.ASSISTANT, "Paris.")
    assert "action_input" not in retry[-1].content
    assert "provide_final_answer" in retry[-1].content
