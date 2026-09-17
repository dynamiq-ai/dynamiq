"""A FUNCTION_CALLING final answer must never lose part of itself silently.

Seen live: a gpt-5.4-mini run returned a JSON-in-string answer that stopped right
after ``"observations":[...],``, with the rest of the object gone. The tool-call
arguments were valid JSON, so the model had closed the ``answer`` string early.
Anything that followed in a sibling key was discarded without a trace by
``FinalAnswerArguments``, and a cut-off or malformed argument string is repaired by
a partial parse that drops everything after the error. Both now come back to the
model as recoverable errors instead of ending the run.
"""

import json
import uuid

import pytest
from litellm import ModelResponse
from litellm.types.utils import ChatCompletionMessageToolCall, Function
from litellm.types.utils import Message as LiteLLMMessage

from dynamiq import connections, prompts
from dynamiq.nodes.agents import Agent
from dynamiq.nodes.agents.agent import FunctionCall, ToolCall
from dynamiq.nodes.agents.exceptions import ActionParsingException
from dynamiq.nodes.llms import OpenAI
from dynamiq.nodes.types import Behavior, InferenceMode
from dynamiq.runnables import RunnableStatus

COMPLETE_ANSWER = json.dumps({"candidates": [{"title": "a"}], "observations": [{"note": "b"}], "silent_scan": {}})


def _make_agent(**kwargs) -> Agent:
    return Agent(
        name="agent",
        llm=OpenAI(
            model="gpt-4o",
            connection=connections.OpenAI(id=str(uuid.uuid4()), api_key="fake-key"),
            prompt=prompts.Prompt(messages=[prompts.Message(role="user", content="{{input}}")]),
        ),
        tools=[],
        inference_mode=InferenceMode.FUNCTION_CALLING,
        **kwargs,
    )


def _final_answer_call(arguments: str | dict) -> dict:
    return {"id": "call_1", "type": "function", "function": {"name": "provide_final_answer", "arguments": arguments}}


class _LLMResult:
    def __init__(self, *tool_calls: dict):
        self.output = {"content": None, "tool_calls": list(tool_calls)}


def test_complete_final_answer_is_returned_verbatim():
    agent = _make_agent()
    args = json.dumps({"thought": "done", "answer": COMPLETE_ANSWER})

    thought, action, answer = agent._handle_function_calling_mode(_LLMResult(_final_answer_call(args)), loop_num=1)

    assert (thought, action, answer) == ("done", "final_answer", COMPLETE_ANSWER)


def test_answer_spilling_into_sibling_keys_is_rejected():
    """The live shape: `answer` ends after `observations`, `silent_scan` sits next to it."""
    agent = _make_agent()
    args = json.dumps(
        {
            "thought": "done",
            "answer": '{"candidates":[{"title":"a"}],"observations":[{"note":"b"}],',
            "silent_scan": {"checked": 3},
        }
    )

    with pytest.raises(ActionParsingException, match=r"unexpected top-level fields: \['silent_scan'\]") as exc:
        agent._handle_function_calling_mode(_LLMResult(_final_answer_call(args)), loop_num=1)
    assert exc.value.recoverable


def test_final_answer_cut_off_by_max_tokens_is_rejected():
    """finish_reason == "length" leaves an unterminated argument string."""
    agent = _make_agent()
    full = json.dumps({"thought": "done", "answer": {"candidates": [1, 2], "observations": [3], "silent_scan": {}}})
    truncated = full[: full.index('"silent_scan"') + 5]

    with pytest.raises(ActionParsingException, match="cut off or malformed") as exc:
        agent._handle_function_calling_mode(_LLMResult(_final_answer_call(truncated)), loop_num=1)
    assert exc.value.recoverable


def test_malformed_final_answer_that_partially_parses_is_rejected():
    """An unescaped quote ends the partial parse early: without the flag, the tail would vanish."""
    malformed = '{"thought":"done","answer":"{\\"candidates\\":[1],\\"observations\\":[2],"silent_scan":{}}"}'
    call = FunctionCall.model_validate({"name": "provide_final_answer", "arguments": malformed})
    assert call.arguments["answer"] == '{"candidates":[1],"observations":[2],'
    assert call.arguments_incomplete

    with pytest.raises(ActionParsingException, match="cut off or malformed"):
        call.parse_as_final_answer()


def test_partial_parse_still_serves_regular_tool_calls():
    """Only the final answer is strict; a tool call keeps the lenient partial parse."""
    call = ToolCall.model_validate(
        {"id": "c", "function": {"name": "some-tool", "arguments": '{"thought":"t","query":"x","limit":'}}
    )

    assert call.function.arguments_incomplete
    assert call.function.parse_as_tool_call().to_action_input() == {"query": "x"}
    assert "arguments_incomplete" not in call.model_dump()["function"]


def test_output_files_is_still_accepted():
    agent = _make_agent()
    args = json.dumps({"thought": "done", "answer": "ok", "output_files": ""})

    _, action, answer = agent._handle_function_calling_mode(_LLMResult(_final_answer_call(args)), loop_num=1)

    assert (action, answer) == ("final_answer", "ok")


def _model_response(arguments: str) -> ModelResponse:
    response = ModelResponse()
    response.choices[0].message = LiteLLMMessage(
        role="assistant",
        content=None,
        tool_calls=[
            ChatCompletionMessageToolCall(
                id=f"call_{uuid.uuid4().hex[:8]}",
                type="function",
                function=Function(name="provide_final_answer", arguments=arguments),
            )
        ],
    )
    return response


def test_agent_asks_again_instead_of_returning_a_split_answer(mocker):
    """End to end: the spilled answer is sent back, and the complete retry is what the run returns."""
    split = json.dumps(
        {
            "thought": "done",
            "answer": '{"candidates":[{"title":"a"}],"observations":[{"note":"b"}],',
            "silent_scan": {},
        }
    )
    complete = json.dumps({"thought": "done", "answer": COMPLETE_ANSWER})
    completion = mocker.patch(
        "dynamiq.nodes.llms.base.BaseLLM._completion",
        side_effect=[_model_response(split), _model_response(complete)],
    )
    agent = _make_agent(max_loops=3, behaviour_on_max_loops=Behavior.RETURN)

    result = agent.run(input_data={"input": "find issues"})

    assert result.status == RunnableStatus.SUCCESS
    assert result.output["content"] == COMPLETE_ANSWER
    assert completion.call_count == 2
    retry_messages = completion.call_args_list[1].kwargs["messages"]
    assert any("unexpected top-level fields" in str(m.get("content")) for m in retry_messages)
