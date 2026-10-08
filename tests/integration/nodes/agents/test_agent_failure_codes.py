import uuid

import pytest
from litellm import ModelResponse

from dynamiq import Workflow, connections
from dynamiq.flows import Flow
from dynamiq.nodes.agents import Agent
from dynamiq.nodes.llms import OpenAI
from dynamiq.nodes.types import Behavior, InferenceMode
from dynamiq.runnables import RunnableErrorCode, RunnableStatus


def _completion_mock(mocker, replies):
    """Patch the LLM call to return each (content, finish_reason) reply in turn, repeating the last one."""
    replies = list(replies)

    def response(*args, **kwargs):
        content, finish_reason = replies.pop(0) if len(replies) > 1 else replies[0]
        model_r = ModelResponse()
        model_r["choices"][0]["message"]["content"] = content
        model_r["choices"][0]["finish_reason"] = finish_reason
        return model_r

    return mocker.patch("dynamiq.nodes.llms.base.BaseLLM._completion", side_effect=response)


def _agent(**kwargs):
    llm = OpenAI(model="gpt-4o-mini", connection=connections.OpenAI(id=str(uuid.uuid4()), api_key="api-key"))
    return Agent(name="Agent", llm=llm, tools=[], **kwargs)


def _run(agent):
    result = Workflow(flow=Flow(nodes=[agent])).run(input_data={"input": "hi"})
    return result, result.to_dict()["output"][agent.id]


def test_refusal_fails_agent_with_model_refusal_code(mocker):
    completion = _completion_mock(mocker, [("", "content_filter")])
    agent = _agent()

    result, agent_result = _run(agent)

    assert result.status == RunnableStatus.FAILURE
    assert agent_result["error"]["type"] == "LLMContentFilteredError"
    assert agent_result["error"]["code"] == RunnableErrorCode.MODEL_REFUSAL.value
    assert result.error.failed_nodes[0].error_code == RunnableErrorCode.MODEL_REFUSAL.value
    assert completion.call_count == 1


@pytest.mark.parametrize(
    "inference_mode",
    [InferenceMode.DEFAULT, InferenceMode.XML, InferenceMode.FUNCTION_CALLING, InferenceMode.STRUCTURED_OUTPUT],
)
def test_consecutive_empty_completions_fail_agent(mocker, inference_mode):
    completion = _completion_mock(mocker, [("", "stop")])
    agent = _agent(inference_mode=inference_mode, max_loops=10)

    result, agent_result = _run(agent)

    assert result.status == RunnableStatus.FAILURE
    assert agent_result["error"]["type"] == "EmptyCompletionError"
    assert agent_result["error"]["code"] == RunnableErrorCode.EMPTY_COMPLETION.value
    assert completion.call_count == agent.max_consecutive_empty_completions


def test_empty_completion_limit_is_configurable(mocker):
    completion = _completion_mock(mocker, [("   ", "stop")])
    agent = _agent(max_consecutive_empty_completions=1)

    result, agent_result = _run(agent)

    assert agent_result["error"]["code"] == RunnableErrorCode.EMPTY_COMPLETION.value
    assert completion.call_count == 1


def test_non_empty_reply_resets_empty_completion_count(mocker):
    _completion_mock(
        mocker,
        [("", "stop"), ("", "stop"), ("no format", "stop"), ("", "stop"), ("Thought: done\nAnswer: 42", "stop")],
    )
    agent = _agent(max_consecutive_empty_completions=3)

    result, agent_result = _run(agent)

    assert result.status == RunnableStatus.SUCCESS
    assert agent_result["output"]["content"] == "42"


def test_max_loops_failure_carries_code(mocker):
    _completion_mock(mocker, [("no format", "stop")])
    agent = _agent(max_loops=2, behaviour_on_max_loops=Behavior.RAISE)

    result, agent_result = _run(agent)

    assert result.status == RunnableStatus.FAILURE
    assert agent_result["error"]["code"] == RunnableErrorCode.MAX_LOOPS_EXCEEDED.value


def test_generic_llm_failure_has_no_code(mocker):
    mocker.patch("dynamiq.nodes.llms.base.BaseLLM._completion", side_effect=RuntimeError("boom"))
    agent = _agent()

    result, agent_result = _run(agent)

    assert result.status == RunnableStatus.FAILURE
    assert agent_result["error"]["type"] == "ValueError"
    assert agent_result["error"]["code"] is None
    assert result.error.failed_nodes[0].error_code is None
