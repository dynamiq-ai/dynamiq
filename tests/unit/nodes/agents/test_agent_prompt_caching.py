"""Agents turn on prompt caching for providers that cache nothing without a breakpoint.

The caching config is applied only around the agent's own LLM call, so tools that share the
same llm instance keep making uncached one-shot calls.
"""

import threading
import time
import uuid

import pytest
from litellm import ModelResponse

from dynamiq import connections, prompts
from dynamiq.nodes.agents import Agent
from dynamiq.nodes.agents.agent import default_cache_control
from dynamiq.nodes.llms import Anthropic, AnthropicCacheControl, Bedrock, OpenAI


def _anthropic(**kwargs):
    return Anthropic(
        name="TestLLM",
        model="claude-sonnet-4-5",
        connection=connections.Anthropic(id=str(uuid.uuid4()), api_key="fake-key"),
        prompt=prompts.Prompt(messages=[prompts.Message(role="user", content="{{input}}")]),
        **kwargs,
    )


def _openai():
    return OpenAI(
        name="TestLLM",
        model="gpt-4o-mini",
        connection=connections.OpenAI(id=str(uuid.uuid4()), api_key="fake-key"),
        prompt=prompts.Prompt(messages=[prompts.Message(role="user", content="{{input}}")]),
    )


def _run(llm):
    agent = Agent(name="a", llm=llm, tools=[], max_loops=2)
    agent.run(input_data={"input": "hi"})
    return agent


def _points(mock_completion):
    return mock_completion.call_args.kwargs.get("cache_control_injection_points")


# -- default_cache_control: which nodes get a config, and when the caller wins --------------


def test_unset_anthropic_gets_a_rolling_breakpoint_at_minus_one():
    control = default_cache_control(_anthropic())
    assert isinstance(control, AnthropicCacheControl)
    assert control.cache_injection_point_index == -1


def test_explicit_none_is_an_opt_out():
    assert default_cache_control(_anthropic(cache_control=None)) is None


def test_caller_supplied_config_is_left_alone():
    assert default_cache_control(_anthropic(cache_control=AnthropicCacheControl(ttl="1h"))) is None


def test_node_without_the_field_is_ignored():
    assert default_cache_control(_openai()) is None


def _bedrock(model):
    return Bedrock(
        name="TestLLM",
        model=model,
        connection=connections.AWS(
            id=str(uuid.uuid4()), access_key_id="k", secret_access_key="s", region="us-east-1"
        ),
        prompt=prompts.Prompt(messages=[prompts.Message(role="user", content="{{input}}")]),
    )


@pytest.mark.parametrize(
    "model",
    [
        "us.meta.llama3-3-70b-instruct-v1:0",  # verified live: rejects a cachePoint outright
        "mistral.mistral-large-2407-v1:0",
        "totally.made-up-model-v9",  # unknown to the registry -- must stay off, not guess
    ],
)
def test_models_without_caching_support_are_left_alone(model):
    """A cachePoint on an unsupporting Bedrock model fails the request rather than being
    ignored, so enabling this by default would break working workflows."""
    assert default_cache_control(_bedrock(model)) is None


@pytest.mark.parametrize(
    "model",
    ["global.anthropic.claude-sonnet-5", "us.anthropic.claude-sonnet-4-6", "amazon.nova-micro-v1:0"],
)
def test_models_with_caching_support_are_enabled(model):
    assert default_cache_control(_bedrock(model)) is not None


# -- the request the agent actually sends ---------------------------------------------------


def test_agent_run_sends_both_breakpoints(mock_llm_executor):
    _run(_anthropic())

    points = _points(mock_llm_executor)
    assert points == [
        {"location": "message", "role": "system", "control": {"type": "ephemeral", "ttl": "5m"}},
        {"location": "message", "index": -1, "control": {"type": "ephemeral", "ttl": "5m"}},
    ]


def test_head_pin_comes_first_so_it_survives_a_tight_block_budget(mock_llm_executor):
    _run(_anthropic())

    assert _points(mock_llm_executor)[0]["role"] == "system"


def test_every_loop_is_cached_not_just_the_first(mock_llm_executor):
    """Assigning cache_control marks the field caller-set; the decision must not be re-derived
    from the llm after the first loop, or caching silently stops after one call."""
    _run(_anthropic())

    react_calls = [call for call in mock_llm_executor.call_args_list if "tools" in call.kwargs]
    assert len(react_calls) > 1, "need a multi-loop run to prove this"
    assert all(call.kwargs.get("cache_control_injection_points") for call in react_calls)


def test_cache_system_false_leaves_only_the_rolling_point(mock_llm_executor):
    _run(_anthropic(cache_control=AnthropicCacheControl(cache_system=False)))

    points = _points(mock_llm_executor)
    assert len(points) == 1
    assert points[0]["index"] == -1


def test_caller_ttl_survives(mock_llm_executor):
    _run(_anthropic(cache_control=AnthropicCacheControl(ttl="1h")))

    assert all(point["control"]["ttl"] == "1h" for point in _points(mock_llm_executor))


def test_explicit_none_sends_no_breakpoints(mock_llm_executor):
    _run(_anthropic(cache_control=None))

    assert _points(mock_llm_executor) is None


def test_openai_agent_sends_no_breakpoints(mock_llm_executor):
    _run(_openai())

    assert _points(mock_llm_executor) is None


# -- isolation: the config must not outlive the call ----------------------------------------


def test_the_llm_node_is_never_mutated(mocker):
    """The config is delivered per call. Nothing is written to the shared node -- not even
    for the duration of the call -- so tools and parallel subagents using the same instance
    are unaffected."""
    llm = _anthropic()
    during = {}

    def capture(self, **kwargs):
        during["cache_control"] = self.cache_control
        during["points"] = kwargs.get("cache_control_injection_points")
        response = ModelResponse()
        response["choices"][0]["message"]["content"] = "mocked_response"
        return response

    mocker.patch("dynamiq.nodes.llms.base.BaseLLM._completion", new=capture)
    _run(llm)

    assert during["points"], "the caching config must still reach the request"
    assert during["cache_control"] is None, "the node was mutated during the call"
    assert llm.cache_control is None


def test_parallel_subagents_sharing_one_llm(mocker):
    """Two agents on one llm instance, running concurrently -- the shape that made the old
    save/restore approach leak, where whichever thread finished last won."""
    llm = _anthropic()
    seen = []

    def slow_call(self, **kwargs):
        seen.append((self.cache_control, kwargs.get("cache_control_injection_points")))
        time.sleep(0.02)
        response = ModelResponse()
        response["choices"][0]["message"]["content"] = "mocked_response"
        return response

    mocker.patch("dynamiq.nodes.llms.base.BaseLLM._completion", new=slow_call)
    threads = [threading.Thread(target=_run, args=(llm,)) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert seen, "both agents should have called the llm"
    assert all(points for _, points in seen), "every call carries its own caching config"
    assert all(control is None for control, _ in seen), "no call mutated the shared node"
    assert llm.cache_control is None


def test_bare_node_outside_an_agent_still_sends_nothing():
    assert "cache_control_injection_points" not in _anthropic().update_completion_params({})


@pytest.mark.parametrize("messages", [[prompts.Message(role="user", content="no system message")]])
def test_head_pin_is_safe_without_a_system_message(messages):
    """The role-targeted point resolves to nothing rather than erroring."""
    llm = _anthropic(cache_control=AnthropicCacheControl())
    params = llm.update_completion_params({"messages": messages})

    assert params["cache_control_injection_points"][0]["role"] == "system"
