"""Agents turn on prompt caching for providers that cache nothing without a breakpoint.

The config is written onto the agent's LLM node once, at construction, so every caller
sharing that instance inherits it -- including the ContextManagerTool. Summarization
therefore caches too, which is cost-only and is the subject of a follow-up PR.
"""

import os
import tempfile
import threading
import time
import uuid

import pytest
from litellm import ModelResponse

from dynamiq import connections, prompts
from dynamiq.nodes.agents import Agent
from dynamiq.nodes.agents.agent import default_cache_control
from dynamiq.nodes.agents.utils import SummarizationConfig
from dynamiq.nodes.llms import Anthropic, AnthropicCacheControl, Bedrock, BedrockCacheControl, OpenAI
from dynamiq.nodes.llms.base import FallbackConfig
from dynamiq.nodes.tools.context_manager import ContextManagerTool
from dynamiq.runnables import RunnableConfig


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


def _agent(llm, **kwargs):
    return Agent(name="a", llm=llm, tools=[], max_loops=2, **kwargs)


def _run(llm):
    agent = _agent(llm)
    agent.run(input_data={"input": "hi"})
    return agent


def _points(mock_completion):
    return mock_completion.call_args.kwargs.get("cache_control_injection_points")


# -- default_cache_control: which nodes get a config, and when the caller wins --------------


def test_unset_anthropic_gets_a_rolling_breakpoint_at_minus_one():
    control = default_cache_control(_anthropic())
    assert isinstance(control, AnthropicCacheControl)
    assert control.cache_injection_point_index == -1


def test_explicit_false_is_an_opt_out():
    assert default_cache_control(_anthropic(cache_control=False)) is None


def test_explicit_none_still_gets_the_default():
    """``None`` is "nothing chosen", not "off" -- it is what a round trip leaves behind."""
    assert default_cache_control(_anthropic(cache_control=None)) is not None


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
        "amazon.nova-micro-v1:0",  # caches, but 400s on a cachePoint in a tool-call message
        "us.openai.gpt-6-astra",  # LiteLLM flags it cacheable, but it caches implicitly
        "totally.made-up-model-v9",  # unknown to the registry -- must stay off, not guess
    ],
)
def test_models_without_caching_support_are_left_alone(model):
    """A cachePoint on an unsupporting Bedrock model fails the request rather than being
    ignored, so enabling this by default would break working workflows."""
    assert default_cache_control(_bedrock(model)) is None


@pytest.mark.parametrize("model", ["global.anthropic.claude-sonnet-5", "us.anthropic.claude-sonnet-4-6"])
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


def test_explicit_false_sends_no_breakpoints(mock_llm_executor):
    _run(_anthropic(cache_control=False))

    assert _points(mock_llm_executor) is None


def test_openai_agent_sends_no_breakpoints(mock_llm_executor):
    _run(_openai())

    assert _points(mock_llm_executor) is None


# -- isolation: the config must not outlive the call ----------------------------------------


def test_the_config_is_written_once_and_never_restored(mocker):
    """No save/restore pair around the call, so no window exists in which a concurrent caller
    sees the node without its config."""
    llm = _anthropic()
    agent = _agent(llm)
    written = llm.cache_control
    assert written is not None, "construction must enable caching"

    during = {}

    def capture(self, **kwargs):
        during.setdefault("controls", []).append(self.cache_control)
        during["points"] = kwargs.get("cache_control_injection_points")
        response = ModelResponse()
        response["choices"][0]["message"]["content"] = "mocked_response"
        return response

    mocker.patch("dynamiq.nodes.llms.base.BaseLLM._completion", new=capture)
    agent.run(input_data={"input": "hi"})

    assert during["points"], "the caching config must still reach the request"
    assert all(c is written for c in during["controls"]), "the node was swapped mid-call"
    assert llm.cache_control is written, "the node was restored after the call"


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
    assert all(points for _, points in seen), "every call carries the caching config"
    assert all(control is not None for control, _ in seen), "a call saw the node with no config"
    # By value, not identity: a construction race may write two equal instances, harmlessly.
    assert len({control.model_dump_json() for control, _ in seen}) == 1
    assert llm.cache_control is not None


# -- what construction writes onto the node -------------------------------------------------


def test_construction_writes_the_config_onto_the_node():
    llm = _anthropic()
    _agent(llm)

    assert isinstance(llm.cache_control, AnthropicCacheControl)


def test_opt_out_is_left_alone():
    llm = _anthropic(cache_control=False)
    _agent(llm)

    assert llm.cache_control is False


def test_a_node_without_the_field_gains_no_extra():
    """`BaseLLM` spreads extra fields into the request, so a `None` here would be sent."""
    llm = _openai()
    _agent(llm)

    assert "cache_control" not in (llm.__pydantic_extra__ or {})


def test_a_second_agent_does_not_rewrite_the_config():
    """Sub-agents and factory-rebuilt agents share one node; the decision is made once."""
    llm = _anthropic()
    _agent(llm)
    written = llm.cache_control
    _agent(llm)

    assert llm.cache_control is written


# -- the fallback chain ---------------------------------------------------------------------


def test_fallback_llm_also_gets_the_default():
    """A separate node, asked separately -- and the call that most needs the cache."""
    fallback = _bedrock("global.anthropic.claude-sonnet-5")
    llm = _anthropic(fallback=FallbackConfig(llm=fallback))
    _agent(llm)

    assert llm.cache_control is not None
    # Its own provider's config class, never the primary's.
    assert isinstance(fallback.cache_control, BedrockCacheControl)


def test_unsupported_fallback_model_is_left_alone():
    """The model check runs against the node that will send the request, so a Claude -> Nova
    fallback gets nothing rather than a cachePoint Nova rejects."""
    fallback = _bedrock("us.amazon.nova-lite-v1:0")
    llm = _anthropic(fallback=FallbackConfig(llm=fallback))
    _agent(llm)

    assert llm.cache_control is not None
    assert fallback.cache_control is None


def test_a_cyclic_fallback_chain_terminates():
    """Nothing forbids a node being its own fallback; the walk must not spin."""
    llm = _anthropic()
    llm.fallback = FallbackConfig(llm=llm)
    _agent(llm)

    assert llm.cache_control is not None


# -- accepted: callers sharing the node inherit it ------------------------------------------


def test_the_summarizer_shares_the_cached_llm(mock_llm_executor):
    """The ContextManagerTool is built with `llm=self.llm`, so summarization now sends
    breakpoints too. Cost-only (a one-shot prompt writes a cache nothing reads back), and
    deliberately left for the follow-up summarization PR to address."""
    llm = _anthropic()
    agent = _agent(llm, summarization_config=SummarizationConfig(enabled=True))
    summarizer = next(t for t in agent.tools if isinstance(t, ContextManagerTool))
    assert summarizer.llm is llm, "the tool must share the agent's node for this to matter"

    summarizer._call_llm_for_summary([prompts.Message(role="user", content="summarise this")], config=RunnableConfig())

    assert _points(mock_llm_executor)


def test_bare_node_outside_an_agent_still_sends_nothing():
    assert "cache_control_injection_points" not in _anthropic().update_completion_params({})


def test_opted_out_node_outside_an_agent_still_sends_nothing():
    assert "cache_control_injection_points" not in _anthropic(cache_control=False).update_completion_params({})


# -- the decision has to survive serialization ----------------------------------------------


def _roundtrip(llm) -> Agent:
    """Dump an agent to YAML and load it back, the way a saved workflow is restored."""
    from dynamiq import Workflow
    from dynamiq.flows import Flow
    from dynamiq.serializers.loaders.yaml import WorkflowYAMLLoader

    agent = Agent(name="a", id="agent_1", llm=llm, tools=[], is_postponed_component_init=True)
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "wf.yaml")
        Workflow(id="wf_1", flow=Flow(id="flow_1", nodes=[agent])).to_yaml_file(path)
        data = WorkflowYAMLLoader.load(file_path=path, connection_manager=None, init_components=False)
        return Workflow.from_yaml_file_data(file_data=data, wf_id="wf_1").flow.nodes[0]


def _caches(agent: Agent) -> bool:
    """Whether this agent's calls carry breakpoints. A config is truthy, `False`/`None` are not."""
    return bool(agent.llm.cache_control)


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({}, True),  # untouched -> the default, and it must survive the round trip
        ({"cache_control": False}, False),  # opt-out, and it must survive too
        ({"cache_control": AnthropicCacheControl(ttl="1h")}, True),  # caller's own config
    ],
    ids=["unset", "opted-out", "custom-config"],
)
def test_caching_intent_survives_a_yaml_round_trip(kwargs, expected):
    """A dumped node is rebuilt field-by-field, so `model_fields_set` marks all of them
    as set -- keying the default off that read a round trip as an opt-out."""
    before = Agent(name="a", llm=_anthropic(**kwargs), tools=[], is_postponed_component_init=True)
    assert _caches(before) is expected, "in-process construction disagrees with the fixture"

    # A *fresh* node: constructing the agent above already wrote the default onto its llm, so
    # reusing it would dump a config in every case and prove nothing about the unset one.
    assert _caches(_roundtrip(_anthropic(**kwargs))) is expected


def test_custom_ttl_survives_a_yaml_round_trip():
    """Not just on/off: the caller's settings have to come back intact."""
    restored = _roundtrip(_anthropic(cache_control=AnthropicCacheControl(ttl="1h", cache_system=False)))

    assert restored.llm.cache_control.ttl == "1h"
    assert restored.llm.cache_control.cache_system is False


@pytest.mark.parametrize("messages", [[prompts.Message(role="user", content="no system message")]])
def test_head_pin_is_safe_without_a_system_message(messages):
    """The role-targeted point resolves to nothing rather than erroring."""
    llm = _anthropic(cache_control=AnthropicCacheControl())
    params = llm.update_completion_params({"messages": messages})

    assert params["cache_control_injection_points"][0]["role"] == "system"
