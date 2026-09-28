"""Bedrock prompt caching: the injection points Dynamiq hands LiteLLM.

Caching failures here are silent -- the request still succeeds, the bill is just
higher -- so these assert the exact payload rather than "something was set".
"""

import pytest

from dynamiq.connections import AWS as AWSConnection
from dynamiq.connections import Anthropic as AnthropicConnection
from dynamiq.nodes.llms import Anthropic, AnthropicCacheControl, Bedrock, BedrockCacheControl

MODEL = "bedrock/us.anthropic.claude-sonnet-4-6"
TOOLS = [{"type": "function", "function": {"name": "search", "parameters": {}}}]


def _llm(**kwargs) -> Bedrock:
    return Bedrock(
        connection=AWSConnection(access_key_id="k", secret_access_key="s", region="us-east-1"),
        model=MODEL,
        is_postponed_component_init=True,
        **kwargs,
    )


def _points(llm: Bedrock, **params) -> list[dict]:
    return llm.update_completion_params({"model": MODEL, **params}).get("cache_control_injection_points", [])


class TestDisabledByDefault:
    def test_no_cache_key_is_added(self):
        """Existing users must see byte-identical params; the key is absent, not empty."""
        params = _llm().update_completion_params({"model": MODEL, "tools": TOOLS})

        assert "cache_control_injection_points" not in params

    def test_explicit_none_is_also_off(self):
        assert _points(_llm(cache_control=None), tools=TOOLS) == []


class TestInjectionPoints:
    def test_system_and_rolling_message_point(self):
        """Two breakpoints: the system prompt (tools render before it, so they are
        covered too), then the rolling message tail."""
        assert _points(_llm(cache_control=BedrockCacheControl()), tools=TOOLS) == [
            {"location": "message", "role": "system", "control": {"type": "ephemeral", "ttl": "5m"}},
            {"location": "message", "index": -1, "control": {"type": "ephemeral", "ttl": "5m"}},
        ]

    def test_system_point_does_not_depend_on_tools(self):
        """Pins the head with or without tools; resolves to nothing if there is no
        system message, rather than erroring."""
        assert _points(_llm(cache_control=BedrockCacheControl())) == [
            {"location": "message", "role": "system", "control": {"type": "ephemeral", "ttl": "5m"}},
            {"location": "message", "index": -1, "control": {"type": "ephemeral", "ttl": "5m"}},
        ]

    def test_system_point_can_be_disabled(self):
        points = _points(_llm(cache_control=BedrockCacheControl(cache_system=False)), tools=TOOLS)

        assert points == [{"location": "message", "index": -1, "control": {"type": "ephemeral", "ttl": "5m"}}]

    def test_cache_tools_is_accepted_as_a_deprecated_alias(self):
        """`cache_tools` shipped in v0.64.0; existing configs must keep working."""
        assert BedrockCacheControl(cache_tools=False).cache_system is False
        assert _points(_llm(cache_control=BedrockCacheControl(cache_tools=False)), tools=TOOLS) == [
            {"location": "message", "index": -1, "control": {"type": "ephemeral", "ttl": "5m"}},
        ]

    @pytest.mark.parametrize("ttl", ["5m", "1h"])
    def test_ttl_is_forwarded(self, ttl):
        points = _points(_llm(cache_control=BedrockCacheControl(ttl=ttl)), tools=TOOLS)

        assert all(p["control"]["ttl"] == ttl for p in points)

    def test_ttl_none_is_omitted_not_null(self):
        """LiteLLM reads `ttl` by presence; a null would be forwarded as a value."""
        points = _points(_llm(cache_control=BedrockCacheControl(ttl=None)), tools=TOOLS)

        assert all("ttl" not in p["control"] for p in points)

    def test_index_is_configurable(self):
        points = _points(_llm(cache_control=BedrockCacheControl(cache_injection_point_index=-2)), tools=TOOLS)

        assert points[-1]["index"] == -2

    def test_control_carries_no_dynamiq_only_fields(self):
        """`cache_system`/`cache_injection_point_index` steer us, not the provider."""
        points = _points(_llm(cache_control=BedrockCacheControl()), tools=TOOLS)

        for point in points:
            assert set(point["control"]) <= {"type", "ttl"}

    def test_points_do_not_share_a_control_dict(self):
        """LiteLLM assigns the control into the message by reference, so two points
        sharing one dict would have the same object land on two messages."""
        points = _points(_llm(cache_control=BedrockCacheControl()), tools=TOOLS)

        assert points[0]["control"] is not points[1]["control"]


class TestEveryRouteIsSafe:
    """A leftover non-message point is spread into the Invoke body and Bedrock 400s.
    Message points are consumed by the hook on every route, so emitting only those is safe."""

    @pytest.mark.parametrize(
        "model",
        [
            "bedrock/us.anthropic.claude-sonnet-4-6",  # converse
            "bedrock/eu.anthropic.claude-some-future-model",  # unknown -> invoke fallback
            "bedrock/invoke/us.anthropic.claude-sonnet-4-6",  # explicit invoke
        ],
    )
    def test_only_message_points_are_emitted(self, model):
        llm = Bedrock(
            connection=AWSConnection(access_key_id="k", secret_access_key="s", region="us-east-1"),
            model=model,
            cache_control=BedrockCacheControl(),
            is_postponed_component_init=True,
        )
        points = llm.update_completion_params({"model": model, "tools": TOOLS})[
            "cache_control_injection_points"
        ]

        assert [p["location"] for p in points] == ["message", "message"]


class TestTheTwoProviderConfigsStayInSync:
    """Two duplicated `_apply_cache_control` implementations, one user-facing contract.

    An agent now resolves each node in a fallback chain separately, so a config never crosses
    providers -- but the two classes are still documented and configured identically, and a
    field or payload that drifts between them is a silent behaviour difference between
    `Anthropic` and `Bedrock`.
    """

    def test_the_two_configs_expose_the_same_fields(self):
        assert set(AnthropicCacheControl.model_fields) == set(BedrockCacheControl.model_fields)

    @pytest.mark.parametrize("config", [AnthropicCacheControl, BedrockCacheControl])
    def test_bedrock_accepts_either_providers_config(self, config):
        assert _llm()._apply_cache_control({}, config())["cache_control_injection_points"] == [
            {"location": "message", "role": "system", "control": {"type": "ephemeral", "ttl": "5m"}},
            {"location": "message", "index": -1, "control": {"type": "ephemeral", "ttl": "5m"}},
        ]

    @pytest.mark.parametrize("config", [AnthropicCacheControl, BedrockCacheControl])
    def test_anthropic_accepts_either_providers_config(self, config):
        llm = Anthropic(
            connection=AnthropicConnection(api_key="k"),
            model="claude-sonnet-4-5",
            is_postponed_component_init=True,
        )

        assert llm._apply_cache_control({}, config())["cache_control_injection_points"] == [
            {"location": "message", "role": "system", "control": {"type": "ephemeral", "ttl": "5m"}},
            {"location": "message", "index": -1, "control": {"type": "ephemeral", "ttl": "5m"}},
        ]


class TestBreakpointBudget:
    def test_stays_within_the_four_block_limit(self):
        """Bedrock rejects >4 cache_control blocks; we contribute at most 2."""
        assert len(_points(_llm(cache_control=BedrockCacheControl()), tools=TOOLS)) <= 2

    def test_appends_to_caller_supplied_points(self):
        """A caller's own breakpoints are kept and counted, never overwritten."""
        existing = [{"location": "message", "index": 0, "control": {"type": "ephemeral"}}]
        params = _llm(cache_control=BedrockCacheControl()).update_completion_params(
            {"model": MODEL, "tools": TOOLS, "cache_control_injection_points": existing}
        )

        assert params["cache_control_injection_points"][0] == existing[0]
        assert len(params["cache_control_injection_points"]) == 3
