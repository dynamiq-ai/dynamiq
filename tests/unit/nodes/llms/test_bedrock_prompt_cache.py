"""Bedrock prompt caching: the injection points Dynamiq hands LiteLLM.

Caching failures here are silent -- the request still succeeds, the bill is just
higher -- so these assert the exact payload rather than "something was set".
"""

import pytest

from dynamiq.connections import AWS as AWSConnection
from dynamiq.nodes.llms.bedrock import Bedrock, BedrockCacheControl

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
    def test_tools_and_rolling_message_point(self):
        """Two breakpoints: the static tool schemas, then the rolling message tail."""
        assert _points(_llm(cache_control=BedrockCacheControl()), tools=TOOLS) == [
            {"location": "tool_config", "control": {"type": "ephemeral", "ttl": "5m"}},
            {"location": "message", "index": -1, "control": {"type": "ephemeral", "ttl": "5m"}},
        ]

    def test_tool_point_skipped_without_tools(self):
        """A tool_config point spends one of Bedrock's four breakpoints for nothing."""
        assert _points(_llm(cache_control=BedrockCacheControl())) == [
            {"location": "message", "index": -1, "control": {"type": "ephemeral", "ttl": "5m"}},
        ]

    def test_tool_point_can_be_disabled(self):
        points = _points(_llm(cache_control=BedrockCacheControl(cache_tools=False)), tools=TOOLS)

        assert [p["location"] for p in points] == ["message"]

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
        """`cache_tools`/`cache_injection_point_index` steer us, not the provider."""
        points = _points(_llm(cache_control=BedrockCacheControl()), tools=TOOLS)

        for point in points:
            assert set(point["control"]) <= {"type", "ttl"}


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
