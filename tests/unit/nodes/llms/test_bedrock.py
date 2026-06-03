"""Unit tests for the Bedrock node's litellm #22637 parallel-tool-calls workaround.

litellm builds an `additionalModelRequestFields.tool_choice` for Claude 4.5+ on Bedrock
that omits Anthropic's required `type` discriminator, so Bedrock rejects the request with
"tool_choice.type: Field required". The Bedrock node idempotently wraps litellm's
`AmazonConverseConfig.map_openai_params` to inject the missing `type` while preserving the
user's `parallel_tool_calls` value (True -> parallel allowed, False -> sequential).
"""

import pytest

from dynamiq.connections import AWS as AWSConnection
from dynamiq.nodes.llms.bedrock import (
    Bedrock,
    _ensure_parallel_tool_choice_type,
    _install_litellm_bedrock_parallel_tool_patch,
)

_MODEL_3_5 = "bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0"
_MODEL_4_5 = "eu.anthropic.claude-sonnet-4-5-20250929-v1:0"


class TestEnsureParallelToolChoiceType:
    """The pure repair helper."""

    def test_adds_type_when_missing_and_parallel_enabled(self):
        params = {"_parallel_tool_use_config": {"tool_choice": {"disable_parallel_tool_use": False}}}
        _ensure_parallel_tool_choice_type(params)
        assert params["_parallel_tool_use_config"]["tool_choice"] == {
            "type": "auto",
            "disable_parallel_tool_use": False,
        }

    def test_adds_type_but_preserves_disabled_flag(self):
        # parallel_tool_calls=False -> disable_parallel_tool_use=True must be kept (sequential).
        params = {"_parallel_tool_use_config": {"tool_choice": {"disable_parallel_tool_use": True}}}
        _ensure_parallel_tool_choice_type(params)
        tool_choice = params["_parallel_tool_use_config"]["tool_choice"]
        assert tool_choice["type"] == "auto"
        assert tool_choice["disable_parallel_tool_use"] is True

    def test_does_not_override_existing_type(self):
        params = {"_parallel_tool_use_config": {"tool_choice": {"type": "any", "disable_parallel_tool_use": False}}}
        _ensure_parallel_tool_choice_type(params)
        assert params["_parallel_tool_use_config"]["tool_choice"]["type"] == "any"

    @pytest.mark.parametrize(
        "params",
        [
            {},
            {"_parallel_tool_use_config": None},
            {"_parallel_tool_use_config": {}},
            {"_parallel_tool_use_config": {"tool_choice": None}},
            {"_parallel_tool_use_config": {"tool_choice": "auto"}},
        ],
    )
    def test_tolerates_missing_or_unexpected_shapes(self, params):
        before = repr(params)
        _ensure_parallel_tool_choice_type(params)  # must not raise
        assert repr(params) == before  # and must not mutate anything


class TestBedrockPatchWiring:
    """Constructing the node installs the patch, once."""

    def test_constructing_bedrock_installs_patch(self):
        Bedrock(model=_MODEL_3_5, connection=AWSConnection())
        from litellm.llms.bedrock.chat.converse_transformation import AmazonConverseConfig

        assert getattr(AmazonConverseConfig, "_dynamiq_parallel_tc_patched", False) is True

    def test_patch_is_idempotent(self):
        from litellm.llms.bedrock.chat.converse_transformation import AmazonConverseConfig

        _install_litellm_bedrock_parallel_tool_patch()
        wrapped = AmazonConverseConfig.map_openai_params
        _install_litellm_bedrock_parallel_tool_patch()
        Bedrock(model=_MODEL_3_5, connection=AWSConnection())
        # Not re-wrapped: same function object after repeated installs / constructions.
        assert AmazonConverseConfig.map_openai_params is wrapped


class TestPatchedMapOpenAIParams:
    """End-to-end: litellm's transform now emits a valid, value-preserving tool_choice."""

    @pytest.mark.parametrize(("parallel_tool_calls", "expected_disable"), [(True, False), (False, True)])
    def test_injects_type_and_honors_value(self, parallel_tool_calls, expected_disable):
        _install_litellm_bedrock_parallel_tool_patch()
        from litellm.llms.bedrock.chat.converse_transformation import AmazonConverseConfig

        optional_params = AmazonConverseConfig().map_openai_params(
            non_default_params={"parallel_tool_calls": parallel_tool_calls},
            optional_params={},
            model=_MODEL_4_5,
            drop_params=False,
        )
        tool_choice = optional_params["_parallel_tool_use_config"]["tool_choice"]
        assert tool_choice["type"] == "auto"
        assert tool_choice["disable_parallel_tool_use"] is expected_disable
