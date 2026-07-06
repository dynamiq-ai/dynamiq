from unittest.mock import MagicMock, patch

import pytest

from dynamiq.connections import AWS as AWSConnection
from dynamiq.connections import Anthropic as AnthropicConnection
from dynamiq.connections import HttpApiKey
from dynamiq.nodes.llms.anthropic import Anthropic
from dynamiq.nodes.llms.base import SAMPLING_PARAMS
from dynamiq.nodes.llms.bedrock import Bedrock
from dynamiq.nodes.llms.custom_llm import CustomLLM
from dynamiq.prompts import Prompt
from dynamiq.runnables import RunnableConfig


def _mock_response(content="ok"):
    """Minimal litellm ModelResponse stand-in for _handle_completion_response."""
    choice = MagicMock()
    choice.message.content = content
    choice.message.tool_calls = None
    response = MagicMock()
    response.choices = [choice]
    response.model_extra = {}
    usage = MagicMock()
    usage.prompt_tokens = usage.completion_tokens = usage.total_tokens = 0
    usage.prompt_tokens_details = None
    usage.cache_read_input_tokens = usage.cache_creation_input_tokens = None
    response.usage = usage
    return response


@pytest.fixture
def anthropic_unsupported():
    return Anthropic(name="a", model="claude-opus-4-8", connection=AnthropicConnection(api_key="x"))


@pytest.fixture
def anthropic_supported():
    return Anthropic(name="a", model="claude-opus-4-6", connection=AnthropicConnection(api_key="x"))


class TestDetection:
    @pytest.mark.parametrize(
        "model",
        [
            # Known rejecting generations.
            "claude-opus-4-7",
            "claude-opus-4-8",
            "claude-sonnet-5",
            "claude-fable-5",
            "claude-mythos-5",
            "claude-mythos-preview",
            # Future versioned releases.
            "claude-opus-4-9",
            "claude-opus-4-10",
            "claude-opus-5",
            "claude-sonnet-5-1",
            "claude-sonnet-6",
        ],
    )
    def test_unsupported_models_detected(self, model):
        llm = Anthropic(name="a", model=model, connection=AnthropicConnection(api_key="x"))
        assert llm._model_rejects_sampling_params() is True

    @pytest.mark.parametrize(
        "template",
        ["bedrock/eu.anthropic.{}-20251101-v1:0", "openrouter/anthropic/{}"],
    )
    def test_detection_is_provider_prefix_independent(self, template):
        # A rejecting model must be detected regardless of the provider prefix wrapped around it,
        # including a trailing date suffix that must not be parsed as the minor version.
        model = template.format("claude-sonnet-5")
        llm = Anthropic(name="a", model=model, connection=AnthropicConnection(api_key="x"))
        assert llm._model_rejects_sampling_params() is True

    @pytest.mark.parametrize(
        "model",
        [
            # Older Anthropic generations that still accept sampling params.
            "claude-opus-4-5",
            "claude-opus-4-6",
            "claude-sonnet-4-6",
            "claude-haiku-4-5",
            # Dated full id for Opus 4.0 — the date must not be read as a minor version above the cutoff.
            "claude-opus-4-20250514",
            # No released Haiku rejects yet; a future one is left to the runtime backstop.
            "claude-haiku-5",
            # Retired pre-4 naming and non-Anthropic models must not false-positive.
            "claude-3-5-sonnet-20241022",
            "gpt-4o",
        ],
    )
    def test_supported_models_not_detected(self, model):
        llm = Anthropic(name="a", model=model, connection=AnthropicConnection(api_key="x"))
        assert llm._model_rejects_sampling_params() is False


class TestDeterministicStrip:
    def test_sampling_params_stripped_for_unsupported(self, anthropic_unsupported):
        params = anthropic_unsupported.update_completion_params(
            {"model": "anthropic/claude-opus-4-8", "temperature": 0.5, "top_p": 0.9, "top_k": 40, "max_tokens": 100}
        )
        for param in SAMPLING_PARAMS:
            assert param not in params
        # Non-sampling params are untouched.
        assert params["max_tokens"] == 100

    def test_sampling_params_kept_for_supported(self, anthropic_supported):
        params = anthropic_supported.update_completion_params(
            {"model": "anthropic/claude-opus-4-6", "temperature": 0.5, "top_p": 0.9}
        )
        assert params["temperature"] == 0.5
        assert params["top_p"] == 0.9

    def test_strip_is_provider_independent_bedrock(self):
        llm = Bedrock(
            name="b",
            model="bedrock/eu.anthropic.claude-opus-4-8-20251101-v1:0",
            connection=AWSConnection(access_key_id="x", secret_access_key="y", region_name="eu-west-1"),
        )
        params = llm.update_completion_params({"model": llm.model, "temperature": 0.7})
        assert "temperature" not in params

    def test_strip_is_provider_independent_custom(self):
        llm = CustomLLM(
            name="c",
            model="anthropic/claude-opus-4-8",
            connection=HttpApiKey(url="https://example.com", api_key="x"),
        )
        params = llm.update_completion_params({"model": llm.model, "temperature": 0.7})
        assert "temperature" not in params


class TestReactiveBackstop:
    def test_recovers_on_unsupported_error(self, anthropic_supported):
        # Model not in the marker list, so the deterministic path leaves the param in.
        anthropic_supported.temperature = 0.5
        common = {"model": "anthropic/claude-opus-4-6", "temperature": 0.5, "max_tokens": 100}
        exc = Exception("litellm.BadRequestError: temperature: Extra inputs are not permitted")
        recovered = anthropic_supported._recover_completion_params(exc, common)
        assert recovered is not None
        assert "temperature" not in recovered
        assert recovered["max_tokens"] == 100
        # The recovery hook is pure: it must not mutate the instance, so a failed retry
        # leaves the reusable node untouched.
        assert anthropic_supported.temperature == 0.5

    def test_persist_recovery_clears_dropped_params(self, anthropic_supported):
        # Persistence runs only after a successful retry, mirroring execute().
        anthropic_supported.temperature = 0.5
        common = {"model": "anthropic/claude-opus-4-6", "temperature": 0.5, "max_tokens": 100}
        recovered = {"model": "anthropic/claude-opus-4-6", "max_tokens": 100}
        anthropic_supported._persist_completion_recovery(common, recovered)
        # Persisted so the next call skips the failing first attempt.
        assert anthropic_supported.temperature is None

    def test_persist_recovery_noop_when_param_kept(self, anthropic_supported):
        # A param still present in the recovered dict must not be cleared.
        anthropic_supported.temperature = 0.5
        common = {"model": "anthropic/claude-opus-4-6", "temperature": 0.5}
        anthropic_supported._persist_completion_recovery(common, dict(common))
        assert anthropic_supported.temperature == 0.5

    def test_no_recovery_on_out_of_range_error(self, anthropic_supported):
        # A genuine validation error must surface, not be silently stripped.
        anthropic_supported.temperature = 5.0
        common = {"model": "anthropic/claude-opus-4-6", "temperature": 5.0}
        exc = Exception("temperature: Input should be less than or equal to 1.0")
        assert anthropic_supported._recover_completion_params(exc, common) is None
        assert anthropic_supported.temperature == 5.0

    def test_no_recovery_when_no_sampling_params_present(self, anthropic_supported):
        exc = Exception("some unrelated unsupported field error")
        assert anthropic_supported._recover_completion_params(exc, {"model": "x", "max_tokens": 10}) is None

    def test_no_recovery_on_unexpected_value_validation_error(self, anthropic_supported):
        # "unexpected value" is a validation error, not an unsupported-param signal.
        anthropic_supported.temperature = 5.0
        common = {"model": "anthropic/claude-opus-4-6", "temperature": 5.0}
        exc = Exception("temperature: unexpected value; permitted range is 0..1")
        assert anthropic_supported._recover_completion_params(exc, common) is None

    def test_recovers_on_unexpected_keyword_error(self, anthropic_supported):
        # A Python-level "unexpected keyword argument" still counts as unsupported.
        anthropic_supported.temperature = 0.5
        common = {"model": "anthropic/claude-opus-4-6", "temperature": 0.5}
        exc = Exception("completion() got an unexpected keyword argument 'temperature'")
        recovered = anthropic_supported._recover_completion_params(exc, common)
        assert recovered is not None
        assert "temperature" not in recovered


class TestBedrockBackstop:
    @pytest.fixture
    def bedrock(self):
        return Bedrock(
            name="b",
            model="bedrock/eu.anthropic.claude-some-future-model",
            connection=AWSConnection(access_key_id="x", secret_access_key="y", region_name="eu-west-1"),
        )

    def test_bedrock_falls_through_to_sampling_backstop(self, bedrock):
        # An unlisted Bedrock model must still get the base sampling recovery,
        # not be dropped because Bedrock overrides the hook.
        bedrock.temperature = 0.5
        common = {"model": bedrock.model, "temperature": 0.5, "max_tokens": 100}
        exc = Exception("litellm.BadRequestError: temperature: Extra inputs are not permitted")
        recovered = bedrock._recover_completion_params(exc, common)
        assert recovered is not None
        assert "temperature" not in recovered
        assert recovered["max_tokens"] == 100

    def test_bedrock_stop_recovery_is_pure(self, bedrock):
        # The stop recovery must not mutate the instance before the retry succeeds.
        bedrock.stop = ["STOP"]
        common = {"model": bedrock.model, "stop": ["STOP"], "max_tokens": 100}
        exc = Exception("This model doesn't support the stopSequences field")
        recovered = bedrock._recover_completion_params(exc, common)
        assert recovered is not None
        assert "stop" not in recovered
        assert bedrock.stop == ["STOP"]
        # Persisted only after a successful retry.
        bedrock._persist_completion_recovery(common, recovered)
        assert bedrock.stop is None


class TestRecoveryLoop:
    def test_retry_failure_triggers_next_recovery(self):
        # Bedrock strips `stop`, the retry fails on sampling, so the loop runs the sampling
        # backstop next and persists both drops.
        with patch("litellm.completion"), patch("litellm.stream_chunk_builder"):
            node = Bedrock(
                name="b",
                model="bedrock/eu.anthropic.claude-some-future-model",
                connection=AWSConnection(access_key_id="x", secret_access_key="y", region_name="eu-west-1"),
                prompt=Prompt(messages=[{"role": "user", "content": "Hello"}]),
                stop=["DONE"],
                temperature=0.5,
            )

            def fake_completion(**params):
                if params.get("stop"):
                    raise Exception("This model doesn't support the stopSequences field")
                if params.get("temperature") is not None:
                    raise Exception("temperature: Extra inputs are not permitted")
                return _mock_response("ok")

            node._completion = fake_completion
            result = node.execute(MagicMock(messages=None, files=None), config=RunnableConfig(callbacks=[]))

            assert result["content"] == "ok"
            assert node.stop is None
            assert node.temperature is None

    def test_unrecoverable_retry_raises_latest_error(self):
        # The most recent failure surfaces, not the first.
        with patch("litellm.completion"), patch("litellm.stream_chunk_builder"):
            node = Bedrock(
                name="b",
                model="bedrock/eu.anthropic.claude-some-future-model",
                connection=AWSConnection(access_key_id="x", secret_access_key="y", region_name="eu-west-1"),
                prompt=Prompt(messages=[{"role": "user", "content": "Hello"}]),
                stop=["DONE"],
                temperature=0.5,
            )

            def fake_completion(**params):
                if params.get("stop"):
                    raise Exception("This model doesn't support the stopSequences field")
                raise Exception("temperature: Input should be less than or equal to 1.0")

            node._completion = fake_completion
            with pytest.raises(Exception, match="less than or equal to 1.0"):
                node.execute(MagicMock(messages=None, files=None), config=RunnableConfig(callbacks=[]))
