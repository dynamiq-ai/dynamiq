import pytest

from dynamiq.connections import AWS as AWSConnection
from dynamiq.connections import Anthropic as AnthropicConnection
from dynamiq.connections import HttpApiKey
from dynamiq.nodes.llms.anthropic import Anthropic
from dynamiq.nodes.llms.base import SAMPLING_PARAMS
from dynamiq.nodes.llms.bedrock import Bedrock
from dynamiq.nodes.llms.custom_llm import CustomLLM


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
            "claude-opus-4-7",
            "claude-opus-4-8",
            "claude-fable-5",
            "claude-mythos-5",
            "claude-mythos-preview",
            "bedrock/eu.anthropic.claude-opus-4-8-20251101-v1:0",
            "openrouter/anthropic/claude-opus-4-7",
        ],
    )
    def test_unsupported_models_detected(self, model):
        llm = Anthropic(name="a", model=model, connection=AnthropicConnection(api_key="x"))
        assert llm._model_rejects_sampling_params() is True

    @pytest.mark.parametrize("model", ["claude-opus-4-6", "claude-sonnet-4-6", "claude-haiku-4-5", "gpt-4o"])
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
