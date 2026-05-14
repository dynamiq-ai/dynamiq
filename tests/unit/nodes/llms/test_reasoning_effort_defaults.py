import pytest

from dynamiq.connections import OpenAI as OpenAIConnection
from dynamiq.nodes.llms.openai import (
    OpenAI,
    ReasoningEffort,
    _MODELS_DEFAULTING_TO_NONE,
    _resolve_default_reasoning_effort,
)


def _llm(model: str, reasoning_effort: ReasoningEffort | None = ReasoningEffort.AUTO) -> OpenAI:
    return OpenAI(
        model=model,
        connection=OpenAIConnection(api_key="test-key"),
        reasoning_effort=reasoning_effort,
    )


def _build(llm: OpenAI) -> dict:
    """Run the OpenAI subclass's params-shaping logic against an empty base dict."""
    return llm.update_completion_params({"max_tokens": None})


class TestResolveDefaultReasoningEffort:
    @pytest.mark.parametrize("model", sorted(_MODELS_DEFAULTING_TO_NONE))
    def test_models_in_none_set_resolve_to_none(self, model):
        assert _resolve_default_reasoning_effort(model) is None

    @pytest.mark.parametrize(
        "model",
        [
            "gpt-5",
            "gpt-5-mini",
            "gpt-5-nano",
            "gpt-5-codex",
            "gpt-5.1-codex",
            "gpt-5.1-codex-max",
            "gpt-5.1-codex-mini",
            "gpt-5.2-pro",
            "gpt-5.4-pro",
            "o3",
            "o3-mini",
            "o4-mini",
            "gpt-4o",
        ],
    )
    def test_unknown_or_non_none_models_fall_back_to_medium(self, model):
        assert _resolve_default_reasoning_effort(model) == ReasoningEffort.MEDIUM


class TestAutoResolutionInCompletionParams:
    @pytest.mark.parametrize(
        "model",
        ["gpt-5.1", "gpt-5.2", "gpt-5.4", "gpt-5.1-2025-11-13", "gpt-5.2-2025-12-11"],
    )
    def test_auto_omits_reasoning_effort_for_native_none_models(self, model):
        params = _build(_llm(model))
        assert "reasoning_effort" not in params

    @pytest.mark.parametrize("model", ["gpt-5", "gpt-5-mini", "gpt-5-nano"])
    def test_auto_falls_back_to_medium_for_other_gpt5_models(self, model):
        params = _build(_llm(model))
        assert params["reasoning_effort"] == ReasoningEffort.MEDIUM

    @pytest.mark.parametrize("model", ["o3", "o3-mini", "o4-mini"])
    def test_auto_falls_back_to_medium_for_o_series(self, model):
        params = _build(_llm(model))
        assert params["reasoning_effort"] == ReasoningEffort.MEDIUM

    @pytest.mark.parametrize("model", ["gpt-5-pro", "gpt-5.2-pro", "gpt-5.4-pro"])
    def test_pro_models_always_high_regardless_of_field(self, model):
        # Existing opinionated override: pro family forces HIGH.
        for effort in (ReasoningEffort.AUTO, ReasoningEffort.LOW, ReasoningEffort.MINIMAL):
            params = _build(_llm(model, reasoning_effort=effort))
            assert params["reasoning_effort"] == ReasoningEffort.HIGH

    def test_chat_variant_never_gets_reasoning_effort(self):
        params = _build(_llm("gpt-5-chat"))
        assert "reasoning_effort" not in params

    def test_non_reasoning_model_path_does_not_inject(self):
        # gpt-4o is not o-series and not gpt-5*, so neither branch runs.
        params = _build(_llm("gpt-4o"))
        assert "reasoning_effort" not in params


class TestExplicitOverrideRespected:
    @pytest.mark.parametrize(
        ("model", "explicit"),
        [
            ("gpt-5.1", ReasoningEffort.LOW),
            ("gpt-5.1", ReasoningEffort.MEDIUM),
            ("gpt-5.2", ReasoningEffort.HIGH),
            ("gpt-5.4", ReasoningEffort.MINIMAL),
            ("gpt-5", ReasoningEffort.LOW),
            ("o4-mini", ReasoningEffort.HIGH),
        ],
    )
    def test_explicit_value_is_sent_verbatim(self, model, explicit):
        params = _build(_llm(model, reasoning_effort=explicit))
        assert params["reasoning_effort"] == explicit

    @pytest.mark.parametrize("model", ["gpt-5.1", "gpt-5"])
    def test_explicit_none_omits_param(self, model):
        params = _build(_llm(model, reasoning_effort=None))
        assert "reasoning_effort" not in params


class TestPrefixedModelIds:
    def test_openai_prefix_is_stripped_before_resolution(self):
        params = _build(_llm("openai/gpt-5.1"))
        assert "reasoning_effort" not in params
