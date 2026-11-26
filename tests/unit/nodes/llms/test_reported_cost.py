from types import SimpleNamespace

import pytest
from litellm import ModelResponse

from dynamiq.nodes.llms.base import BaseLLM
from dynamiq.nodes.llms.ollama import Ollama


def _response(**usage):
    return ModelResponse(usage={"prompt_tokens": 57, "completion_tokens": 46, "total_tokens": 103, **usage})


def test_a_model_without_a_litellm_price_takes_the_cost_the_provider_reported():
    usage = BaseLLM.get_usage_data(model="openrouter/vendor/unpriced-model", completion=_response(cost=1.275e-05))

    assert usage.total_tokens_cost_usd == 1.275e-05


def test_a_model_without_a_price_or_a_reported_cost_has_no_cost():
    assert (
        BaseLLM.get_usage_data(model="openrouter/vendor/unpriced-model", completion=_response()).total_tokens_cost_usd
        is None
    )


def test_a_priced_model_keeps_litellm_s_price():
    usage = BaseLLM.get_usage_data(model="gpt-4o-mini", completion=_response(cost=99.0))

    assert 0 < usage.total_tokens_cost_usd < 1


@pytest.mark.parametrize(
    ("raw", "complete"),
    [
        ({"prompt_tokens": 0, "completion_tokens": 0}, True),
        ({"prompt_tokens": 12, "completion_tokens": 7}, True),
        ({"prompt_tokens": 12}, False),
        ({"completion_tokens": 7}, False),
        ({}, False),
    ],
)
def test_usage_completeness_requires_both_observed_counters(raw, complete):
    usage = BaseLLM.get_usage_data(model="gpt-4o-mini", completion=SimpleNamespace(usage=raw))
    assert usage.complete is complete


@pytest.mark.parametrize(
    ("raw", "complete"),
    [
        ({"prompt_eval_count": 0, "eval_count": 0}, True),
        ({"prompt_eval_count": 12, "eval_count": 7}, True),
        ({"prompt_tokens": 0, "completion_tokens": 0}, True),
        ({"prompt_tokens": 12, "completion_tokens": 7}, True),
        ({"prompt_eval_count": 12}, False),
        ({"eval_count": 7}, False),
        ({}, False),
        ({"prompt_eval_count": True, "eval_count": 7}, False),
        ({"prompt_eval_count": -1, "eval_count": 7}, False),
        ({"prompt_tokens": 12, "completion_tokens": False}, False),
        ({"prompt_tokens": 12, "completion_tokens": -1}, False),
        ({"prompt_eval_count": None, "eval_count": 7}, False),
    ],
)
def test_ollama_usage_completeness_distinguishes_reported_zero_from_missing_or_invalid(raw, complete):
    usage = Ollama.get_usage_data(model="ollama/llama32", completion=SimpleNamespace(usage=raw))
    assert usage.complete is complete
    assert usage.total_tokens_cost_usd is None
    assert usage.prompt_tokens_cost_usd is None
    assert usage.completion_tokens_cost_usd is None
    if complete:
        prompt = raw.get("prompt_eval_count", raw.get("prompt_tokens"))
        output = raw.get("eval_count", raw.get("completion_tokens"))
        assert usage.prompt_tokens == prompt
        assert usage.completion_tokens == output
        assert usage.total_tokens == prompt + output
