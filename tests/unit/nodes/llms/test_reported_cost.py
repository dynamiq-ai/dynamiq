from litellm import ModelResponse

from dynamiq.nodes.llms.base import BaseLLM


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
