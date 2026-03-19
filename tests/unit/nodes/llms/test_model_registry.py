"""Unit tests for the custom model registry and BaseLLM fallback integration."""

import json
from unittest.mock import patch

import pytest

from dynamiq.nodes.llms.base import LLM_DEFAULT_MAX_TOKENS
from dynamiq.nodes.llms.registry import ModelRegistry

MODEL_A = "test-org/model-a"
MODEL_B = "test-org/model-b"

TEST_REGISTRY_DATA = {
    MODEL_A: {
        "max_input_tokens": 50_000,
        "max_output_tokens": 10_000,
        "max_tokens": 10_000,
        "supports_vision": False,
        "supports_pdf_input": False,
    },
    MODEL_B: {
        "max_input_tokens": 200_000,
        "max_output_tokens": 64_000,
        "max_tokens": 64_000,
        "supports_vision": True,
        "supports_pdf_input": True,
    },
}


@pytest.fixture()
def registry(tmp_path) -> ModelRegistry:
    path = tmp_path / "models.json"
    path.write_text(json.dumps(TEST_REGISTRY_DATA))
    return ModelRegistry(path=path)


@pytest.mark.parametrize(
    ("model", "expected_tokens", "expected_vision"),
    [
        (MODEL_A, 50_000, False),
        (MODEL_B, 200_000, True),
        (f"together_ai/{MODEL_A}", 50_000, False),
        (f"together_ai/{MODEL_B}", 200_000, True),
        (f"openai/{MODEL_A}", 50_000, False),
        (f"together_ai/{MODEL_A.upper()}", 50_000, False),
    ],
)
def test_registry_resolves_tokens_and_vision(registry, model, expected_tokens, expected_vision):
    assert registry.get_max_tokens(model) == expected_tokens
    assert registry.supports_vision(model) is expected_vision


def test_unknown_model_returns_none(registry):
    assert registry.get_model_info("unknown/model") is None
    assert registry.get_max_tokens("unknown/model") is None


@pytest.fixture()
def _litellm_unknown():
    with (
        patch("dynamiq.nodes.llms.base.get_model_info", side_effect=Exception("Unknown")),
        patch("dynamiq.nodes.llms.base.get_max_tokens", side_effect=Exception("Unknown")),
        patch("dynamiq.nodes.llms.base.supports_vision", side_effect=Exception("Unknown")),
        patch("dynamiq.nodes.llms.base.supports_pdf_input", side_effect=Exception("Unknown")),
    ):
        yield


@pytest.fixture()
def _patch_registry(registry):
    with patch("dynamiq.nodes.llms.registry.model_registry", registry):
        yield


@pytest.mark.usefixtures("_litellm_unknown", "_patch_registry")
@pytest.mark.parametrize(
    ("model", "expected_tokens", "expected_vision"),
    [
        (MODEL_A, 50_000, False),
        (MODEL_B, 200_000, True),
    ],
)
def test_basellm_falls_back_to_registry(model, expected_tokens, expected_vision):
    from dynamiq.connections import TogetherAI as TogetherAIConnection
    from dynamiq.nodes.llms.togetherai import TogetherAI

    llm = TogetherAI(model=model, connection=TogetherAIConnection(api_key="test-key"))

    assert llm.model == f"together_ai/{model}"
    assert llm.get_token_limit() == expected_tokens
    assert llm.is_vision_supported is expected_vision


@pytest.mark.usefixtures("_litellm_unknown", "_patch_registry")
def test_totally_unknown_model_returns_default():
    from dynamiq.connections import TogetherAI as TogetherAIConnection
    from dynamiq.nodes.llms.togetherai import TogetherAI

    llm = TogetherAI(model="unknown/x", connection=TogetherAIConnection(api_key="test-key"))
    assert llm.get_token_limit() == LLM_DEFAULT_MAX_TOKENS
