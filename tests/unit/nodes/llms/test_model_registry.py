"""Unit tests for the custom model registry and BaseLLM fallback integration."""

import json
from unittest.mock import patch

import pytest

from dynamiq.nodes.llms.base import LLM_DEFAULT_MAX_TOKENS
from dynamiq.nodes.llms.registry import ModelMetadata, ModelRegistry, model_registry

MODEL_A = "test-org/model-a"
MODEL_B = "test-org/model-b"
MODEL_C = "test-org/model-c"  # registry entry that omits the supports_function_calling flag

TEST_REGISTRY_DATA = {
    MODEL_A: {
        "max_input_tokens": 50_000,
        "max_output_tokens": 10_000,
        "max_tokens": 10_000,
        "supports_vision": False,
        "supports_pdf_input": False,
        "supports_function_calling": False,
    },
    MODEL_B: {
        "max_input_tokens": 200_000,
        "max_output_tokens": 64_000,
        "max_tokens": 64_000,
        "supports_vision": True,
        "supports_pdf_input": True,
        "supports_function_calling": True,
    },
    MODEL_C: {
        "max_input_tokens": 32_000,
        "max_tokens": 32_000,
        "supports_vision": False,
    },
}


@pytest.fixture(autouse=True)
def _restore_litellm_model_cost():
    """Keep the suite hermetic: ``sync_to_litellm`` mutates the global ``litellm.model_cost``."""
    import litellm

    snapshot = dict(litellm.model_cost)
    yield
    litellm.model_cost.clear()
    litellm.model_cost.update(snapshot)


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


@pytest.mark.parametrize(
    ("model", "expected_fc"),
    [
        (MODEL_A, False),
        (MODEL_B, True),
        (f"together_ai/{MODEL_A}", False),
        (f"together_ai/{MODEL_B}", True),
        (f"together_ai/{MODEL_B.upper()}", True),
        (MODEL_C, None),  # entry exists but omits the flag
        ("unknown/model", None),
    ],
)
def test_registry_supports_function_calling(registry, model, expected_fc):
    assert registry.supports_function_calling(model) is expected_fc


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
    with patch("dynamiq.nodes.llms.base.model_registry", registry):
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


@pytest.mark.usefixtures("_litellm_unknown", "_patch_registry")
def test_model_info_override_takes_priority():
    """model_info on the LLM instance overrides both litellm and model_registry."""
    from dynamiq.connections import TogetherAI as TogetherAIConnection
    from dynamiq.nodes.llms.togetherai import TogetherAI

    llm = TogetherAI(
        model=MODEL_A,
        connection=TogetherAIConnection(api_key="test-key"),
        model_info={"max_input_tokens": 99_999, "supports_vision": True, "supports_pdf_input": True},
    )

    # Registry has 50_000 for MODEL_A, but model_info should win
    assert llm.get_token_limit() == 99_999
    assert llm.is_vision_supported is True
    assert llm.is_pdf_input_supported is True


def test_registry_supports_video_input():
    """Video-input support has no litellm signal at all -- it only ever comes from an
    explicit registry entry or a model_info override."""
    import json as _json

    data = {
        "video-model": {"supports_video_input": True},
        "no-video-model": {"supports_video_input": False},
        "unflagged-model": {"max_input_tokens": 1000},
    }
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "models.json"
        path.write_text(_json.dumps(data))
        reg = ModelRegistry(path=path)

    assert reg.supports_video_input("video-model") is True
    assert reg.supports_video_input("no-video-model") is False
    assert reg.supports_video_input("unflagged-model") is None
    assert reg.supports_video_input("unknown-model") is None


@pytest.mark.usefixtures("_litellm_unknown", "_patch_registry")
def test_basellm_video_input_not_in_registry_defaults_false():
    """A model with no supports_video_input entry (litellm has no such signal either)
    defaults to unsupported."""
    from dynamiq.connections import TogetherAI as TogetherAIConnection
    from dynamiq.nodes.llms.togetherai import TogetherAI

    llm = TogetherAI(model=MODEL_B, connection=TogetherAIConnection(api_key="test-key"))
    assert llm.is_video_input_supported is False


def test_basellm_video_input_model_info_override_takes_priority():
    """model_info.supports_video_input overrides the registry regardless of what litellm
    itself knows about the model."""
    from dynamiq.connections import TogetherAI as TogetherAIConnection
    from dynamiq.nodes.llms.togetherai import TogetherAI

    llm = TogetherAI(
        model=MODEL_A,
        connection=TogetherAIConnection(api_key="test-key"),
        model_info={"supports_video_input": True},
    )
    assert llm.is_video_input_supported is True


@pytest.mark.usefixtures("_litellm_unknown", "_patch_registry")
def test_model_info_none_falls_through():
    """When model_info is None, the existing lookup chain is used."""
    from dynamiq.connections import TogetherAI as TogetherAIConnection
    from dynamiq.nodes.llms.togetherai import TogetherAI

    llm = TogetherAI(model=MODEL_B, connection=TogetherAIConnection(api_key="test-key"))
    assert llm.model_info is None
    assert llm.get_token_limit() == 200_000
    assert llm.is_vision_supported is True


@pytest.mark.usefixtures("_litellm_unknown", "_patch_registry")
@pytest.mark.parametrize(("model", "expected_fc"), [(MODEL_A, False), (MODEL_B, True)])
def test_basellm_function_calling_falls_back_to_registry(model, expected_fc):
    """When litellm doesn't know the model, FC support comes from the custom registry."""
    from dynamiq.connections import TogetherAI as TogetherAIConnection
    from dynamiq.nodes.llms.togetherai import TogetherAI

    llm = TogetherAI(model=model, connection=TogetherAIConnection(api_key="test-key"))
    assert llm.is_function_calling_supported is expected_fc


@pytest.mark.usefixtures("_litellm_unknown", "_patch_registry")
def test_basellm_unknown_function_calling_defaults_true(mocker):
    """A model unknown to both litellm and the registry is assumed FC-capable (warn + allow)."""
    from dynamiq.connections import TogetherAI as TogetherAIConnection
    from dynamiq.nodes.llms.togetherai import TogetherAI

    mock_logger = mocker.patch("dynamiq.nodes.llms.base.logger")
    llm = TogetherAI(model="unknown/x", connection=TogetherAIConnection(api_key="test-key"))

    assert llm.is_function_calling_supported is True
    msg = mock_logger.warning.call_args.args[0]
    assert "unknown to litellm and the custom registry" in msg


@pytest.mark.usefixtures("_litellm_unknown", "_patch_registry")
def test_basellm_registry_entry_without_fc_flag_allows_with_accurate_log(mocker):
    """A registry entry that omits the FC flag is allowed (not hard-blocked), but the
    warning must not falsely claim the model is unknown to the registry."""
    from dynamiq.connections import TogetherAI as TogetherAIConnection
    from dynamiq.nodes.llms.togetherai import TogetherAI

    mock_logger = mocker.patch("dynamiq.nodes.llms.base.logger")
    llm = TogetherAI(model=MODEL_C, connection=TogetherAIConnection(api_key="test-key"))

    assert llm.is_function_calling_supported is True
    msg = mock_logger.warning.call_args.args[0]
    assert "registry entry" in msg
    assert "unknown to litellm and the custom registry" not in msg


@pytest.mark.usefixtures("_litellm_unknown", "_patch_registry")
def test_model_info_function_calling_override_takes_priority():
    """model_info.supports_function_calling overrides both litellm and the registry."""
    from dynamiq.connections import TogetherAI as TogetherAIConnection
    from dynamiq.nodes.llms.togetherai import TogetherAI

    # Registry says MODEL_B supports FC, but the explicit override wins.
    llm = TogetherAI(
        model=MODEL_B,
        connection=TogetherAIConnection(api_key="test-key"),
        model_info={"supports_function_calling": False},
    )
    assert llm.is_function_calling_supported is False


def test_basellm_uses_litellm_when_model_is_known():
    """When litellm knows the model, its FC verdict is used (registry not consulted)."""
    from dynamiq.connections import TogetherAI as TogetherAIConnection
    from dynamiq.nodes.llms.togetherai import TogetherAI

    llm = TogetherAI(model=MODEL_A, connection=TogetherAIConnection(api_key="test-key"))
    with (
        patch("dynamiq.nodes.llms.base.get_model_info", return_value={"supports_function_calling": False}),
        patch("dynamiq.nodes.llms.base.supports_function_calling", return_value=False) as mock_fc,
    ):
        assert llm.is_function_calling_supported is False
        mock_fc.assert_called_once()


# ---------------------------------------------------------------------------
# sync_to_litellm: gap-fill registry entries into litellm.model_cost
# ---------------------------------------------------------------------------


def test_sync_registers_unknown_model_into_litellm(monkeypatch):
    """After sync, a model litellm didn't know reports FC support via litellm itself."""
    import litellm

    monkeypatch.setenv("DYNAMIQ_SYNC_MODEL_REGISTRY_TO_LITELLM", "1")

    reg = ModelRegistry()
    reg._models = {MODEL_B.lower(): dict(TEST_REGISTRY_DATA[MODEL_B])}

    # Sanity: litellm must not already know this synthetic model.
    with pytest.raises(Exception):
        litellm.get_model_info(model=MODEL_B.lower())

    reg.sync_to_litellm()

    assert litellm.get_model_info(model=MODEL_B.lower())["supports_function_calling"] is True
    # litellm's lookup is provider-prefix tolerant, so the prefixed form resolves too.
    assert litellm.supports_function_calling(f"together_ai/{MODEL_B}") is True


def test_sync_does_not_clobber_known_litellm_model(monkeypatch):
    """A model litellm already knows must keep litellm's own metadata, not ours."""
    import litellm

    monkeypatch.setenv("DYNAMIQ_SYNC_MODEL_REGISTRY_TO_LITELLM", "1")

    known = "gpt-4o-mini"  # shipped in litellm's model map
    original = dict(litellm.get_model_info(model=known))

    reg = ModelRegistry()
    # Deliberately poisoned metadata for a model litellm already knows.
    reg._models = {known: {"max_input_tokens": 1, "supports_function_calling": False}}
    reg.sync_to_litellm()

    after = litellm.get_model_info(model=known)
    assert after["max_input_tokens"] == original["max_input_tokens"]
    assert after["max_input_tokens"] != 1


def test_sync_refreshes_known_canonical_provider_model(monkeypatch, mocker):
    """Canonical provider keys may refresh explicit fields without dropping provider metadata."""
    import litellm

    monkeypatch.setenv("DYNAMIQ_SYNC_MODEL_REGISTRY_TO_LITELLM", "1")
    existing = {
        "litellm_provider": "test_provider",
        "max_input_tokens": 1,
        "provider_owned_field": "preserved",
    }
    mocker.patch.object(litellm, "get_model_info", return_value=existing)
    register_model = mocker.patch.object(litellm, "register_model")

    reg = ModelRegistry()
    reg._models = {
        "test_provider/model-a": {
            "litellm_provider": "test_provider",
            "max_input_tokens": 2,
        }
    }
    reg.sync_to_litellm()

    register_model.assert_called_once_with(
        {
            "test_provider/model-a": {
                "litellm_provider": "test_provider",
                "max_input_tokens": 2,
                "provider_owned_field": "preserved",
            }
        }
    )


def test_sync_disabled_still_registers_models_into_registry(monkeypatch):
    """The disable env only gates the litellm gap-fill -- models passed to sync_to_litellm
    must still be added to the registry itself, and must NOT leak into litellm."""
    import litellm

    monkeypatch.setenv("DYNAMIQ_SYNC_MODEL_REGISTRY_TO_LITELLM", "false")

    reg = ModelRegistry()
    reg.sync_to_litellm(models={MODEL_B: ModelMetadata(**TEST_REGISTRY_DATA[MODEL_B])})

    # Registry registration is unconditional.
    assert reg.get_model_info(MODEL_B) is not None
    assert reg.supports_function_calling(MODEL_B) is True
    # ...but the litellm sync was skipped (check model_cost directly; get_model_info caches).
    assert MODEL_B.lower() not in litellm.model_cost


def test_sync_respects_disable_env(monkeypatch):
    """With the env var disabled, nothing is registered into litellm."""
    import litellm

    monkeypatch.setenv("DYNAMIQ_SYNC_MODEL_REGISTRY_TO_LITELLM", "false")

    reg = ModelRegistry()
    reg._models = {MODEL_A.lower(): dict(TEST_REGISTRY_DATA[MODEL_A])}
    reg.sync_to_litellm()

    with pytest.raises(Exception):
        litellm.get_model_info(model=MODEL_A.lower())


def test_sync_keeps_function_calling_params_under_drop_params(monkeypatch):
    """Core fix: under ``drop_params=True`` litellm silently strips ``tools`` for a model it
    doesn't know to support function calling, so FUNCTION_CALLING mode breaks. Once our
    registry entry (marking the model FC-capable) is synced, litellm keeps ``tools``."""
    import litellm

    monkeypatch.setenv("DYNAMIQ_SYNC_MODEL_REGISTRY_TO_LITELLM", "1")

    # together_ai gates `tools` on per-model FC support, so an unknown model gets them
    # dropped under drop_params (this is the bug this branch fixes).
    model = "together_ai/unknown-fc-model"
    tools = [
        {
            "type": "function",
            "function": {"name": "noop", "description": "noop", "parameters": {"type": "object", "properties": {}}},
        }
    ]

    def tools_survive() -> bool:
        params = litellm.utils.get_optional_params(
            model=model,
            custom_llm_provider="together_ai",
            tools=tools,
            tool_choice="auto",
            drop_params=True,
        )
        return "tools" in params

    # Before sync: litellm doesn't know the model supports FC -> tools dropped.
    assert tools_survive() is False

    reg = ModelRegistry()
    reg.sync_to_litellm(
        models={model: ModelMetadata(supports_function_calling=True, mode="chat", litellm_provider="together_ai")}
    )

    # After sync: litellm now knows the model is FC-capable -> tools preserved.
    assert tools_survive() is True


def test_sync_is_exception_safe_without_litellm(monkeypatch):
    """If importing litellm fails, sync must swallow it and not raise."""
    import builtins

    real_import = builtins.__import__

    def _boom(name, *args, **kwargs):
        if name == "litellm":
            raise ImportError("simulated")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _boom)

    reg = ModelRegistry()
    reg._models = {MODEL_A.lower(): dict(TEST_REGISTRY_DATA[MODEL_A])}
    reg.sync_to_litellm()  # must not raise


@pytest.mark.parametrize(
    ("model_id", "expected"),
    [
        (
            "MiniMax-M3",
            {
                "max_input_tokens": 1_000_000,
                "max_output_tokens": 128_000,
                "supports_vision": True,
                "supports_video_input": True,
                "supports_adaptive_thinking": True,
            },
        ),
        (
            "MiniMax-M2.7",
            {
                "max_input_tokens": 204_800,
                "max_output_tokens": 131_072,
                "supports_vision": False,
                "supports_video_input": False,
                "supports_adaptive_thinking": False,
                "supports_tool_choice": True,
            },
        ),
    ],
)
def test_minimax_registry_resolves_official_model_ids(model_id, expected):
    info = model_registry.get_model_info(model_id)

    assert info is not None
    assert {field: info[field] for field in expected} == expected
    assert model_registry.get_model_info(f"minimax/{model_id}") == info
    assert model_registry.get_model_info(f"anthropic/{model_id}") == info
    assert f"minimax/{model_id}".lower() in model_registry.list_models()
    assert ModelMetadata(**info).model_dump(exclude_none=True) == info


def test_minimax_m3_registry_preserves_tiered_pricing():
    import litellm

    info = model_registry.get_model_info("MiniMax-M3")

    assert info is not None
    expected_pricing = {
        "input_cost_per_token": 0.0000003,
        "output_cost_per_token": 0.0000012,
        "cache_read_input_token_cost": 0.00000006,
        "input_cost_per_token_priority": 0.00000045,
        "output_cost_per_token_priority": 0.0000018,
        "cache_read_input_token_cost_priority": 0.00000009,
        "input_cost_per_token_above_512k_tokens": 0.0000006,
        "output_cost_per_token_above_512k_tokens": 0.0000024,
        "cache_read_input_token_cost_above_512k_tokens": 0.00000012,
    }
    custom_pricing = {field: info[field] for field in expected_pricing}
    litellm_info = litellm.get_model_info(model="minimax/MiniMax-M3")

    # Every field we ship for M3 is a litellm-recognized pricing key, so after sync litellm honors our values.
    # (litellm has no "above 512k + priority" combined key, so that tier is intentionally not modelled here.)
    assert custom_pricing == expected_pricing
    assert {field: litellm_info[field] for field in expected_pricing} == expected_pricing
    assert litellm.cost_per_token(
        model="MiniMax-M3",
        custom_llm_provider="minimax",
        prompt_tokens=1,
        completion_tokens=1,
        service_tier="priority",
    ) == pytest.approx((0.00000045, 0.0000018))
    assert litellm.cost_per_token(
        model="MiniMax-M3",
        custom_llm_provider="minimax",
        prompt_tokens=512_001,
        completion_tokens=1,
    ) == pytest.approx((512_001 * 0.0000006, 0.0000024))
