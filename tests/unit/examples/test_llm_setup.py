import importlib
import sys
import types

import pytest

EXPECTED_SUPPORTED_PROVIDERS = ("claude", "gpt", "cohere", "groq", "gemini")


def _install_dynamiq_stubs(monkeypatch):
    dynamiq_module = types.ModuleType("dynamiq")
    connections_module = types.ModuleType("dynamiq.connections")
    nodes_module = types.ModuleType("dynamiq.nodes")
    llms_module = types.ModuleType("dynamiq.nodes.llms")

    for name in ("Anthropic", "Cohere", "Gemini", "Groq", "OpenAI"):
        monkeypatch.setattr(
            connections_module, name, type(f"{name}Connection", (), {}), raising=False
        )

    monkeypatch.setitem(sys.modules, "dynamiq", dynamiq_module)
    monkeypatch.setitem(sys.modules, "dynamiq.connections", connections_module)
    monkeypatch.setitem(sys.modules, "dynamiq.nodes", nodes_module)
    monkeypatch.setitem(sys.modules, "dynamiq.nodes.llms", llms_module)

    for module_name, class_name in (
        ("anthropic", "Anthropic"),
        ("cohere", "Cohere"),
        ("gemini", "Gemini"),
        ("groq", "Groq"),
        ("openai", "OpenAI"),
    ):
        module = types.ModuleType(f"dynamiq.nodes.llms.{module_name}")
        monkeypatch.setattr(
            module,
            class_name,
            type(class_name, (), {"__init__": lambda self, **kwargs: None}),
            raising=False,
        )
        monkeypatch.setitem(sys.modules, module.__name__, module)


@pytest.fixture()
def import_llm_setup(monkeypatch):
    _install_dynamiq_stubs(monkeypatch)

    def _import(module_path):
        sys.modules.pop(module_path, None)
        return importlib.import_module(module_path)

    return _import


@pytest.mark.parametrize(
    "module_path",
    [
        "examples.llm_setup",
        "examples.components.tools.extra_utils.utils_llm",
    ],
)
def test_invalid_model_provider_error_lists_supported_providers(
    import_llm_setup, module_path
):
    llm_setup = import_llm_setup(module_path)
    with pytest.raises(ValueError) as exc_info:
        llm_setup.setup_llm(model_provider="watsonx")

    message = str(exc_info.value)
    assert "Invalid model provider: watsonx" in message
    for provider in EXPECTED_SUPPORTED_PROVIDERS:
        assert provider in message
