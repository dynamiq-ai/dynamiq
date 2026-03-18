"""
Custom LLM model registry.

Provides model metadata (token limits, capability flags) as a fallback
when litellm's built-in registry does not recognise a model.

The registry uses the same JSON format as litellm's
``model_prices_and_context_window.json``.  On import it loads
``model_registry.json`` from this directory (if present).  Additional
entries can be registered at runtime via ``model_registry.register()``,
or by placing/editing the JSON file.

Usage::

    from dynamiq.nodes.llms.registry import model_registry

    # Programmatic registration
    model_registry.register("my-provider/custom-model", {
        "max_input_tokens": 128_000,
        "max_output_tokens": 16_384,
        "supports_vision": True,
    })

    # Lookup (returns None on miss)
    info = model_registry.get_model_info("my-provider/custom-model")
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from dynamiq.utils.logger import logger

_REGISTRY_FILE = Path(__file__).with_name("model_registry.json")

_DEFAULT_MAX_TOKENS = 4096


class ModelRegistry:
    """Litellm-compatible model metadata registry backed by a JSON file."""

    def __init__(self, path: Path | None = None) -> None:
        self._models: dict[str, dict[str, Any]] = {}
        if path is not None:
            self.load(path)

    def load(self, path: Path | str) -> None:
        """Load model definitions from a JSON file (litellm format).

        The ``sample_spec`` key is silently skipped.

        Args:
            path: Path to the JSON file.
        """
        path = Path(path)
        if not path.exists():
            logger.debug("ModelRegistry: %s not found, skipping.", path)
            return
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("ModelRegistry: failed to load %s: %s", path, exc)
            return
        for key, value in data.items():
            if key == "sample_spec" or not isinstance(value, dict):
                continue
            self._models[key] = value
        logger.debug("ModelRegistry: loaded %d models from %s", len(self._models), path)

    def register(self, model: str, info: dict[str, Any]) -> None:
        """Add or update a model entry.

        Args:
            model: Model identifier (e.g. ``"openai/my-ft-model"``).
            info: Dict following the litellm model spec (``max_input_tokens``,
                ``max_output_tokens``, ``supports_vision``, etc.).
        """
        self._models[model] = {**self._models.get(model, {}), **info}

    def _resolve(self, model: str) -> dict[str, Any] | None:
        """Look up model info, stripping the litellm provider prefix if needed.

        Tries the full model string first (e.g. ``"together_ai/zai-org/GLM-5"``),
        then the part after the first ``/`` (e.g. ``"zai-org/GLM-5"``).
        """
        info = self._models.get(model)
        if info is not None:
            return info
        sep = model.find("/")
        if sep != -1:
            return self._models.get(model[sep + 1 :])
        return None

    def get_model_info(self, model: str) -> dict[str, Any] | None:
        """Return the full info dict or ``None`` if the model is unknown."""
        return self._resolve(model)

    def get_max_tokens(self, model: str) -> int | None:
        """Return ``max_input_tokens`` for *model*, or ``None`` if unknown."""
        info = self._resolve(model)
        if info is None:
            return None
        return info.get("max_input_tokens") or info.get("max_tokens")

    def supports_vision(self, model: str) -> bool | None:
        """Return vision support flag or ``None`` if unknown."""
        info = self._resolve(model)
        if info is None:
            return None
        return info.get("supports_vision")

    def supports_pdf_input(self, model: str) -> bool | None:
        """Return PDF input support flag or ``None`` if unknown."""
        info = self._resolve(model)
        if info is None:
            return None
        return info.get("supports_pdf_input")

    def list_models(self) -> list[str]:
        """Return all registered model identifiers."""
        return list(self._models)


model_registry = ModelRegistry(path=_REGISTRY_FILE)
