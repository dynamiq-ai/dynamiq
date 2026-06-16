from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from dynamiq.utils.logger import logger

REGISTRY_FILE = Path(__file__).with_name("model_registry.json")

_SYNC_ENV_VAR = "DYNAMIQ_SYNC_MODEL_REGISTRY_TO_LITELLM"
_SYNC_DISABLED_VALUES = {"0", "false", "no", "off"}


def _sync_to_litellm_enabled() -> bool:
    """Whether registry entries should be mirrored into litellm's ``model_cost`` (default on)."""
    return os.getenv(_SYNC_ENV_VAR, "1").strip().lower() not in _SYNC_DISABLED_VALUES


class ModelRegistry:
    """Litellm-compatible model metadata registry backed by a JSON file."""

    def __init__(self, path: Path | None = None) -> None:
        self._models: dict[str, dict[str, Any]] = {}
        if path is not None:
            self.load(path)

    def load(self, path: Path | str) -> None:
        """Load model definitions from a JSON file.

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
            if not isinstance(value, dict):
                continue
            self._models[key.lower()] = value
        logger.debug("ModelRegistry: loaded %d models from %s", len(self._models), path)
        self.sync_to_litellm()

    def sync_to_litellm(self) -> None:
        """Gap-fill loaded registry entries into litellm's in-memory ``model_cost``.

        For every loaded model litellm does NOT already recognize, register it so that
        ``litellm.supports_function_calling`` and ``drop_params`` honor our metadata. Never
        clobbers an entry litellm already knows. Per-process, non-persistent, exception-safe:
        a missing or misbehaving litellm can never break ``import dynamiq``.
        """
        if not _sync_to_litellm_enabled() or not self._models:
            return
        try:
            import litellm
        except Exception as exc:
            logger.debug("ModelRegistry: litellm unavailable, skipping sync: %s", exc)
            return

        registered = skipped = failed = 0
        for key, info in self._models.items():
            try:
                existing = litellm.get_model_info(model=key)  # case-/prefix-tolerant lookup
            except Exception:
                existing = None
            if existing:
                skipped += 1
                continue
            try:
                litellm.register_model({key: info})
                registered += 1
            except Exception as exc:
                failed += 1
                logger.debug("ModelRegistry: failed to register %s into litellm: %s", key, exc)

        log = logger.warning if failed else logger.debug
        log("ModelRegistry->litellm sync: registered=%d skipped=%d failed=%d", registered, skipped, failed)

    def register(self, model: str, info: dict[str, Any]) -> None:
        """Add or update a model entry.

        Args:
            model: Model identifier (e.g. ``"openai/my-ft-model"``).
            info: Dict following the litellm model spec (``max_input_tokens``,
                ``max_output_tokens``, ``supports_vision``, etc.).
        """
        model = model.lower()
        self._models[model] = {**self._models.get(model, {}), **info}

    def _resolve(self, model: str) -> dict[str, Any] | None:
        """Look up model info, stripping the litellm provider prefix if needed.

        Tries the full model string first (e.g. ``"together_ai/zai-org/GLM-5"``),
        then the part after the first ``/`` (e.g. ``"zai-org/GLM-5"``).
        """
        model_lower = model.lower()
        info = self._models.get(model_lower)
        if info is not None:
            return info
        sep = model_lower.find("/")
        if sep != -1:
            return self._models.get(model_lower[sep + 1 :])
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

    def supports_function_calling(self, model: str) -> bool | None:
        """Return function-calling support flag or ``None`` if unknown."""
        info = self._resolve(model)
        if info is None:
            return None
        return info.get("supports_function_calling")

    def supports_pdf_input(self, model: str) -> bool | None:
        """Return PDF input support flag or ``None`` if unknown."""
        info = self._resolve(model)
        if info is None:
            return None
        return info.get("supports_pdf_input")

    def list_models(self) -> list[str]:
        """Return all registered model identifiers."""
        return list(self._models)


model_registry = ModelRegistry(path=REGISTRY_FILE)
