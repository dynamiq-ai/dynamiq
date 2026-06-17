from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from dynamiq.utils.logger import logger

REGISTRY_FILE = Path(__file__).with_name("model_registry.json")

_SYNC_ENV_VAR = "DYNAMIQ_SYNC_MODEL_REGISTRY_TO_LITELLM"


def _sync_to_litellm_enabled() -> bool:
    """Whether registry entries should be mirrored into litellm's ``model_cost`` (default on).

    Only the literal ``"false"`` (case-insensitive) disables it; unset or any other value
    keeps it on.
    """
    return os.getenv(_SYNC_ENV_VAR, "true").strip().lower() != "false"


class ModelMetadata(BaseModel):
    """Structured, litellm-compatible model metadata.

    The common litellm fields are typed for validation/discoverability. Unknown fields are
    ignored (Pydantic default), so only the fields listed here are carried into the registry.
    """

    max_tokens: int | None = None
    max_input_tokens: int | None = None
    max_output_tokens: int | None = None
    mode: str | None = None
    litellm_provider: str | None = None
    supports_vision: bool | None = None
    supports_pdf_input: bool | None = None
    supports_function_calling: bool | None = None
    supports_parallel_function_calling: bool | None = None
    supports_response_schema: bool | None = None
    supports_system_messages: bool | None = None
    input_cost_per_token: float | None = None
    output_cost_per_token: float | None = None


def _as_info_dict(info: ModelMetadata | dict[str, Any]) -> dict[str, Any]:
    """Normalize a model entry to a plain dict, dropping unset (``None``) fields."""
    if isinstance(info, ModelMetadata):
        return info.model_dump(exclude_none=True)
    return info


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

    def sync_to_litellm(self, models: dict[str, ModelMetadata | dict[str, Any]] | None = None) -> None:
        """Gap-fill registry entries into litellm's in-memory ``model_cost``.

        For every selected model litellm does NOT already recognize, register it so litellm's
        own metadata lookups honor our values. Never clobbers an entry litellm already knows.
        Per-process, non-persistent, exception-safe: a missing or misbehaving litellm can
        never break ``import dynamiq``.

        Args:
            models: New model definitions (``{model_id: info}``) to add to the registry and
                sync directly. Defaults to syncing all already-loaded models.

        Note:
            Registering ``models`` into the registry is unconditional; the
            ``DYNAMIQ_SYNC_MODEL_REGISTRY_TO_LITELLM`` toggle only gates the litellm gap-fill
            below, so callers can still extend the registry with the sync disabled.
        """
        if models is None:
            items = self._models
        else:
            items = {}
            for model, info in models.items():
                self.register(model, info)
                items[model.lower()] = self._models[model.lower()]
        if not _sync_to_litellm_enabled():
            return
        if not items:
            return
        try:
            import litellm
        except Exception as exc:
            logger.debug("ModelRegistry: litellm unavailable, skipping sync: %s", exc)
            return

        registered = skipped = failed = 0
        for key, info in items.items():
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

    def register(self, model: str, info: ModelMetadata | dict[str, Any]) -> None:
        """Add or update a model entry.

        Args:
            model: Model identifier (e.g. ``"openai/my-ft-model"``).
            info: A :class:`ModelMetadata` (preferred) or a litellm-spec dict
                (``max_input_tokens``, ``supports_vision``, etc.). Merged onto any existing
                entry, so partial updates only touch the fields provided.
        """
        model = model.lower()
        self._models[model] = {**self._models.get(model, {}), **_as_info_dict(info)}

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
