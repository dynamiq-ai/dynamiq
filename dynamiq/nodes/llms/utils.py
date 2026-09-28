from __future__ import annotations


def litellm_video_input_flag(model: str) -> bool | None:
    """Read litellm's per-model ``supports_video_input`` flag from ``litellm.model_cost``.

    litellm's public ``get_model_info()`` drops this field (not part of its ``ModelInfo``
    schema); this reaches past it the same way litellm's own ``get_supported_regions()``
    does for another omitted field. ``_get_model_info_helper`` is private and unversioned,
    hence the try/except -- if litellm renames it, this just stops contributing.

    Returns ``None`` (not ``False``) when litellm has no explicit answer, since its
    video-input coverage is sparse even for models that do support it.
    """
    try:
        import litellm
        from litellm.utils import _get_model_info_helper

        resolved_model, provider, _, _ = litellm.get_llm_provider(model=model)
        info = _get_model_info_helper(model=resolved_model, custom_llm_provider=provider)
        key = info.get("key")
        if key is None:
            return None
        return litellm.model_cost.get(key, {}).get("supports_video_input")
    except Exception:
        return None
