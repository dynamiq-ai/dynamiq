from typing import Any, ClassVar, Literal

from pydantic import BaseModel

from dynamiq.connections import Anthropic as AnthropicConnection
from dynamiq.nodes.llms.base import BaseLLM
from dynamiq.prompts.prompts import VisionMessageType
from dynamiq.utils.logger import logger


def _patch_litellm_anthropic_strict_forward() -> bool:
    """Make LiteLLM forward ``strict: true`` from OpenAI-shape tools to Anthropic.

    LiteLLM's ``AnthropicConfig._map_tool_helper`` translates an OpenAI-style
    tool definition into Anthropic's native shape but drops the ``strict``
    field on the way through. Until that's fixed upstream, this monkey-patch
    wraps the method and lifts ``function.strict`` onto the resulting Anthropic
    tool dict so Anthropic's grammar-constrained sampling actually engages.

    The patch is defensive: it logs and returns ``False`` if anything looks
    different from what we expect (LiteLLM moved the method, refactored the
    config class, etc.) — in that case strict-on-Anthropic via LiteLLM stops
    working but nothing else breaks. It is also idempotent: re-importing this
    module won't double-patch.

    Returns:
        True if the patch was applied (or already in place), False if it was
        skipped due to unexpected LiteLLM internals.
    """
    try:
        from litellm.llms.anthropic.chat import transformation as _xform
    except Exception as exc:  # ImportError, ModuleNotFoundError, etc.
        logger.debug("LiteLLM Anthropic strict patch: import failed: %s", exc)
        return False

    config_cls = getattr(_xform, "AnthropicConfig", None)
    if config_cls is None:
        logger.warning("LiteLLM Anthropic strict patch: AnthropicConfig not found; skipping.")
        return False

    original = getattr(config_cls, "_map_tool_helper", None)
    if original is None:
        logger.warning("LiteLLM Anthropic strict patch: _map_tool_helper not found; skipping.")
        return False

    if getattr(original, "__dynamiq_strict_patch__", False):
        return True  # already applied

    def _patched(self, tool):
        returned_tool, mcp_server = original(self, tool)
        try:
            if returned_tool is not None and isinstance(tool, dict) and tool.get("type") == "function":
                fn = tool.get("function") or {}
                strict = fn.get("strict")
                if strict is not None:
                    returned_tool["strict"] = strict
        except Exception as exc:
            # Never let a patching failure break the LiteLLM call path.
            logger.debug("LiteLLM Anthropic strict patch: lift failed: %s", exc)
        return returned_tool, mcp_server

    _patched.__dynamiq_strict_patch__ = True
    config_cls._map_tool_helper = _patched
    return True


_LITELLM_ANTHROPIC_STRICT_PATCHED = _patch_litellm_anthropic_strict_forward()
if _LITELLM_ANTHROPIC_STRICT_PATCHED:
    logger.debug("Patched LiteLLM AnthropicConfig to forward `strict: true` for Anthropic tools.")

# Anthropic strict tool use — per-request caps documented in the API:
# https://platform.claude.com/docs/en/agents-and-tools/tool-use/strict-tool-use
ANTHROPIC_MAX_STRICT_TOOLS = 20


def _clean_anthropic_strict_schema(schema: Any) -> Any:
    """Recursively clean a schema for Anthropic's strict tool-use mode.

    - Forces ``additionalProperties: false`` on every object that declares
      ``properties``.
    - Free-form objects (``dict[str, Any]`` → ``{"type": "object"}`` with no
      ``properties``) are converted to JSON-encoded string fields, since strict
      mode can't express an open object. The agent parses them back to dicts
      before tool validation (see ``_normalize_fields``).
    - Optional fields stay omitted from ``required`` (Anthropic's native shape;
      no null-union trick).
    """
    if not isinstance(schema, dict):
        return schema

    schema_type = schema.get("type")
    is_object = schema_type == "object" or (isinstance(schema_type, list) and "object" in schema_type)
    if is_object and "properties" not in schema:
        desc = schema.get("description", "")
        return {
            "type": "string",
            "description": (f"{desc} " if desc else "") + "Provide as a JSON-encoded object string.",
        }

    cleaned: dict = {}
    for key, value in schema.items():
        if key == "default" and value is None:
            # A null default conveys optionality, which Anthropic expresses via
            # ``required`` omission. Drop it so it can't clash with a now non-null type.
            continue
        if key == "type" and isinstance(value, list):
            # Anthropic conveys optionality by omitting the field from ``required``,
            # not via a null-union. Strip ``null`` so a nullable scalar/enum keeps a
            # single declared type (e.g. ``["string", "null"]`` -> ``"string"``);
            # Anthropic rejects an enum whose declared type is ``["string", "null"]``.
            non_null = [t for t in value if t != "null"]
            cleaned["type"] = non_null[0] if len(non_null) == 1 else (non_null or value)
        elif key == "properties" and isinstance(value, dict):
            cleaned["properties"] = {k: _clean_anthropic_strict_schema(v) for k, v in value.items()}
        elif key == "items" and isinstance(value, dict):
            cleaned["items"] = _clean_anthropic_strict_schema(value)
        elif key in ("anyOf", "oneOf", "allOf") and isinstance(value, list):
            branches = [_clean_anthropic_strict_schema(v) if isinstance(v, dict) else v for v in value]
            if key in ("anyOf", "oneOf"):
                # Drop the ``{"type": "null"}`` branch — nullability is conveyed by
                # leaving the field out of ``required`` (Anthropic's native shape).
                non_null = [b for b in branches if not (isinstance(b, dict) and b.get("type") == "null")]
                branches = non_null or branches
            cleaned[key] = branches
        else:
            cleaned[key] = value

    # Inline a single-branch anyOf/oneOf left over after dropping the null branch, so
    # Anthropic sees a plain typed schema (e.g. a nullable enum) instead of a 1-item union.
    for union_key in ("anyOf", "oneOf"):
        branches = cleaned.get(union_key)
        if isinstance(branches, list) and len(branches) == 1 and isinstance(branches[0], dict):
            del cleaned[union_key]
            for k, v in branches[0].items():
                cleaned.setdefault(k, v)

    cleaned_type = cleaned.get("type")
    if cleaned_type == "object" or (isinstance(cleaned_type, list) and "object" in cleaned_type):
        cleaned["additionalProperties"] = False

    return cleaned


class AnthropicCacheControl(BaseModel):
    """Anthropic prompt caching configuration."""

    type: Literal["ephemeral"] = "ephemeral"
    ttl: Literal["5m", "1h"] | None = "5m"
    cache_injection_point_index: int = -2


class Anthropic(BaseLLM):
    """Anthropic LLM node.

    This class provides an implementation for the Anthropic Language Model node.

    Attributes:
        connection (AnthropicConnection | None): The connection to use for the Anthropic LLM.
        cache_control (AnthropicCacheControl | None): The cache control configuration.
        strict_tools: Inherited from :class:`BaseLLM`. False (default, or an empty
            list) ships every tool as-is with no strict guarantee; True cleans each
            tool's schema to Anthropic's strict subset and attaches ``strict: true``
            (up to :data:`ANTHROPIC_MAX_STRICT_TOOLS` per request); a list of tool
            (function) names makes only those tools strict and ships the rest
            untouched. Use a list to exclude tools whose schema exceeds Anthropic's
            strict grammar-compilation budget (the ``Schema is too complex for
            compilation`` 400).
    """

    connection: AnthropicConnection | None = None
    MODEL_PREFIX = "anthropic/"
    MAX_STRICT_TOOLS: ClassVar[int] = ANTHROPIC_MAX_STRICT_TOOLS
    cache_control: AnthropicCacheControl | None = None

    def __init__(self, **kwargs):
        """Initialize the Anthropic LLM node.

        Args:
            **kwargs: Additional keyword arguments.
        """
        if kwargs.get("client") is None and kwargs.get("connection") is None:
            kwargs["connection"] = AnthropicConnection()
        super().__init__(**kwargs)

    @staticmethod
    def _convert_non_image_to_file_content(messages: list[dict]) -> list[dict]:
        for message in messages:
            content = message.get("content")
            if not isinstance(content, list):
                continue

            new_content = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == VisionMessageType.IMAGE_URL and "image_url" in item:
                    url = item["image_url"].get("url", "")
                    if url.startswith("data:") and not url.startswith("data:image/"):
                        logger.debug("Anthropic: converting non-image image_url to file content format")
                        new_content.append(
                            {
                                "type": VisionMessageType.FILE,
                                "file": {"file_data": url},
                            }
                        )
                    else:
                        new_content.append(item)
                else:
                    new_content.append(item)

            message["content"] = new_content

        return messages

    def get_messages(self, prompt, input_data) -> list[dict]:
        """
        Format messages and convert non-image files to Anthropic file content format.
        """
        messages = super().get_messages(prompt, input_data)
        return self._convert_non_image_to_file_content(messages)

    def update_completion_params(self, params: dict[str, Any]) -> dict[str, Any]:
        """Attach Anthropic prompt caching configuration to completion params."""
        params = super().update_completion_params(params)
        if self.cache_control:
            params.setdefault("cache_control_injection_points", []).append(
                {
                    "location": "message",
                    "index": self.cache_control.cache_injection_point_index,
                    "control": self.cache_control.model_dump(
                        exclude_none=True,
                        exclude={"cache_injection_point_index"},
                    ),
                }
            )
        return params

    def _to_strict_function(self, fn: dict) -> dict:
        """Clean one tool's schema to Anthropic's strict shape and attach ``strict``.

        Cleans the parameter schema to Anthropic's strict shape (optionality via
        ``required`` omission, free-form objects → JSON-string fields,
        ``additionalProperties: false``) and attaches ``strict: true``. A function
        without a dict ``parameters`` is returned unchanged (nothing to make strict).

        See :meth:`BaseLLM.transform_tool_schemas` for the shared gating, whitelist,
        per-request cap (:attr:`MAX_STRICT_TOOLS`), and fail-safe fallback that drive
        this hook.
        """
        out = dict(fn)
        parameters = out.get("parameters")
        if not isinstance(parameters, dict):
            return out
        out["parameters"] = _clean_anthropic_strict_schema(parameters)
        out["strict"] = True
        return out
