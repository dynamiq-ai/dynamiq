from typing import Any, Literal

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

# JSON Schema keywords Anthropic's strict-mode compiler rejects. Strip them
# before sending; keep semantics in the description.
_ANTHROPIC_STRICT_UNSUPPORTED_KEYWORDS = frozenset(
    {
        "minimum",
        "maximum",
        "exclusiveMinimum",
        "exclusiveMaximum",
        "multipleOf",
        "minLength",
        "maxLength",
        "uniqueItems",
        "minContains",
        "maxContains",
        "minProperties",
        "maxProperties",
        "patternProperties",
        "unevaluatedProperties",
        "contains",
    }
)

_ANTHROPIC_STRICT_FORMAT_ALLOWED = frozenset(
    {"date-time", "time", "date", "duration", "email", "hostname", "uri", "ipv4", "ipv6", "uuid"}
)


def _clean_anthropic_strict_schema(schema: Any) -> Any:
    """Recursively clean a schema for Anthropic's strict tool-use mode.

    - Strips keywords Anthropic's compiler rejects (numeric/length bounds,
      ``patternProperties``, etc. — see ``_ANTHROPIC_STRICT_UNSUPPORTED_KEYWORDS``).
    - Clamps ``minItems`` to ``0`` or ``1`` (only values Anthropic supports).
    - Strips ``format`` values outside Anthropic's allowed set.
    - Forces ``additionalProperties: false`` on every object.
    - Optional fields stay omitted from ``required`` (Anthropic's native shape;
      no null-union trick).
    """
    if not isinstance(schema, dict):
        return schema

    cleaned: dict = {}
    for key, value in schema.items():
        if key in _ANTHROPIC_STRICT_UNSUPPORTED_KEYWORDS:
            continue
        if key == "minItems" and isinstance(value, int) and value > 1:
            continue
        if key == "format" and value not in _ANTHROPIC_STRICT_FORMAT_ALLOWED:
            continue
        if key == "properties" and isinstance(value, dict):
            cleaned["properties"] = {k: _clean_anthropic_strict_schema(v) for k, v in value.items()}
        elif key == "items" and isinstance(value, dict):
            cleaned["items"] = _clean_anthropic_strict_schema(value)
        elif key in ("anyOf", "oneOf", "allOf") and isinstance(value, list):
            cleaned[key] = [_clean_anthropic_strict_schema(v) if isinstance(v, dict) else v for v in value]
        else:
            cleaned[key] = value

    if cleaned.get("type") == "object":
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
        strict_tools: When True (default), attach ``strict: true`` to each tool's
            function entry (up to :data:`ANTHROPIC_MAX_STRICT_TOOLS` per request)
            and clean each schema to Anthropic's strict subset. Set False to ship
            tools as-is with no strict guarantee — the model still tries to
            follow the schema but conformance isn't decoder-enforced.
    """

    connection: AnthropicConnection | None = None
    MODEL_PREFIX = "anthropic/"
    cache_control: AnthropicCacheControl | None = None
    strict_tools: bool = True

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

    def transform_tool_schemas(self, tools: list[dict]) -> list[dict]:
        """Clean each tool's schema for Anthropic and attach ``strict: true``.

        Skip strict (and skip cleaning) when ``self.strict_tools`` is False, or
        once the request has reached the per-request cap of
        :data:`ANTHROPIC_MAX_STRICT_TOOLS` strict tools.
        """
        if not self.strict_tools:
            return tools

        out: list[dict] = []
        strict_count = 0
        for tool in tools:
            if not isinstance(tool, dict):
                out.append(tool)
                continue
            tool = dict(tool)
            fn = tool.get("function")
            if isinstance(fn, dict) and strict_count < ANTHROPIC_MAX_STRICT_TOOLS:
                fn = dict(fn)
                parameters = fn.get("parameters")
                if isinstance(parameters, dict):
                    fn["parameters"] = _clean_anthropic_strict_schema(parameters)
                fn["strict"] = True
                tool["function"] = fn
                strict_count += 1
            out.append(tool)
        return out
