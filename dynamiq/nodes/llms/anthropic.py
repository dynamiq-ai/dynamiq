from typing import Any, Literal

from pydantic import BaseModel, Field

from dynamiq.connections import Anthropic as AnthropicConnection
from dynamiq.nodes.llms.base import BaseLLM
from dynamiq.prompts.prompts import VisionMessageType
from dynamiq.utils.logger import logger

ANTHROPIC_MAX_STRICT_TOOLS = 15


def _clean_anthropic_schema(schema: Any, add_additional_properties_false: bool = False) -> Any:
    """Lightweight schema cleanup for Anthropic's tool_use input_schema.

    Removes / simplifies fields that commonly cause Anthropic tool schema issues:
    - Removes ``default`` values.
    - Simplifies nullable unions like ``{"type": ["null", "string"]}`` -> ``{"type": "string"}``
      (Anthropic prefers absence-via-required-subset over null values).
    - Recurses through ``properties`` / ``items`` / ``anyOf`` / ``oneOf`` / ``allOf``.
    - Optionally enforces ``additionalProperties: False`` on object types (required
      by Anthropic's structured-outputs beta).

    Adapted from Letta's anthropic_client._clean_property_schema (Apache 2.0).
    """
    if not isinstance(schema, dict):
        return schema

    cleaned: dict = {}

    if "type" in schema:
        t = schema.get("type")
        if isinstance(t, list):
            non_null = [x for x in t if x != "null"]
            if len(non_null) == 1:
                cleaned["type"] = non_null[0]
            elif len(non_null) > 1:
                cleaned["type"] = non_null
            else:
                cleaned["type"] = "string"
        else:
            cleaned["type"] = t

    for key, value in schema.items():
        if key in ("type", "default"):
            continue
        if key == "properties" and isinstance(value, dict):
            cleaned["properties"] = {
                k: _clean_anthropic_schema(v, add_additional_properties_false) for k, v in value.items()
            }
        elif key == "items" and isinstance(value, dict):
            cleaned["items"] = _clean_anthropic_schema(value, add_additional_properties_false)
        elif key in ("anyOf", "oneOf", "allOf") and isinstance(value, list):
            cleaned[key] = [
                _clean_anthropic_schema(v, add_additional_properties_false) if isinstance(v, dict) else v for v in value
            ]
        elif key == "additionalProperties" and isinstance(value, dict):
            cleaned[key] = _clean_anthropic_schema(value, add_additional_properties_false)
        else:
            cleaned[key] = value

    if add_additional_properties_false and cleaned.get("type") == "object":
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
        force_tool_choice: When True (default), force the model to call a tool
            whenever tools are present in function-calling mode. Translates
            ``tool_choice`` ``None``/``"auto"``/``"required"`` into Anthropic's
            ``{"type": "any", "disable_parallel_tool_use": true}`` shape so the
            model cannot bail out with a text-only response.
        strict_allowlist: Tool names eligible for Anthropic's structured-outputs
            beta. Tools on this list get ``"strict": true`` attached and the
            ``anthropic-beta: structured-outputs-2025-11-13`` header is added.
            Capped at ``ANTHROPIC_MAX_STRICT_TOOLS`` per request.
    """
    connection: AnthropicConnection | None = None
    MODEL_PREFIX = "anthropic/"
    cache_control: AnthropicCacheControl | None = None
    force_tool_choice: bool = True
    strict_allowlist: set[str] = Field(default_factory=set)

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
        """Attach Anthropic prompt caching configuration to completion params.

        Strict tool use is GA on Anthropic and engaged by attaching
        ``strict: true`` directly to a tool's function entry — no beta header
        required. See ``transform_tool_schemas``.
        """
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
        """Clean schemas for Anthropic and attach strict flag for allowlisted tools.

        - Strips ``default``, simplifies ``["null", "x"]`` -> ``"x"``, recurses
          into nested shapes.
        - For tools in ``strict_allowlist`` (capped at 15), attaches
          ``strict: true``; the beta header is added in ``update_completion_params``.
        """
        out: list[dict] = []
        strict_count = 0
        for tool in tools:
            if not isinstance(tool, dict):
                out.append(tool)
                continue
            tool = dict(tool)
            fn = tool.get("function")
            if isinstance(fn, dict):
                fn = dict(fn)
                tool_name = fn.get("name", "")
                use_strict = tool_name in self.strict_allowlist and strict_count < ANTHROPIC_MAX_STRICT_TOOLS
                parameters = fn.get("parameters")
                if isinstance(parameters, dict):
                    fn["parameters"] = _clean_anthropic_schema(parameters, add_additional_properties_false=use_strict)
                if use_strict:
                    fn["strict"] = True
                    strict_count += 1
                else:
                    fn.pop("strict", None)
                tool["function"] = fn
            out.append(tool)
        return out

    def transform_tool_choice(self, tool_choice: Any, tools: list[dict] | None) -> Any:
        """Translate generic tool_choice into Anthropic's native shape.

        Forces a tool call when ``force_tool_choice`` is enabled (the default),
        which prevents the "model emits only thought, no action" failure mode.
        """
        if not tools:
            return tool_choice
        if isinstance(tool_choice, dict):
            return tool_choice

        base = {"disable_parallel_tool_use": True}
        if tool_choice in (None, "auto"):
            if self.force_tool_choice:
                return {"type": "any", **base}
            return {"type": "auto", **base}
        if tool_choice == "required":
            return {"type": "any", **base}
        if tool_choice == "none":
            return {"type": "auto", **base}
        if isinstance(tool_choice, str):
            return {"type": "tool", "name": tool_choice, **base}
        return tool_choice
