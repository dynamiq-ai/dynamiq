import enum
from functools import cached_property
from typing import Any, ClassVar

from dynamiq.connections import OpenAI as OpenAIConnection
from dynamiq.nodes.llms.base import BaseLLM
from dynamiq.utils.logger import logger

# Keywords OpenAI structured outputs / strict function calling do not support.
# See https://platform.openai.com/docs/guides/structured-outputs#supported-schemas
_OPENAI_STRICT_UNSUPPORTED_KEYWORDS = frozenset(
    {
        "minLength",
        "maxLength",
        "minimum",
        "maximum",
        "exclusiveMinimum",
        "exclusiveMaximum",
        "multipleOf",
        "patternProperties",
        "unevaluatedProperties",
        "contains",
        "minContains",
        "maxContains",
        "minProperties",
        "maxProperties",
        "uniqueItems",
    }
)


def _to_openai_strict_property(prop: Any) -> Any:
    """Convert a property schema into OpenAI strict-mode form.

    - ``anyOf`` of ``[primitive, null]`` is flattened to a type array ``["X", "null"]``.
    - Nested objects get ``additionalProperties: false`` and every property in ``required``.
    - Arrays' ``items`` are converted recursively.
    - Unsupported keywords are stripped.
    """
    if not isinstance(prop, dict):
        return prop

    if "anyOf" in prop and "type" not in prop:
        primitive_types: list[str] = []
        has_complex = False
        for option in prop["anyOf"]:
            if isinstance(option, dict) and "type" in option:
                opt_type = option["type"]
                if opt_type in ("object", "array") or isinstance(opt_type, list):
                    has_complex = True
                    break
                primitive_types.append(opt_type)
            else:
                has_complex = True
                break
        if not has_complex and primitive_types:
            out: dict[str, Any] = {"type": primitive_types}
            for key in ("description", "default", "enum", "title"):
                if key in prop:
                    out[key] = prop[key]
            return out
        return {
            "anyOf": [_to_openai_strict_property(opt) for opt in prop["anyOf"]],
            **{k: v for k, v in prop.items() if k in ("description", "title", "default")},
        }

    cleaned: dict = {}
    for key, value in prop.items():
        if key in _OPENAI_STRICT_UNSUPPORTED_KEYWORDS:
            continue
        cleaned[key] = value

    param_type = cleaned.get("type")
    if param_type == "object":
        nested = cleaned.get("properties", {})
        cleaned["properties"] = {k: _to_openai_strict_property(v) for k, v in nested.items()}
        cleaned["required"] = list(nested.keys())
        cleaned["additionalProperties"] = False
        return cleaned
    if param_type == "array" and isinstance(cleaned.get("items"), dict):
        cleaned["items"] = _to_openai_strict_property(cleaned["items"])
        return cleaned
    return cleaned


def _to_openai_strict_function(fn: dict) -> dict:
    """Convert a function-tool definition into OpenAI strict-mode form.

    Top-level properties all become required; optional fields are re-encoded as
    nullable type arrays. ``additionalProperties: false`` is set at every object
    level. ``strict: true`` is attached.
    """
    out = dict(fn)
    parameters = out.get("parameters")
    if not isinstance(parameters, dict):
        return out

    properties = parameters.get("properties", {})
    original_required = set(parameters.get("required", []))
    converted_props: dict[str, Any] = {}
    for name, prop in properties.items():
        converted = _to_openai_strict_property(prop)
        # Optional fields → add "null" to the type so they can be in required.
        if name not in original_required and isinstance(converted, dict):
            t = converted.get("type")
            if isinstance(t, str) and t != "null":
                converted["type"] = [t, "null"]
            elif isinstance(t, list) and "null" not in t:
                converted["type"] = [*t, "null"]
        converted_props[name] = converted

    new_parameters: dict[str, Any] = {
        "type": "object",
        "properties": converted_props,
        "required": list(converted_props.keys()),
        "additionalProperties": False,
    }
    for key in ("description", "title"):
        if key in parameters:
            new_parameters[key] = parameters[key]

    out["parameters"] = new_parameters
    out["strict"] = True
    return out


class ReasoningEffort(str, enum.Enum):
    """
    The reasoning effort to use for the OpenAI LLM.
    """

    AUTO = "auto"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    MINIMAL = "minimal"


# OpenAI models whose native default for 'reasoning_effort' is "none"
_MODELS_DEFAULTING_TO_NONE: frozenset[str] = frozenset(
    {
        "gpt-5.1",
        "gpt-5.1-2025-11-13",
        "gpt-5.2",
        "gpt-5.2-2025-12-11",
        "gpt-5.3-codex",
        "gpt-5.4",
        "gpt-5.4-2026-03-05",
        "gpt-5.4-mini",
        "gpt-5.4-mini-2026-03-17",
        "gpt-5.4-nano",
        "gpt-5.4-nano-2026-03-17",
    }
)


def _resolve_default_reasoning_effort(model_lower: str) -> ReasoningEffort | None:
    """Return the implicit default for ``model_lower``, or ``None`` to omit the param.

    ``model_lower`` is the model identifier with the ``openai/`` prefix stripped
    and lowercased. Models not on the explicit "defaults to none" list fall back
    to the historical default of ``MEDIUM``.
    """
    if model_lower in _MODELS_DEFAULTING_TO_NONE:
        return None
    return ReasoningEffort.MEDIUM


class Verbosity(str, enum.Enum):
    """
    The verbosity level for the OpenAI LLM.
    """

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class OpenAI(BaseLLM):
    """OpenAI LLM node.

    This class provides an implementation for the OpenAI Language Model node.

    Attributes:
        connection (OpenAIConnection | None): The connection to use for the OpenAI LLM.
        strict_tools: When True (default), convert each tool's parameter schema
            into OpenAI structured-outputs strict form (every property required,
            optionals re-encoded as nullable types, ``additionalProperties: false``
            on every object) and attach ``strict: true``. Set False to ship tools
            as-is.
    """
    connection: OpenAIConnection | None = None
    reasoning_effort: ReasoningEffort | None = ReasoningEffort.AUTO
    verbosity: Verbosity | None = Verbosity.MEDIUM
    strict_tools: bool = True
    O_SERIES_MODEL_PREFIXES: ClassVar[tuple[str, ...]] = ("o1", "o3", "o4")
    MODEL_PREFIX = "openai/"

    def __init__(self, **kwargs):
        """Initialize the OpenAI LLM node.

        Args:
            **kwargs: Additional keyword arguments.
        """
        if kwargs.get("client") is None and kwargs.get("connection") is None:
            kwargs["connection"] = OpenAIConnection()
        super().__init__(**kwargs)

    @cached_property
    def is_o_series_model(self) -> bool:
        """
        Determine if the model belongs to the O-series (e.g. o1 or o3, o4)
        by checking if the model starts with any of the O-series prefixes.
        """
        model_lower = self.model.lower().removeprefix(self.MODEL_PREFIX)
        return any(model_lower.startswith(prefix) for prefix in self.O_SERIES_MODEL_PREFIXES)

    def _effective_reasoning_effort(self, model_lower: str) -> ReasoningEffort | None:
        """Resolve reasoning_effor parameter for the model."""
        if self.reasoning_effort == ReasoningEffort.AUTO:
            return _resolve_default_reasoning_effort(model_lower)
        return self.reasoning_effort

    @staticmethod
    def _apply_reasoning_effort(params: dict[str, Any], effort: ReasoningEffort | None) -> None:
        """Set or remove ``reasoning_effort`` on ``params`` based on ``effort``."""
        if effort is None:
            params.pop("reasoning_effort", None)
        else:
            params["reasoning_effort"] = effort

    def transform_tool_schemas(self, tools: list[dict]) -> list[dict]:
        """Convert each tool's schema into OpenAI strict-mode form.

        Skipped when ``self.strict_tools`` is False. On per-tool conversion
        failure (e.g. an exotic schema the converter can't safely transform),
        the original tool is kept without ``strict`` so the request still
        succeeds.
        """
        if not self.strict_tools:
            return tools

        out: list[dict] = []
        for tool in tools:
            if not isinstance(tool, dict):
                out.append(tool)
                continue
            tool = dict(tool)
            fn = tool.get("function")
            if isinstance(fn, dict):
                try:
                    tool["function"] = _to_openai_strict_function(fn)
                except Exception as exc:
                    logger.warning(
                        "OpenAI strict conversion failed for tool %s: %s; keeping non-strict schema",
                        fn.get("name", "<unknown>"),
                        exc,
                    )
                    fn = dict(fn)
                    fn.pop("strict", None)
                    tool["function"] = fn
            out.append(tool)
        return out

    def update_completion_params(self, params: dict[str, Any]) -> dict[str, Any]:
        """
        Override the base method to update the completion parameters for OpenAI.
        For new series models, use "max_completion_tokens" instead of "max_tokens".
        """
        params = super().update_completion_params(params)

        new_params = params.copy()
        model_lower = self.model.lower().removeprefix(self.MODEL_PREFIX)
        if self.is_o_series_model:
            new_params["max_completion_tokens"] = self.max_tokens
            if model_lower.startswith("o3") or model_lower.startswith("o4"):
                self._apply_reasoning_effort(new_params, self._effective_reasoning_effort(model_lower))
            if model_lower not in ["o3-mini"]:
                new_params.pop("stop", None)
            new_params.pop("max_tokens", None)
            new_params.pop("temperature", None)
        elif model_lower.startswith("gpt-5"):
            if "chat" not in model_lower:
                new_params["verbosity"] = self.verbosity
                if "pro" in model_lower:
                    self._apply_reasoning_effort(new_params, ReasoningEffort.HIGH)
                else:
                    self._apply_reasoning_effort(new_params, self._effective_reasoning_effort(model_lower))
            new_params["max_completion_tokens"] = self.max_tokens
            new_params.pop("stop", None)
            new_params.pop("max_tokens", None)

        return new_params
