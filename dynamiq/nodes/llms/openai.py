import enum
from functools import cached_property
from typing import Any, ClassVar

from dynamiq.connections import OpenAI as OpenAIConnection
from dynamiq.nodes.llms.base import BaseLLM
from dynamiq.utils.logger import logger


def _to_openai_strict_property(prop: Any) -> Any:
    """Recursively convert a property schema into OpenAI strict-mode form.

    - ``anyOf`` of primitives + null is flattened to a type array.
    - ``anyOf`` of complex shapes is preserved; each branch is converted recursively.
    - Nested objects get ``additionalProperties: false`` and every property listed in ``required``.
    - Arrays' ``items`` are converted recursively.
    - ``default`` is preserved (OpenAI tolerates it on primitives).

    Adapted from Letta's helpers._convert_to_structured_output_helper (Apache 2.0).
    """
    if not isinstance(prop, dict):
        return prop

    if "anyOf" in prop and "type" not in prop:
        primitive_types: list[str] = []
        has_complex = False
        for option in prop["anyOf"]:
            if isinstance(option, dict) and "type" in option:
                opt_type = option["type"]
                if opt_type in ("object", "array"):
                    has_complex = True
                    break
                if isinstance(opt_type, list):
                    has_complex = True
                    break
                primitive_types.append(opt_type)
            else:
                has_complex = True
                break

        if not has_complex and primitive_types:
            new_prop: dict[str, Any] = {"type": primitive_types}
            for key in ("description", "default", "enum", "title"):
                if key in prop:
                    new_prop[key] = prop[key]
            return new_prop

        return {
            "anyOf": [_to_openai_strict_property(opt) for opt in prop["anyOf"]],
            **{k: v for k, v in prop.items() if k in ("description", "title", "default")},
        }

    if "type" not in prop:
        return prop

    param_type = prop["type"]

    if isinstance(param_type, list):
        new_prop = {"type": param_type}
        for key in ("description", "default", "enum", "title"):
            if key in prop:
                new_prop[key] = prop[key]
        return new_prop

    if param_type == "object":
        nested_props = prop.get("properties", {})
        result: dict[str, Any] = {
            "type": "object",
            "properties": {k: _to_openai_strict_property(v) for k, v in nested_props.items()},
            "required": list(nested_props.keys()),
            "additionalProperties": False,
        }
        if "description" in prop:
            result["description"] = prop["description"]
        if "title" in prop:
            result["title"] = prop["title"]
        return result

    if param_type == "array":
        items = prop.get("items", {"type": "string"})
        result = {"type": "array", "items": _to_openai_strict_property(items)}
        if "description" in prop:
            result["description"] = prop["description"]
        if "title" in prop:
            result["title"] = prop["title"]
        return result

    new_prop = {"type": param_type}
    for key in ("description", "default", "enum", "title", "format"):
        if key in prop:
            new_prop[key] = prop[key]
    return new_prop


def _to_openai_strict_function(fn: dict) -> dict:
    """Convert a function-tool definition into OpenAI strict-mode form.

    Every top-level property becomes required (with optionality re-encoded via
    ``["x", "null"]`` type arrays), and ``additionalProperties: false`` is set
    at every object level. Sets ``strict: true``.

    If the input schema can't be safely converted, returns the original
    function dict unchanged (caller logs a warning).
    """
    out = dict(fn)
    parameters = out.get("parameters")
    if not isinstance(parameters, dict):
        return out

    properties = parameters.get("properties", {})
    converted_props = {k: _to_openai_strict_property(v) for k, v in properties.items()}
    new_parameters = {
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
    """
    connection: OpenAIConnection | None = None
    reasoning_effort: ReasoningEffort | None = ReasoningEffort.AUTO
    verbosity: Verbosity | None = Verbosity.MEDIUM
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
        """Convert each tool's parameter schema into OpenAI strict-mode form.

        Optional fields are re-expressed as nullable-required so strict mode
        applies broadly instead of being dropped whenever a tool has optionals.
        On any conversion failure, the original tool is kept (without strict)
        so the request still goes through.
        """
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
