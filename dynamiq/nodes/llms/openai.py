import enum
from functools import cached_property
from typing import Any, ClassVar

from dynamiq.connections import OpenAI as OpenAIConnection
from dynamiq.nodes.llms.base import BaseLLM


def _add_null_to_type(prop: Any) -> Any:
    """Make a converted property nullable so it can sit in ``required`` while
    letting the model signal "leave it at the default" by emitting ``null``.

    Handles plain types (``"x"`` → ``["x", "null"]``), type arrays, and complex
    ``anyOf`` unions (adds a ``{"type": "null"}`` branch). No-op if already nullable
    or if it's a stringified free-form object.

    The input is never mutated: a shallow copy is returned when a change is needed,
    otherwise the original object is returned unchanged.
    """
    if not isinstance(prop, dict):
        return prop
    t = prop.get("type")
    if isinstance(t, str) and t != "null":
        return {**prop, "type": [t, "null"]}
    if isinstance(t, list) and "null" not in t:
        return {**prop, "type": [*t, "null"]}
    if "anyOf" in prop:
        branches = prop["anyOf"]
        if not any(isinstance(b, dict) and b.get("type") == "null" for b in branches):
            return {**prop, "anyOf": [*branches, {"type": "null"}]}
    return prop


def _to_openai_strict_property(prop: Any) -> Any:
    """Convert a property schema into OpenAI strict-mode form.

    - ``anyOf`` of ``[primitive, null]`` is flattened to a type array ``["X", "null"]``.
    - Nested objects get ``additionalProperties: false`` and every property in
      ``required``. Nested fields that have a default (i.e. were NOT in the
      object's own ``required``) are made nullable, so the model can emit ``null``
      to leave them at their default. The agent strips those nulls before tool
      validation so the Python default applies (see ``_normalize_fields``).
    - Arrays' ``items`` are converted recursively.
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

    cleaned: dict = dict(prop)

    param_type = cleaned.get("type")
    # `type` may be a plain string ("object") or a nullable type-array
    # (["object", "null"] for `dict | None`). Detect both.
    is_nullable_type = isinstance(param_type, list) and "null" in param_type
    is_object = param_type == "object" or (isinstance(param_type, list) and "object" in param_type)
    is_array = param_type == "array" or (isinstance(param_type, list) and "array" in param_type)

    if is_object:
        if "properties" not in cleaned:
            # Free-form object (dict[str, Any]). Strict can't express an open
            # object, so represent it as a JSON-encoded string. The agent parses
            # it back to a dict before tool validation (see _normalize_fields).
            # Preserve nullability so a `dict | None` field can still be null.
            desc = cleaned.get("description", "")
            return {
                "type": ["string", "null"] if is_nullable_type else "string",
                "description": (f"{desc} " if desc else "") + "Provide as a JSON-encoded object string.",
            }
        nested = cleaned["properties"]
        nested_required = set(cleaned.get("required", []))
        converted_nested: dict[str, Any] = {}
        for k, v in nested.items():
            cv = _to_openai_strict_property(v)
            if k not in nested_required:
                cv = _add_null_to_type(cv)
            converted_nested[k] = cv
        cleaned["properties"] = converted_nested
        cleaned["required"] = list(nested.keys())
        cleaned["additionalProperties"] = False
        return cleaned
    if is_array and isinstance(cleaned.get("items"), dict):
        cleaned["items"] = _to_openai_strict_property(cleaned["items"])
        return cleaned
    return cleaned


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
        strict_tools: Inherited from :class:`BaseLLM`. False (default, or an empty
            list) ships every tool as-is; True converts each tool's parameter schema
            into OpenAI structured-outputs strict form (every property required,
            optionals re-encoded as nullable types, ``additionalProperties: false``
            on every object) and attaches ``strict: true``; a list of tool (function)
            names converts only those tools and ships the rest untouched.
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

    def _to_strict_function(self, fn: dict) -> dict:
        """Convert one tool's schema into OpenAI structured-outputs strict form.

        Every property is promoted to ``required`` and ``additionalProperties: false``
        is set at every object level. Fields that have a default (NOT in the original
        ``required``) — or that are already nullable — are made nullable so the model
        can emit ``null`` to leave them at their default; the agent then strips those
        nulls before tool validation so the Python default applies. Genuinely-required
        fields (no default) stay non-nullable and must be emitted. ``strict: true``
        is attached. Free-form objects (``dict[str, Any]``) are handled inside
        ``_to_openai_strict_property`` (converted to JSON-string fields).

        See :meth:`BaseLLM.transform_tool_schemas` for the shared gating,
        whitelist, and fail-safe fallback that drive this hook.
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
            if name not in original_required:
                converted = _add_null_to_type(converted)
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
