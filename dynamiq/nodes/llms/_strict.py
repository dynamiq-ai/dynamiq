"""Shared strict tool-calling schema transforms and provider mixins.

Two strict "shapes" are supported, both producing tighter, valid JSON Schemas
that constrain a model's tool arguments:

- **OpenAI structured-outputs strict** (:func:`to_openai_strict_function`, via
  :class:`OpenAIStrictToolsMixin`): every property promoted to ``required`` with
  optionals re-encoded as nullable type-unions (``["string", "null"]``),
  ``additionalProperties: false`` on every object. Used by OpenAI and Cerebras,
  which accept the ``strict`` flag and tightened schema and forward them through
  LiteLLM unchanged.

- **Required-omission strict subset** (:func:`to_strict_subset_function`, via
  :class:`SubsetStrictToolsNoFlagMixin`): optionality expressed by omitting
  fields from ``required`` (no null-unions), free-form objects encoded as JSON
  strings, ``additionalProperties: false`` on closed objects, ``null`` stripped
  from type unions. Used by Cohere (whose API rejects an unknown ``strict`` field,
  so the flag is omitted and strictness is enabled request-level instead).
  Anthropic uses the same subset shape but keeps its own copy of the transform in
  ``anthropic.py`` (it attaches ``strict`` and relies on a LiteLLM patch).

In both shapes optional fields the model leaves unset come back as ``null`` (or
absent); the agent strips those before tool validation so Python defaults apply
(see ``_normalize_fields``). Free-form ``dict[str, Any]`` fields arrive as
JSON-encoded strings and are parsed back the same way.

Providers wire one of these in by mixing the relevant mixin into their class. The
shared gating, whitelist, per-request cap, and fail-safe fallback live in
:meth:`BaseLLM.transform_tool_schemas`.
"""

from typing import Any

# ---------------------------------------------------------------------------
# OpenAI structured-outputs strict shape (nullable-union optionality)
# ---------------------------------------------------------------------------


def add_null_to_type(prop: Any) -> Any:
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


def to_openai_strict_property(prop: Any) -> Any:
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
            "anyOf": [to_openai_strict_property(opt) for opt in prop["anyOf"]],
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
            cv = to_openai_strict_property(v)
            if k not in nested_required:
                cv = add_null_to_type(cv)
            converted_nested[k] = cv
        cleaned["properties"] = converted_nested
        cleaned["required"] = list(nested.keys())
        cleaned["additionalProperties"] = False
        return cleaned
    if is_array and isinstance(cleaned.get("items"), dict):
        cleaned["items"] = to_openai_strict_property(cleaned["items"])
        return cleaned
    return cleaned


def to_openai_strict_function(fn: dict) -> dict:
    """Convert one tool's schema into OpenAI structured-outputs strict form.

    Every property is promoted to ``required`` and ``additionalProperties: false``
    is set at every object level. Fields that have a default (NOT in the original
    ``required``) — or that are already nullable — are made nullable so the model
    can emit ``null`` to leave them at their default; the agent then strips those
    nulls before tool validation so the Python default applies. Genuinely-required
    fields (no default) stay non-nullable and must be emitted. ``strict: true``
    is attached. Free-form objects (``dict[str, Any]``) are handled inside
    :func:`to_openai_strict_property` (converted to JSON-string fields).

    A function without a dict ``parameters`` is returned unchanged.
    """
    out = dict(fn)
    parameters = out.get("parameters")
    if not isinstance(parameters, dict):
        return out

    properties = parameters.get("properties", {})
    original_required = set(parameters.get("required", []))
    converted_props: dict[str, Any] = {}
    for name, prop in properties.items():
        converted = to_openai_strict_property(prop)
        if name not in original_required:
            converted = add_null_to_type(converted)
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


# ---------------------------------------------------------------------------
# Required-omission strict subset (Anthropic + native-schema providers)
# ---------------------------------------------------------------------------


def clean_strict_subset_schema(schema: Any) -> Any:
    """Recursively clean a schema to the required-omission strict subset.

    - Forces ``additionalProperties: false`` on every object that declares
      ``properties``.
    - Free-form objects (``dict[str, Any]`` → ``{"type": "object"}`` with no
      ``properties``) are converted to JSON-encoded string fields, since strict
      mode can't express an open object. The agent parses them back to dicts
      before tool validation (see ``_normalize_fields``).
    - Optional fields stay omitted from ``required`` (native optionality shape;
      no null-union trick), and ``null`` is stripped from type unions.
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
            # A null default conveys optionality, which is expressed via
            # ``required`` omission. Drop it so it can't clash with a now non-null type.
            continue
        if key == "type" and isinstance(value, list):
            # Optionality is conveyed by omitting the field from ``required``,
            # not via a null-union. Strip ``null`` so a nullable scalar/enum keeps a
            # single declared type (e.g. ``["string", "null"]`` -> ``"string"``);
            # native schema dialects reject an enum whose declared type is ``["string", "null"]``.
            non_null = [t for t in value if t != "null"]
            cleaned["type"] = non_null[0] if len(non_null) == 1 else (non_null or value)
        elif key == "properties" and isinstance(value, dict):
            cleaned["properties"] = {k: clean_strict_subset_schema(v) for k, v in value.items()}
        elif key == "items" and isinstance(value, dict):
            cleaned["items"] = clean_strict_subset_schema(value)
        elif key in ("anyOf", "oneOf", "allOf") and isinstance(value, list):
            branches = [clean_strict_subset_schema(v) if isinstance(v, dict) else v for v in value]
            if key in ("anyOf", "oneOf"):
                # Drop the ``{"type": "null"}`` branch — nullability is conveyed by
                # leaving the field out of ``required`` (native optionality shape).
                non_null = [b for b in branches if not (isinstance(b, dict) and b.get("type") == "null")]
                branches = non_null or branches
            cleaned[key] = branches
        else:
            cleaned[key] = value

    # Inline a single-branch anyOf/oneOf left over after dropping the null branch, so
    # the provider sees a plain typed schema (e.g. a nullable enum) instead of a 1-item union.
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


def to_strict_subset_function(fn: dict, *, attach_flag: bool = True) -> dict:
    """Clean one tool's schema to the required-omission strict subset.

    Cleans the parameter schema (optionality via ``required`` omission, free-form
    objects → JSON-string fields, ``additionalProperties: false``). When
    ``attach_flag`` is true (default) a ``strict: true`` field is attached for
    providers that honor it; set it false for providers whose API rejects an
    unknown ``strict`` field on tools (e.g. Cohere) — they still get the tighter
    schema. A function without a dict ``parameters`` is returned unchanged.
    """
    out = dict(fn)
    parameters = out.get("parameters")
    if not isinstance(parameters, dict):
        return out
    out["parameters"] = clean_strict_subset_schema(parameters)
    if attach_flag:
        out["strict"] = True
    return out


# ---------------------------------------------------------------------------
# Provider mixins — wire one strict shape into a provider via its `_to_strict_function` hook
# ---------------------------------------------------------------------------


class OpenAIStrictToolsMixin:
    """Mixin: convert tools to the OpenAI structured-outputs strict shape.

    For OpenAI and OpenAI-compatible chat-completions providers that forward the
    ``strict`` flag and the tightened schema through LiteLLM unchanged. See
    :meth:`BaseLLM.transform_tool_schemas` for the shared gating/whitelist/cap.
    """

    def _to_strict_function(self, fn: dict) -> dict:
        return to_openai_strict_function(fn)


class SubsetStrictToolsNoFlagMixin:
    """Mixin: tighten tools to the required-omission strict subset, no ``strict`` flag.

    For providers whose API hard-rejects an unknown ``strict`` field on tools
    (Cohere returns a 400 ``unknown field: parameter 'strict' is not a valid
    field``). The schema is still tightened (enum/closed-object/optionality
    constraints), so the model is steered the same way — only the flag is omitted.
    """

    def _to_strict_function(self, fn: dict) -> dict:
        return to_strict_subset_function(fn, attach_flag=False)
