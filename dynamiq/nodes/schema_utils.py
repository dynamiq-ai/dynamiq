"""Dependency-neutral helpers for transforming node input schemas.

Kept free of any ``Node``/agent imports so both :mod:`dynamiq.nodes.node` and the
agent schema generator can import it without a circular dependency.
"""

import copy
from typing import Any

from pydantic import BaseModel, create_model
from pydantic_core import PydanticUndefined

from dynamiq.nodes.types import InputParamMode


def apply_param_modes(schema: type[BaseModel], param_modes: dict[str, InputParamMode]) -> type[BaseModel]:
    """Return a copy of ``schema`` with per-field agent exposure tuned.

    The override rewrites the Pydantic model itself (not a parallel config) so the
    LLM-facing tool schema and the execution-time validation stay consistent — both
    are derived from the returned model. Both modes act on optional fields only.

    Modes (keyed by ``input_schema`` field name):
        * ``required`` -- an optional field becomes required (its default is dropped),
          so the agent must always supply it.
        * ``hidden``   -- the field is marked ``is_accessible_to_agent=False`` so it is
          omitted from the agent-facing schema; the field's own default is used at
          execution time. A required field cannot be hidden (it would have no value).

    Args:
        schema: The tool's input schema model.
        param_modes: Mapping of field name to desired mode. Empty leaves the schema
            untouched (current behavior).

    Returns:
        A new model subclassing ``schema`` with the requested fields adjusted, or
        ``schema`` unchanged when ``param_modes`` is empty.

    Raises:
        ValueError: if a key is not a field of ``schema``, a mode is invalid, a
            required field is requested to be hidden, or a field that is already hidden
            from the agent (``is_accessible_to_agent=False``) is requested to be required.
    """
    if not param_modes:
        return schema

    unknown = set(param_modes) - set(schema.model_fields)
    if unknown:
        raise ValueError(f"Unknown field(s) in param_modes for {schema.__name__}: {sorted(unknown)}")

    field_overrides: dict[str, Any] = {}
    for name, mode in param_modes.items():
        try:
            mode = InputParamMode(mode)
        except ValueError:
            raise ValueError(f"Invalid mode {mode!r} for field {name!r}; expected 'required' or 'hidden'.")

        field = copy.deepcopy(schema.model_fields[name])

        if mode == InputParamMode.REQUIRED:
            if field.json_schema_extra and field.json_schema_extra.get("is_accessible_to_agent", True) is False:
                raise ValueError(
                    f"Field {name!r} is not exposed to the agent (is_accessible_to_agent=False) and cannot be "
                    "made required; the agent could never supply it, so validation would always fail."
                )
            field.default = PydanticUndefined
            field.default_factory = None
        elif mode == InputParamMode.HIDDEN:
            if field.is_required():
                raise ValueError(
                    f"Field {name!r} is required and cannot be hidden (it would have no value at execution); "
                    "give it a default first."
                )
            extra = dict(field.json_schema_extra or {})
            extra["is_accessible_to_agent"] = False
            field.json_schema_extra = extra

        field_overrides[name] = (field.annotation, field)

    return create_model(
        schema.__name__,
        __base__=schema,
        __module__=schema.__module__,
        **field_overrides,
    )
