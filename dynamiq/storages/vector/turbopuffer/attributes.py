"""Conversion of document metadata into Turbopuffer attribute values.

Turbopuffer stores flat attributes only. A value's type is fixed by the namespace schema, arrays
must hold a single type without nulls, and a dict of numbers would be read as a sparse vector. The
helpers here turn arbitrary metadata into values that satisfy those rules.
"""

import json
import math
from datetime import date, datetime
from decimal import Decimal
from enum import Enum
from typing import Any

MAX_FILTERABLE_BYTES = 4096
MAX_INT = 2**63 - 1
MIN_INT = -(2**63)

STRING = "string"
INT = "int"
UINT = "uint"
FLOAT = "float"
BOOL = "bool"
DATETIME = "datetime"
UUID = "uuid"

_SCALAR_TYPES = {STRING, INT, UINT, FLOAT, BOOL, DATETIME, UUID}


def is_array_type(attribute_type: str | None) -> bool:
    return bool(attribute_type) and attribute_type.startswith("[]")


def element_type(attribute_type: str) -> str:
    return attribute_type[2:]


def _json_dumps(value: Any) -> str:
    return json.dumps(value, default=str, ensure_ascii=False)


def _scalar(value: Any) -> Any:
    """Return a plain Python scalar for value, or None when it has no usable value.

    Subclasses of str, int and float (as returned by PDF parsers, for example) become their base
    type, so they serialize like ordinary values.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, Enum):
        return _scalar(value.value)
    if isinstance(value, int):
        value = int(value)
        return value if MIN_INT <= value <= MAX_INT else str(value)
    if isinstance(value, float):
        value = float(value)
        return value if math.isfinite(value) else None
    if isinstance(value, Decimal):
        return _scalar(float(value))
    if isinstance(value, str):
        return str(value)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if hasattr(value, "item") and callable(value.item):
        try:
            return _scalar(value.item())
        except (TypeError, ValueError):
            pass
    return str(value)


def normalize_value(value: Any) -> Any:
    """Turn a metadata value into a flat value Turbopuffer accepts.

    Dicts and lists that contain dicts or lists become JSON strings. Lists drop None elements and,
    when their elements differ in type, become lists of strings, with ints and floats becoming
    floats. Returns None when the value should not be stored.
    """
    if isinstance(value, dict):
        return _json_dumps(value)
    if isinstance(value, (list, tuple, set)):
        items = list(value)
        if any(isinstance(item, (dict, list, tuple, set)) for item in items):
            return _json_dumps(items)
        scalars = [s for s in (_scalar(item) for item in items) if s is not None]
        if not scalars:
            return []
        if all(isinstance(s, bool) for s in scalars):
            return scalars
        if all(isinstance(s, str) for s in scalars):
            return scalars
        if all(isinstance(s, int) and not isinstance(s, bool) for s in scalars):
            return scalars
        if all(isinstance(s, (int, float)) and not isinstance(s, bool) for s in scalars):
            return [float(s) for s in scalars]
        return [to_string(s) for s in scalars]
    return _scalar(value)


def infer_type(value: Any) -> str:
    """Return the Turbopuffer type for a normalized value.

    An empty list is typed as a list of strings, the most common list in document metadata, so an
    empty access list still declares a filterable attribute.
    """
    if isinstance(value, list):
        if not value:
            return f"[]{STRING}"
        return f"[]{infer_type(value[0])}"
    if isinstance(value, bool):
        return BOOL
    if isinstance(value, int):
        return INT
    if isinstance(value, float):
        return FLOAT
    return STRING


def to_string(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return value
    if isinstance(value, (list, dict)):
        return _json_dumps(value)
    return str(value)


def _to_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        try:
            return int(value.strip())
        except ValueError:
            return None
    return None


def _to_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            number = float(value.strip())
        except ValueError:
            return None
        return number if math.isfinite(number) else None
    return None


def _to_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.strip().lower() in {"true", "false"}:
        return value.strip().lower() == "true"
    return None


def coerce_scalar(value: Any, attribute_type: str) -> Any:
    """Convert a scalar to the given Turbopuffer type, or return None when it cannot be converted."""
    if value is None:
        return None
    if attribute_type == STRING:
        return to_string(value)
    if attribute_type == INT:
        return _to_int(value)
    if attribute_type == UINT:
        number = _to_int(value)
        return number if number is not None and number >= 0 else None
    if attribute_type == FLOAT:
        return _to_float(value)
    if attribute_type == BOOL:
        return _to_bool(value)
    if attribute_type in (DATETIME, UUID):
        return value if isinstance(value, str) else None
    return None


def coerce_value(value: Any, attribute_type: str) -> Any:
    """Convert a normalized value to the given Turbopuffer type.

    Returns None when the value cannot be stored under that type, so the caller can leave the
    attribute out instead of failing the whole write.
    """
    if attribute_type == "[]unknown":
        # An attribute first written as an empty list has no element type yet, and the first
        # non-empty list sets it.
        return value if isinstance(value, list) else None
    if is_array_type(attribute_type):
        scalar_type = element_type(attribute_type)
        if scalar_type not in _SCALAR_TYPES:
            return None
        items = value if isinstance(value, list) else [value]
        coerced = [coerce_scalar(item, scalar_type) for item in items]
        return [item for item in coerced if item is not None]
    if attribute_type not in _SCALAR_TYPES:
        return None
    if isinstance(value, list):
        return to_string(value) if attribute_type == STRING else None
    return coerce_scalar(value, attribute_type)


def longest_string_bytes(value: Any) -> int:
    """Return the UTF-8 size of the longest string in a value, or of the value itself."""
    if isinstance(value, str):
        return len(value.encode("utf-8"))
    if isinstance(value, list):
        return max((longest_string_bytes(item) for item in value), default=0)
    return 0
