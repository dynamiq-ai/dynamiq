"""Conversion of Dynamiq filters into Turbopuffer filters.

Turbopuffer rejects a filter on an attribute the namespace has never stored, compares an array
attribute only through Contains operators, and matches nulls with Lt and Lte. Given the namespace
schema, the converter evaluates conditions on missing attributes itself and picks the operator
that matches the attribute's type, so results match the other vector stores. A condition on an
attribute that holds values but is not filterable cannot be answered, so it raises instead.
"""

from typing import Any

from dynamiq.storages.vector.exceptions import VectorStoreFilterException
from dynamiq.storages.vector.utils import normalize_filters

from .attributes import coerce_value, element_type, is_array_type, normalize_value


class _Constant:
    """A filter that matches every document or none, before it is turned into a Turbopuffer filter."""

    def __init__(self, matches: bool):
        self.matches = matches

    def __repr__(self) -> str:
        return "MATCH_ALL" if self.matches else "MATCH_NONE"


MATCH_ALL = _Constant(True)
MATCH_NONE = _Constant(False)

# Every document has an id, so these match every document and none.
FILTER_ALL = ["id", "NotEq", None]
FILTER_NONE = ["id", "Eq", None]


def convert_filters(filters: dict[str, Any] | None, schema: dict[str, dict] | None = None) -> Any:
    """
    Convert filters from Dynamiq format to Turbopuffer format.

    Args:
        filters (dict[str, Any] | None): Filters in Dynamiq format.
        schema (dict[str, dict] | None): The namespace schema. Without it, every attribute is taken
            to exist and to hold a single value.

    Returns:
        A Turbopuffer filter, MATCH_ALL when the filters match every document, or MATCH_NONE when
        they match none.

    Raises:
        VectorStoreFilterException: If the filters are malformed or use an attribute that is not
            filterable.
    """
    if filters is None:
        return MATCH_ALL
    if not isinstance(filters, dict):
        raise VectorStoreFilterException("Filters must be a dictionary")

    normalized = normalize_filters(filters)
    if not normalized:
        return MATCH_ALL

    return _convert(normalized, schema)


def to_turbopuffer_filter(converted: Any) -> list:
    """Return a Turbopuffer filter for a converted filter, including the constants."""
    if converted is MATCH_ALL:
        return FILTER_ALL
    if converted is MATCH_NONE:
        return FILTER_NONE
    return converted


def combine_and(*filters: Any) -> Any:
    """Combine converted filters with AND."""
    operands = []
    for f in filters:
        if f is MATCH_NONE:
            return MATCH_NONE
        if f is MATCH_ALL or f is None:
            continue
        operands.append(f)
    if not operands:
        return MATCH_ALL
    if len(operands) == 1:
        return operands[0]
    return ["And", operands]


def _combine_or(operands: list[Any]) -> Any:
    kept = []
    for f in operands:
        if f is MATCH_ALL:
            return MATCH_ALL
        if f is MATCH_NONE:
            continue
        kept.append(f)
    if not kept:
        return MATCH_NONE
    if len(kept) == 1:
        return kept[0]
    return ["Or", kept]


def _negate(f: Any) -> Any:
    if f is MATCH_ALL:
        return MATCH_NONE
    if f is MATCH_NONE:
        return MATCH_ALL
    return ["Not", f]


def _convert(condition: dict[str, Any], schema: dict[str, dict] | None) -> Any:
    if "field" in condition:
        return _convert_comparison(condition, schema)

    if "operator" not in condition:
        raise VectorStoreFilterException(f"'operator' key missing in {condition}")
    if "conditions" not in condition:
        raise VectorStoreFilterException(f"'conditions' key missing in {condition}")

    operator = condition["operator"]
    operands = [_convert(c, schema) for c in condition["conditions"]]
    if operator == "AND":
        return combine_and(*operands)
    if operator == "OR":
        return _combine_or(operands)
    if operator == "NOT":
        return _negate(combine_and(*operands))
    raise VectorStoreFilterException(f"Unknown logical operator '{operator}'")


def referenced_fields(filters: dict[str, Any] | None) -> set[str]:
    """Return the attributes the filters read."""
    if not isinstance(filters, dict):
        return set()

    fields: set[str] = set()

    def walk(condition: Any) -> None:
        if not isinstance(condition, dict):
            return
        if isinstance(field := condition.get("field"), str):
            fields.add(field[len("metadata.") :] if field.startswith("metadata.") else field)
        for nested in condition.get("conditions") or []:
            walk(nested)

    walk(normalize_filters(filters))
    return fields


def _attribute(field: str, schema: dict[str, dict] | None) -> tuple[bool, str | None]:
    """Return whether the attribute exists, and its type when known."""
    if schema is None:
        return True, None
    config = schema.get(field)
    if config is None:
        return False, None
    filterable = config.get("filterable", not config.get("full_text_search"))
    if field != "id" and not filterable:
        raise VectorStoreFilterException(
            f"Attribute '{field}' is not filterable in this namespace, so filters cannot use it."
        )
    return True, config.get("type")


def _require_list(field: str, operator: str, value: Any) -> list:
    if not isinstance(value, list):
        raise VectorStoreFilterException(f"{field}'s value must be a list when using '{operator}' comparator")
    return value


def _convert_comparison(condition: dict[str, Any], schema: dict[str, dict] | None) -> Any:
    if "operator" not in condition:
        raise VectorStoreFilterException(f"'operator' key missing in {condition}")
    if "value" not in condition:
        raise VectorStoreFilterException(f"'value' key missing in {condition}")

    field: str = condition["field"]
    if field.startswith("metadata."):
        field = field[len("metadata.") :]
    operator: str = condition["operator"]
    value: Any = condition["value"]

    if operator not in _OPERATORS:
        raise VectorStoreFilterException(f"Unknown comparison operator '{operator}'")
    if operator in ("in", "not in", "contains_any", "contains_all"):
        value = _require_list(field, operator, value)

    exists, attribute_type = _attribute(field, schema)
    if not exists:
        # The attribute is missing on every document, so each condition is evaluated against null.
        return _MISSING_ATTRIBUTE[operator](value)

    if attribute_type is not None and value is not None:
        value = _coerce_filter_value(value, attribute_type, operator)
        if value is MATCH_NONE:
            return MATCH_ALL if operator in _NEGATED_OPERATORS else MATCH_NONE

    if is_array_type(attribute_type):
        return _ARRAY_OPERATORS[operator](field, value)
    return _OPERATORS[operator](field, value)


def _coerce_filter_value(value: Any, attribute_type: str, operator: str) -> Any:
    """Convert a filter value to the attribute's type.

    Values go through the same normalization as written values, so a datetime, Decimal, Enum or
    NumPy scalar matches what was stored for it. Returns MATCH_NONE when no stored value can match:
    a single value of another type, a list with no value of the attribute's type, or a contains_all
    list with any value of another type.
    """
    scalar_type = element_type(attribute_type) if is_array_type(attribute_type) else attribute_type
    if not isinstance(value, list):
        coerced = coerce_value(normalize_value(value), scalar_type)
        return MATCH_NONE if coerced is None else coerced

    coerced = [coerce_value(normalize_value(v), scalar_type) for v in value]
    kept = [v for v in coerced if v is not None]
    if operator == "contains_all" and len(kept) < len(coerced):
        return MATCH_NONE
    if value and not kept:
        return MATCH_NONE
    return kept


def _eq(field: str, value: Any) -> Any:
    return [field, "Eq", value]


def _not_eq(field: str, value: Any) -> Any:
    return [field, "NotEq", value]


def _range(op: str) -> Any:
    def convert(field: str, value: Any) -> Any:
        if value is None:
            return MATCH_NONE
        condition = [field, op, value]
        if op in ("Lt", "Lte"):
            # Turbopuffer counts a null as smaller than any value.
            return ["And", [[field, "NotEq", None], condition]]
        return condition

    return convert


def _in(field: str, value: list) -> Any:
    return [field, "In", value] if value else MATCH_NONE


def _not_in(field: str, value: list) -> Any:
    return [field, "NotIn", value] if value else MATCH_ALL


def _contains_all_scalar(field: str, value: list) -> Any:
    distinct = list(dict.fromkeys(value))
    if not distinct:
        return MATCH_ALL
    if len(distinct) > 1:
        return MATCH_NONE
    return [field, "Eq", distinct[0]]


def _array_eq(field: str, value: Any) -> Any:
    if value is None:
        return [field, "Eq", None]
    return [field, "Contains", value]


def _array_not_eq(field: str, value: Any) -> Any:
    if value is None:
        return [field, "NotEq", None]
    return [field, "NotContains", value]


def _array_range(op: str) -> Any:
    def convert(field: str, value: Any) -> Any:
        if value is None:
            return MATCH_NONE
        return [field, f"Any{op}", value]

    return convert


def _array_contains_any(field: str, value: list) -> Any:
    return [field, "ContainsAny", value] if value else MATCH_NONE


def _array_not_contains_any(field: str, value: list) -> Any:
    return [field, "NotContainsAny", value] if value else MATCH_ALL


def _array_contains_all(field: str, value: list) -> Any:
    return combine_and(*[[field, "Contains", v] for v in dict.fromkeys(value)])


_OPERATORS = {
    "==": _eq,
    "!=": _not_eq,
    ">": _range("Gt"),
    ">=": _range("Gte"),
    "<": _range("Lt"),
    "<=": _range("Lte"),
    "in": _in,
    "not in": _not_in,
    "contains_any": _in,
    "contains_all": _contains_all_scalar,
}

_ARRAY_OPERATORS = {
    "==": _array_eq,
    "!=": _array_not_eq,
    ">": _array_range("Gt"),
    ">=": _array_range("Gte"),
    "<": _array_range("Lt"),
    "<=": _array_range("Lte"),
    "in": _array_contains_any,
    "not in": _array_not_contains_any,
    "contains_any": _array_contains_any,
    "contains_all": _array_contains_all,
}

_NEGATED_OPERATORS = {"!=", "not in"}

_MISSING_ATTRIBUTE = {
    "==": lambda value: MATCH_ALL if value is None else MATCH_NONE,
    "!=": lambda value: MATCH_NONE if value is None else MATCH_ALL,
    ">": lambda value: MATCH_NONE,
    ">=": lambda value: MATCH_NONE,
    "<": lambda value: MATCH_NONE,
    "<=": lambda value: MATCH_NONE,
    "in": lambda value: MATCH_NONE,
    "not in": lambda value: MATCH_ALL,
    "contains_any": lambda value: MATCH_NONE,
    "contains_all": lambda value: MATCH_ALL if not value else MATCH_NONE,
}
