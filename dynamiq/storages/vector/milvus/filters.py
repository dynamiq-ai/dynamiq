from typing import Any

from dynamiq.storages.vector.exceptions import VectorStoreFilterException

COMPARISON_OPERATORS = ["==", "!=", ">", ">=", "<", "<=", "in", "not in"]

LOGICAL_OPERATORS = {"AND": "and", "OR": "or"}


def _normalize_filters(filters: dict[str, Any]) -> str:
    """
    Converts filters to Milvus compatible filters.

    Args:
        filters (dict[str, Any]): The filters to be normalized.

    Returns:
        str: Normalized filters compatible with Milvus.

    Raises:
        VectorStoreFilterException: If filters are not in the correct format.
    """
    if not isinstance(filters, dict):
        msg = "Filters must be a dictionary"
        raise VectorStoreFilterException(msg)

    if "field" in filters:
        return _parse_comparison_condition(filters)
    return _parse_logical_condition(filters)


def _parse_logical_condition(condition: dict[str, Any]) -> str:
    """
    Parses a logical condition in the filter.

    Args:
        condition (dict[str, Any]): The logical condition to be parsed.

    Returns:
        dict[str, Any]: Parsed logical condition.

    Raises:
        VectorStoreFilterException: If the condition is not properly formatted or has an unknown
        operator.
    """
    if "operator" not in condition:
        msg = f"'operator' key missing in {condition}"
        raise VectorStoreFilterException(msg)
    if "conditions" not in condition:
        msg = f"'conditions' key missing in {condition}"
        raise VectorStoreFilterException(msg)

    operator = condition["operator"]
    conditions = [_parse_comparison_condition(c) for c in condition["conditions"]]

    if operator in LOGICAL_OPERATORS:
        return f" {LOGICAL_OPERATORS[operator]} ".join(conditions)

    msg = f"Unknown logical operator '{operator}'"
    raise VectorStoreFilterException(msg)


def _parse_comparison_condition(condition: dict[str, Any]) -> str:
    """
    Parses a comparison condition in the filter.

    Args:
        condition (dict[str, Any]): The comparison condition to be parsed.

    Returns:
        dict[str, Any]: Parsed comparison condition.

    Raises:
        VectorStoreFilterException: If the condition is not properly formatted.
    """
    if "field" not in condition:
        return _parse_logical_condition(condition)

    field: str = condition["field"]
    if "operator" not in condition:
        msg = f"'operator' key missing in {condition}"
        raise VectorStoreFilterException(msg)
    if "value" not in condition:
        msg = f"'value' key missing in {condition}"
        raise VectorStoreFilterException(msg)
    operator: str = condition["operator"]
    if field.startswith("metadata."):
        field = field.replace("metadata.", "")

    value: Any = condition["value"]
    if operator in COMPARISON_OPERATORS:
        return f"{field} {operator} {value}"

    msg = f"Unknown comparison operator '{operator}'"
    raise VectorStoreFilterException(msg)
