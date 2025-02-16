from typing import Any

from dynamiq.storages.vector.exceptions import VectorStoreFilterException


def _normalize_filters(filters: dict[str, Any]) -> dict[str, Any]:
    """
    Converts filters to Elasticsearch compatible filters.

    Args:
        filters (dict[str, Any]): The filters to be normalized.

    Returns:
        dict[str, Any]: Normalized filters compatible with Elasticsearch.

    Raises:
        VectorStoreFilterException: If filters are not in the correct format.

    Reference:
        https://www.elastic.co/guide/en/elasticsearch/reference/current/query-dsl-bool-query.html
    """
    if not isinstance(filters, dict):
        msg = "Filters must be a dictionary"
        raise VectorStoreFilterException(msg)

    if "field" in filters:
        return {LOGICAL_OPERATORS["AND"]: [_parse_comparison_condition(filters)]}
    return _parse_logical_condition(filters)


def _is_all_comparison(conditions: list[dict[str, Any]]) -> bool:
    """
    Checks if all conditions in the list are comparison conditions.

    Args:
        conditions (list[dict[str, Any]]): A list of conditions.

    Returns:
        bool: True if all conditions are comparison conditions; otherwise, False.
    """
    for condition in conditions:
        if "field" not in condition:
            return False
    return True


def _is_all_logical(conditions: list[dict[str, Any]]) -> bool:
    """
    Checks if all conditions in the list are logical conditions.

    Args:
        conditions (list[dict[str, Any]]): A list of conditions.

    Returns:
        bool: True if all conditions are logical conditions; otherwise, False.
    """
    for condition in conditions:
        if "conditions" not in condition:
            return False
    return True


def _parse_logical_condition(condition: dict[str, Any], initial: bool = True) -> dict[str, Any]:
    """
    Parses a logical condition in the filter.

    Args:
        condition (dict[str, Any]): The logical condition to be parsed.
        initial (bool, optional): A flag indicating if the condition is at the initial parsing level. Defaults to True.

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
    conditions = condition["conditions"]

    if operator == "AND":
        if _is_all_logical(conditions):
            if initial:
                conditions = [_parse_logical_condition(c, initial=False) for c in conditions]
                return {k: v for d in conditions for k, v in d.items()}
            else:
                msg = f"Logical conditions are only allowed at levels 0 and 1: {condition}"
                raise VectorStoreFilterException(msg)

        if not _is_all_comparison(conditions):
            msg = f"A conditions must be either logical or comparison, not both at the same level: {condition}"
            raise VectorStoreFilterException(msg)

    conditions = [_parse_comparison_condition(c) for c in conditions]

    if operator in LOGICAL_OPERATORS:
        return {LOGICAL_OPERATORS[operator]: conditions}

    msg = f"Unknown logical operator '{operator}'"
    raise VectorStoreFilterException(msg)


def _parse_comparison_condition(condition: dict[str, Any]) -> dict[str, Any]:
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
        msg = f"'field' key missing in {condition}"
        raise VectorStoreFilterException(msg)
    if "operator" not in condition:
        msg = f"'operator' key missing in {condition}"
        raise VectorStoreFilterException(msg)
    if "value" not in condition:
        msg = f"'value' key missing in {condition}"
        raise VectorStoreFilterException(msg)

    field: str = condition["field"]
    operator: str = condition["operator"]
    value: Any = condition["value"]

    if operator not in COMPARISON_OPERATORS:
        msg = f"Unknown comparison operator '{operator}'"
        raise VectorStoreFilterException(msg)

    return COMPARISON_OPERATORS[operator](field, value)


def _equal(field: str, value: Any) -> dict[str, Any]:
    """
    Creates an equality comparison filter.

    Args:
        field (str): The field to compare.
        value (Any): The value to compare against.

    Returns:
        dict[str, Any]: An equality comparison filter.

    Raises:
        VectorStoreFilterException: If the value type is not supported.
    """
    supported_types = (str, int, float, bool)
    if not isinstance(value, supported_types):
        msg = (
            f"Unsupported type for 'equal' comparison: {type(value)}. "
            f"Types supported by Elasticsearch are: {supported_types}"
        )
        raise VectorStoreFilterException(msg)

    return {"match": {field: value}}


def _greater_than(field: str, value: Any) -> dict[str, Any]:
    """
    Creates a greater than comparison filter.

    Args:
        field (str): The field to compare.
        value (Any): The value to compare against.

    Returns:
        dict[str, Any]: A greater than comparison filter.

    Raises:
        VectorStoreFilterException: If the value type is not supported.
    """
    supported_types = (int, float)
    if not isinstance(value, supported_types):
        msg = (
            f"Unsupported type for 'greater than' comparison: {type(value)}. "
            f"Types supported by Elasticsearch are: {supported_types}"
        )
        raise VectorStoreFilterException(msg)

    return {"range": {field: {"gt": value}}}


def _greater_than_equal(field: str, value: Any) -> dict[str, Any]:
    """
    Creates a greater than or equal comparison filter.

    Args:
        field (str): The field to compare.
        value (Any): The value to compare against.

    Returns:
        dict[str, Any]: A greater than or equal comparison filter.

    Raises:
        VectorStoreFilterException: If the value type is not supported.
    """
    supported_types = (int, float)
    if not isinstance(value, supported_types):
        msg = (
            f"Unsupported type for 'greater than equal' comparison: {type(value)}. "
            f"Types supported by Elasticsearch are: {supported_types}"
        )
        raise VectorStoreFilterException(msg)

    return {"range": {field: {"gte": value}}}


def _less_than(field: str, value: Any) -> dict[str, Any]:
    """
    Creates a less than comparison filter.

    Args:
        field (str): The field to compare.
        value (Any): The value to compare against.

    Returns:
        dict[str, Any]: A less than comparison filter.

    Raises:
        VectorStoreFilterException: If the value type is not supported.
    """
    supported_types = (int, float)
    if not isinstance(value, supported_types):
        msg = (
            f"Unsupported type for 'less than' comparison: {type(value)}. "
            f"Types supported by Elasticsearch are: {supported_types}"
        )
        raise VectorStoreFilterException(msg)

    return {"range": {field: {"lt": value}}}


def _less_than_equal(field: str, value: Any) -> dict[str, Any]:
    """
    Creates a less than or equal comparison filter.

    Args:
        field (str): The field to compare.
        value (Any): The value to compare against.

    Returns:
        dict[str, Any]: A less than or equal comparison filter.

    Raises:
        VectorStoreFilterException: If the value type is not supported.
    """
    supported_types = (int, float)
    if not isinstance(value, supported_types):
        msg = (
            f"Unsupported type for 'less than equal' comparison: {type(value)}. "
            f"Types supported by Elasticsearch are: {supported_types}"
        )
        raise VectorStoreFilterException(msg)

    return {"range": {field: {"lte": value}}}


COMPARISON_OPERATORS = {
    "==": _equal,
    ">": _greater_than,
    ">=": _greater_than_equal,
    "<": _less_than,
    "<=": _less_than_equal,
}

LOGICAL_OPERATORS = {"AND": "must", "OR": "should", "NOT": "must_not"}
