from typing import Any

from dynamiq.storages.vector.exceptions import VectorStoreFilterException


def _normalize_filters(filters: dict[str, Any]) -> dict[str, Any]:
    """
    Converts filters to Pinecone compatible filters.

    Args:
        filters (dict[str, Any]): The filters to be normalized.

    Returns:
        dict[str, Any]: Normalized filters compatible with Pinecone.

    Raises:
        VectorStoreFilterException: If filters are not in the correct format.

    Reference:
        https://docs.pinecone.io/docs/metadata-filtering
    """
    if not isinstance(filters, dict):
        msg = "Filters must be a dictionary"
        raise VectorStoreFilterException(msg)

    if "field" in filters:
        return _parse_comparison_condition(filters)
    return _parse_logical_condition(filters)


def _parse_logical_condition(condition: dict[str, Any]) -> dict[str, Any]:
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
        # 'field' key is only found in comparison dictionaries.
        # We assume this is a logic dictionary since it's not present.
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
        # Remove the "metadata." prefix if present.
        # Documents are flattened when using the PineconeDocumentStore
        # so we don't need to specify the "metadata." prefix.
        # Instead of raising an error we handle it gracefully.
        field = field.replace("metadata.", "")

    value: Any = condition["value"]

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
            f"Types supported by Pinecone are: {supported_types}"
        )
        raise VectorStoreFilterException(msg)

    return {field: {"$eq": value}}


def _not_equal(field: str, value: Any) -> dict[str, Any]:
    """
    Creates a not equal comparison filter.

    Args:
        field (str): The field to compare.
        value (Any): The value to compare against.

    Returns:
        dict[str, Any]: A not equal comparison filter.

    Raises:
        VectorStoreFilterException: If the value type is not supported.
    """
    supported_types = (str, int, float, bool)
    if not isinstance(value, supported_types):
        msg = (
            f"Unsupported type for 'not equal' comparison: {type(value)}. "
            f"Types supported by Pinecone are: {supported_types}"
        )
        raise VectorStoreFilterException(msg)

    return {field: {"$ne": value}}


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
            f"Types supported by Pinecone are: {supported_types}"
        )
        raise VectorStoreFilterException(msg)

    return {field: {"$gt": value}}


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
            f"Types supported by Pinecone are: {supported_types}"
        )
        raise VectorStoreFilterException(msg)

    return {field: {"$gte": value}}


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
            f"Types supported by Pinecone are: {supported_types}"
        )
        raise VectorStoreFilterException(msg)

    return {field: {"$lt": value}}


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
            f"Types supported by Pinecone are: {supported_types}"
        )
        raise VectorStoreFilterException(msg)

    return {field: {"$lte": value}}


def _not_in(field: str, value: Any) -> dict[str, Any]:
    """
    Creates a not in comparison filter.

    Args:
        field (str): The field to compare.
        value (Any): The list of values to compare against.

    Returns:
        dict[str, Any]: A not in comparison filter.

    Raises:
        VectorStoreFilterException: If the value is not a list or contains unsupported types.
    """
    if not isinstance(value, list):
        msg = (
            f"{field}'s value must be a list when using 'not in' comparator in Pinecone"
        )
        raise VectorStoreFilterException(msg)

    supported_types = (int, float, str)
    for v in value:
        if not isinstance(v, supported_types):
            msg = (
                f"Unsupported type for 'not in' comparison: {type(v)}. "
                f"Types supported by Pinecone are: {supported_types}"
            )
            raise VectorStoreFilterException(msg)

    return {field: {"$nin": value}}


def _in(field: str, value: Any) -> dict[str, Any]:
    """
    Creates an in comparison filter.

    Args:
        field (str): The field to compare.
        value (Any): The list of values to compare against.

    Returns:
        dict[str, Any]: An in comparison filter.

    Raises:
        VectorStoreFilterException: If the value is not a list or contains unsupported types.
    """
    if not isinstance(value, list):
        msg = f"{field}'s value must be a list when using 'in' comparator in Pinecone"
        raise VectorStoreFilterException(msg)

    supported_types = (int, float, str)
    for v in value:
        if not isinstance(v, supported_types):
            msg = (
                f"Unsupported type for 'in' comparison: {type(v)}. "
                f"Types supported by Pinecone are: {supported_types}"
            )
            raise VectorStoreFilterException(msg)

    return {field: {"$in": value}}


COMPARISON_OPERATORS = {
    "==": _equal,
    "!=": _not_equal,
    ">": _greater_than,
    ">=": _greater_than_equal,
    "<": _less_than,
    "<=": _less_than_equal,
    "in": _in,
    "not in": _not_in,
}

LOGICAL_OPERATORS = {"AND": "$and", "OR": "$or"}
