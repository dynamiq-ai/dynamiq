from datetime import datetime
from typing import Any, Literal

from psycopg.sql import SQL
from psycopg.types.json import Jsonb

from dynamiq.storages.vector.exceptions import VectorStoreFilterException

NO_VALUE = "no_value"

PYTHON_TO_PGVECTOR_DATA_TYPE = {
    str: "text",
    int: "int",
    float: "float",
    bool: "bool",
}


def _convert_filters_to_query(
    filters: dict[str, Any], operator: Literal["WHERE", "AND"] = "WHERE"
) -> tuple[SQL, tuple]:
    """
    Convert filters from dict format to an SQL query and a tuple of params to query pgvector.

    Args:
        filters (dict[str, Any]): The filters to convert.
        operator (Literal["WHERE", "AND"], optional): The logical operator to use. Defaults to "WHERE".

    Returns:
        tuple[SQL, tuple]: The SQL query and a tuple of params to query pgvector.
    """
    if "field" in filters:
        query, values = _parse_comparison_condition(filters)
    else:
        query, values = _parse_logical_condition(filters)

    where_clause = SQL(f" {operator} ") + SQL(query)
    params = tuple(value for value in values if value != NO_VALUE)

    return where_clause, params


def _parse_logical_condition(condition: dict[str, Any]) -> tuple[str, list[Any]]:
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
    if operator not in LOGICAL_OPERATORS:
        msg = f"unknown logical operator '{operator}'."
        raise VectorStoreFilterException(msg)

    query_parts, values = [], []
    for c in condition["conditions"]:
        if "field" in c:
            query, vals = _parse_comparison_condition(c)
        else:
            query, vals = _parse_logical_condition(c)

        query_parts.append(query)
        values.append(vals)

    values = [item for sublist in values for item in sublist] if isinstance(values[0], list) else values

    if operator == "AND":
        sql_query = f"({' AND '.join(query_parts)})"
    elif operator == "OR":
        sql_query = f"({' OR '.join(query_parts)})"
    else:
        msg = f"Unknown logical operator '{operator}'"
        raise VectorStoreFilterException(msg)

    return sql_query, values


def _parse_comparison_condition(condition: dict[str, Any]) -> tuple[str, list[Any]]:
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
    if operator not in COMPARISON_OPERATORS:
        msg = f"Unknown comparison operator '{operator}'. Valid operators are: {list(COMPARISON_OPERATORS.keys())}"
        raise VectorStoreFilterException(msg)

    value: Any = condition["value"]

    if field.startswith("metadata."):
        field = _parse_metadata_fields_in_metadata(field, value)

    field, value = COMPARISON_OPERATORS[operator](field, value)
    return field, [value]


def _parse_metadata_fields_in_metadata(field: str, value: Any) -> str:
    """
    Parses metadata fields by using the ->> operator to access keys in the metadata JSONB field
    and casting them to the correct type.

    Args:
        field (str): The field name.
        value (Any): The value to determine the type for casting.

    Returns:
        str: The modified field with the correct type casting.
    """

    # Get the field name from the metadata field
    field_name = field.split(".", 1)[-1]

    # Retrieve the field using the ->> operator
    field = f"metadata->>'{field_name}'"

    # Determine the type for casting
    type_value = None
    if isinstance(value, list) and value:
        type_value = PYTHON_TO_PGVECTOR_DATA_TYPE.get(type(value[0]))
    else:
        type_value = PYTHON_TO_PGVECTOR_DATA_TYPE.get(type(value))

    # Cast the field to the correct type
    if type_value:
        field = f"({field})::{type_value}"

    return field


def _validate_equality_value(value: Any) -> None:
    """
    Validate value type for equality operators.

    Args:
        value (Any): Value to be validated.

    Raises:
        VectorStoreFilterException: If the value type is not supported.
    """
    supported_types = (str, int, float, bool)
    if not isinstance(value, supported_types):
        msg = f"Unsupported type for comparison: {type(value)}. " f"Types supported by PGVector are: {supported_types}"
        raise VectorStoreFilterException(msg)


def _validate_list_value(field: str, value: Any, operator: str) -> None:
    """
    Validate list values for 'in' and 'not in' operators.

    Args:
        field (str): Field name.
        value (Any): Value to be validated.
        operator (str): Comparison operator.

    Raises:
        VectorStoreFilterException: If the value is not a list or contains unsupported types.
    """
    if not isinstance(value, list):
        msg = f"{field}'s value must be a list when using '{operator}' comparator in PGVector"
        raise VectorStoreFilterException(msg)

    supported_types = (int, float, str)
    for v in value:
        if not isinstance(v, supported_types):
            msg = (
                f"Unsupported type for '{operator}' comparison: {type(v)}. "
                f"Types supported by PGVector are: {supported_types}"
            )
            raise VectorStoreFilterException(msg)


def _validate_comparison_value(value: Any) -> None:
    """
    Common validation for comparison operators.

    Args:
        value (Any): Value to be validated.

    Raises:
        VectorStoreFilterException: If the value is of an unsupported type.
    """
    if isinstance(value, str):
        try:
            datetime.fromisoformat(value)
        except (ValueError, TypeError) as exc:
            msg = "Can't compare strings using operators '>', '>=', '<', '<='. "
            raise VectorStoreFilterException(msg) from exc
    if isinstance(value, (list, Jsonb)):
        msg = f"Filter value can't be of type {type(value)} using operators '>', '>=', '<', '<='"
        raise VectorStoreFilterException(msg)


def _comparison_operator(field: str, value: Any, operator: str) -> tuple[str, Any]:
    """
    Generic comparison operator function.

    Args:
        field (str): Field name.
        value (Any): Value to compare against.
        operator (str): Comparison operator.

    Returns:
        tuple[str, Any]: The query string and the value to be used in the query.
    """
    _validate_comparison_value(value)
    return f"{field} {operator} %s", value


def _equal(field: str, value: Any) -> tuple[str, Any]:
    """
    Creates a `==` comparison filter.

    Args:
        field (str): The field to compare.
        value (Any): The value to compare against.

    Returns:
        tuple[str, list]: A `==` comparison SQL query and params to use in the query.

    Raises:
        VectorStoreFilterException: If the value type is not supported.
    """
    if value is None:
        return f"{field} IS NULL", NO_VALUE
    _validate_equality_value(value)
    return f"{field} = %s", value


def _not_equal(field: str, value: Any) -> tuple[str, Any]:
    """
    Creates a `!=` comparison filter.

    Args:
        field (str): The field to compare.
        value (Any): The value to compare against.

    Returns:
        tuple[str, list]: A `!=` comparison SQL query and params to use in the query.

    Raises:
        VectorStoreFilterException: If the value type is not supported.
    """
    _validate_equality_value(value)
    return f"{field} IS DISTINCT FROM %s", value


def _greater_than(field: str, value: Any) -> tuple[str, Any]:
    """
    Creates a `>` comparison filter.

    Args:
        field (str): The field to compare.
        value (Any): The value to compare against.

    Returns:
        tuple[str, list]: A `>` comparison SQL query and params to use in the query.

    Raises:
        VectorStoreFilterException: If the value type is not supported.
    """
    return _comparison_operator(field, value, ">")


def _greater_than_equal(field: str, value: Any) -> tuple[str, Any]:
    """
    Creates a `>=` comparison filter.

    Args:
        field (str): The field to compare.
        value (Any): The value to compare against.

    Returns:
        tuple[str, list]: A `>=` comparison SQL query and params to use in the query.

    Raises:
        VectorStoreFilterException: If the value type is not supported.
    """
    return _comparison_operator(field, value, ">=")


def _less_than(field: str, value: Any) -> tuple[str, Any]:
    """
    Creates a `<` comparison filter.

    Args:
        field (str): The field to compare.
        value (Any): The value to compare against.

    Returns:
        tuple[str, list]: A `<` comparison SQL query and params to use in the query.

    Raises:
        VectorStoreFilterException: If the value type is not supported.
    """
    return _comparison_operator(field, value, "<")


def _less_than_equal(field: str, value: Any) -> tuple[str, Any]:
    """
    Creates a `<=` comparison filter.

    Args:
        field (str): The field to compare.
        value (Any): The value to compare against.

    Returns:
        tuple[str, list]: A `<=` comparison SQL query and params to use in the query.

    Raises:
        VectorStoreFilterException: If the value type is not supported.
    """
    return _comparison_operator(field, value, "<=")


def _not_in(field: str, value: Any) -> tuple[str, list]:
    """
    Creates a `not in` comparison filter.

    Args:
        field (str): The field to compare.
        value (Any): The list of values to compare against.

    Returns:
        tuple[str, list]: A `not in` comparison SQL query and params to use in the query.

    Raises:
        VectorStoreFilterException: If the value is not a list or contains unsupported types.
    """
    _validate_list_value(field, value, "not in")
    return f"{field} IS NULL OR {field} != ALL(%s::text[])", [value]


def _in(field: str, value: Any) -> tuple[str, list]:
    """
    Creates a `in` comparison filter.

    Args:
        field (str): The field to compare.
        value (Any): The list of values to compare against.

    Returns:
        tuple[str, list]: A `in` comparison SQL query and params to use in the query.

    Raises:
        VectorStoreFilterException: If the value is not a list or contains unsupported types.
    """
    _validate_list_value(field, value, "in")
    return f"{field} = ANY(%s::text[])", [value]


LOGICAL_OPERATORS = ["AND", "OR"]

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
