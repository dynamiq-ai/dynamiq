from typing import Any

from dateutil import parser
from weaviate.collections.classes.filters import Filter, FilterReturn

from dynamiq.storages.vector.exceptions import VectorStoreFilterException


def convert_filters(filters: dict[str, Any]) -> FilterReturn:
    """
    Convert filters from dynamiq format to Weaviate format.

    Args:
        filters (dict[str, Any]): Filters in dynamiq format.

    Returns:
        FilterReturn: Filters in Weaviate format.

    Raises:
        VectorStoreFilterException: If filters are not a dictionary.
    """
    if not isinstance(filters, dict):
        msg = "Filters must be a dictionary"
        raise VectorStoreFilterException(msg)

    if "field" in filters:
        return Filter.all_of([_parse_comparison_condition(filters)])
    return _parse_logical_condition(filters)


OPERATOR_INVERSE = {
    "==": "!=",
    "!=": "==",
    ">": "<=",
    ">=": "<",
    "<": ">=",
    "<=": ">",
    "in": "not in",
    "not in": "in",
    "AND": "OR",
    "OR": "AND",
    "NOT": "OR",
}


def _invert_condition(filters: dict[str, Any]) -> dict[str, Any]:
    """
    Invert condition recursively. Weaviate doesn't support NOT filters so we need to invert them.

    Args:
        filters (dict[str, Any]): The filter condition to invert.

    Returns:
        dict[str, Any]: The inverted filter condition.
    """
    inverted_condition = filters.copy()
    if "operator" not in filters:
        return inverted_condition
    inverted_condition["operator"] = OPERATOR_INVERSE[filters["operator"]]
    if "conditions" in filters:
        inverted_condition["conditions"] = []
        for condition in filters["conditions"]:
            inverted_condition["conditions"].append(_invert_condition(condition))

    return inverted_condition


LOGICAL_OPERATORS = {
    "AND": Filter.all_of,
    "OR": Filter.any_of,
}


def _parse_logical_condition(condition: dict[str, Any]) -> FilterReturn:
    """
    Parse logical conditions in the filter.

    Args:
        condition (dict[str, Any]): The logical condition to parse.

    Returns:
        FilterReturn: The parsed logical condition.

    Raises:
        VectorStoreFilterException: If the operator or conditions are missing or unknown.
    """
    if "operator" not in condition:
        msg = f"'operator' key missing in {condition}"
        raise VectorStoreFilterException(msg)
    if "conditions" not in condition:
        msg = f"'conditions' key missing in {condition}"
        raise VectorStoreFilterException(msg)

    operator = condition["operator"]
    if operator in ["AND", "OR"]:
        operands = []
        for c in condition["conditions"]:
            if "field" not in c:
                operands.append(_parse_logical_condition(c))
            else:
                operands.append(_parse_comparison_condition(c))
        return LOGICAL_OPERATORS[operator](operands)
    elif operator == "NOT":
        inverted_conditions = _invert_condition(condition)
        return _parse_logical_condition(inverted_conditions)
    else:
        msg = f"Unknown logical operator '{operator}'"
        raise VectorStoreFilterException(msg)


def _handle_date(value: Any) -> str:
    """
    Handle date values by converting them to ISO format if possible.

    Args:
        value (Any): The value to handle.

    Returns:
        str: The handled value.
    """
    if isinstance(value, str):
        try:
            return parser.isoparse(value).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
        except ValueError:
            pass
    return value


def _equal(field: str, value: Any) -> FilterReturn:
    """
    Create an equality filter.

    Args:
        field (str): The field to filter on.
        value (Any): The value to compare against.

    Returns:
        FilterReturn: The equality filter.
    """
    if value is None:
        return Filter.by_property(field).is_none(True)
    return Filter.by_property(field).equal(_handle_date(value))


def _not_equal(field: str, value: Any) -> FilterReturn:
    """
    Create a not equal filter.

    Args:
        field (str): The field to filter on.
        value (Any): The value to compare against.

    Returns:
        FilterReturn: The not equal filter.
    """
    if value is None:
        return Filter.by_property(field).is_none(False)

    return Filter.by_property(field).not_equal(
        _handle_date(value)
    ) | Filter.by_property(field).is_none(True)


def _greater_than(field: str, value: Any) -> FilterReturn:
    """
    Create a greater than filter.

    Args:
        field (str): The field to filter on.
        value (Any): The value to compare against.

    Returns:
        FilterReturn: The greater than filter.

    Raises:
        VectorStoreFilterException: If the value is incompatible with the greater than operation.
    """
    if value is None:
        return _match_no_document(field)
    if isinstance(value, str):
        try:
            parser.isoparse(value)
        except (ValueError, TypeError) as exc:
            msg = (
                "Can't compare strings using operators '>', '>=', '<', '<='. "
                "Strings are only comparable if they are ISO formatted dates."
            )
            raise VectorStoreFilterException(msg) from exc
    return Filter.by_property(field).greater_than(_handle_date(value))


def _greater_than_equal(field: str, value: Any) -> FilterReturn:
    """
    Create a greater than or equal filter.

    Args:
        field (str): The field to filter on.
        value (Any): The value to compare against.

    Returns:
        FilterReturn: The greater than or equal filter.

    Raises:
        VectorStoreFilterException: If the value is incompatible with the greater than or equal operation.
    """
    if value is None:
        return _match_no_document(field)
    if isinstance(value, str):
        try:
            parser.isoparse(value)
        except (ValueError, TypeError) as exc:
            msg = (
                "Can't compare strings using operators '>', '>=', '<', '<='. "
                "Strings are only comparable if they are ISO formatted dates."
            )
            raise VectorStoreFilterException(msg) from exc
    return Filter.by_property(field).greater_or_equal(_handle_date(value))


def _less_than(field: str, value: Any) -> FilterReturn:
    """
    Create a less than filter.

    Args:
        field (str): The field to filter on.
        value (Any): The value to compare against.

    Returns:
        FilterReturn: The less than filter.

    Raises:
        VectorStoreFilterException: If the value is incompatible with the less than operation.
    """
    if value is None:
        return _match_no_document(field)
    if isinstance(value, str):
        try:
            parser.isoparse(value)
        except (ValueError, TypeError) as exc:
            msg = (
                "Can't compare strings using operators '>', '>=', '<', '<='. "
                "Strings are only comparable if they are ISO formatted dates."
            )
            raise VectorStoreFilterException(msg) from exc
    return Filter.by_property(field).less_than(_handle_date(value))


def _less_than_equal(field: str, value: Any) -> FilterReturn:
    """
    Create a less than or equal filter.

    Args:
        field (str): The field to filter on.
        value (Any): The value to compare against.

    Returns:
        FilterReturn: The less than or equal filter.

    Raises:
        VectorStoreFilterException: If the value is incompatible with the less than or equal operation.
    """
    if value is None:
        return _match_no_document(field)
    if isinstance(value, str):
        try:
            parser.isoparse(value)
        except (ValueError, TypeError) as exc:
            msg = (
                "Can't compare strings using operators '>', '>=', '<', '<='. "
                "Strings are only comparable if they are ISO formatted dates."
            )
            raise VectorStoreFilterException(msg) from exc
    return Filter.by_property(field).less_or_equal(_handle_date(value))


def _in(field: str, value: Any) -> FilterReturn:
    """
    Create an 'in' filter.

    Args:
        field (str): The field to filter on.
        value (Any): The value to compare against.

    Returns:
        FilterReturn: The 'in' filter.

    Raises:
        VectorStoreFilterException: If the value is not a list.
    """
    if not isinstance(value, list):
        msg = f"{field}'s value must be a list when using 'in' or 'not in' comparators"
        raise VectorStoreFilterException(msg)

    return Filter.by_property(field).contains_any(value)


def _not_in(field: str, value: Any) -> FilterReturn:
    """
    Create a 'not in' filter.

    Args:
        field (str): The field to filter on.
        value (Any): The value to compare against.

    Returns:
        FilterReturn: The 'not in' filter.

    Raises:
        VectorStoreFilterException: If the value is not a list.
    """
    if not isinstance(value, list):
        msg = f"{field}'s value must be a list when using 'in' or 'not in' comparators"
        raise VectorStoreFilterException(msg)
    operands = [Filter.by_property(field).not_equal(v) for v in value]
    return Filter.all_of(operands)


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


def _parse_comparison_condition(condition: dict[str, Any]) -> FilterReturn:
    """
    Parse comparison conditions in the filter.

    Args:
        condition (dict[str, Any]): The comparison condition to parse.

    Returns:
        FilterReturn: The parsed comparison condition.

    Raises:
        VectorStoreFilterException: If the operator or value is missing in the condition.
    """
    field: str = condition["field"]

    if field.startswith("metadata."):
        # Documents are flattened otherwise we wouldn't be able to properly query them.
        # We're forced to flatten because Weaviate doesn't support querying of nested properties
        # as of now. If we don't flatten the documents we can't filter them.
        # As time of writing this they have it in their backlog, see:
        # https://github.com/weaviate/weaviate/issues/3694
        field = field.replace("metadata.", "")

    if "operator" not in condition:
        msg = f"'operator' key missing in {condition}"
        raise VectorStoreFilterException(msg)
    if "value" not in condition:
        msg = f"'value' key missing in {condition}"
        raise VectorStoreFilterException(msg)
    operator: str = condition["operator"]
    value: Any = condition["value"]

    return COMPARISON_OPERATORS[operator](field, value)


def _match_no_document(field: str) -> FilterReturn:
    """
    Returns a filter that will match no Document.

    This is used to keep the behavior consistent between different Document Stores.

    Args:
        field (str): The field to create the filter on.

    Returns:
        FilterReturn: A filter that matches no Document.
    """
    operands = [Filter.by_property(field).is_none(val) for val in [False, True]]
    return Filter.all_of(operands)
