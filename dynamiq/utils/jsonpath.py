from jsonpath_ng import parse
from jsonpath_ng.exceptions import JsonPathParserError

JSONPATH_EXPRESSION_PREFIXES = ("$", "@")
COALESCE_SEPARATOR = "||"


def is_jsonpath(path: str) -> bool:
    """
    Check if the given string is a valid JSONPath expression.

    Args:
        path (str): The string to be checked.

    Returns:
        bool: True if the string is a valid JSONPath expression, False otherwise.
    """
    try:
        parse(path)
        return True
    except JsonPathParserError:
        return False


def is_expression(path: str, expression_prefixes: tuple = JSONPATH_EXPRESSION_PREFIXES) -> bool:
    """
    Check whether a string is a rooted JSONPath expression rather than a literal value.

    `is_jsonpath` alone is not enough here: a bare word such as "a" parses as a valid
    JSONPath, so plain text would otherwise be mistaken for a path.

    Args:
        path (str): The string to be checked.
        expression_prefixes (tuple): The prefixes a JSONPath expression must start with.

    Returns:
        bool: True if the string is a rooted JSONPath expression, False otherwise.
    """
    if not isinstance(path, str) or not path.startswith(expression_prefixes):
        return False

    try:
        return is_jsonpath(path)
    except Exception:
        return False


def _coalesce(json: dict | list, candidates: list[str], expression_prefixes: tuple):
    """
    Resolve an ordered list of alternatives, e.g. "$.a.output.x || $.b.output.y".

    Returns the value of the first candidate that resolves to something, or None.
    A candidate that is not a rooted JSONPath is a literal default, so a trailing
    "|| no result" always yields a populated field. Used when branches of a Choice
    converge on a single field and only one of them produced a value.

    Args:
        json (dict | list): The input JSON object to resolve against.
        candidates (list[str]): Alternatives, already split and stripped.
        expression_prefixes (tuple): The prefixes of the JSONPath expressions.

    Returns:
        The first resolved value, or None if nothing resolved.
    """
    for candidate in candidates:
        if not is_expression(candidate, expression_prefixes):
            return candidate

        found = [match for match in parse(candidate).find(json) if match.value is not None]
        if found:
            return found[0].value if len(found) == 1 else [match.value for match in found]

    return None


def mapper(
    json: dict | list,
    expression_map: dict,
    expression_prefixes: tuple = JSONPATH_EXPRESSION_PREFIXES,
    use_expression_as_value: bool = True,
) -> dict:
    """
    Map values from a JSON object or list to a new dictionary based on a mapping configuration.

    Args:
        json (dict | list): The input JSON object or list to be mapped.
        expression_map (dict): A dictionary defining the mapping configuration.
        expression_prefixes (tuple): The prefixes of the JSONPath expressions.
        use_expression_as_value (bool): Whether to use the expression as value if the value is not found.

    Returns:
        dict: A new dictionary with mapped values according to the provided configuration.

    Raises:
        TypeError: If the map is not a dictionary or if the json is neither a dictionary nor a list.
        ValueError: If there's an error in JSONPath parsing.
    """
    if not expression_map:
        return json
    if not isinstance(expression_map, dict):
        raise TypeError("Invalid `expression_map`: must be a dictionary")
    if not isinstance(json, dict) and not isinstance(json, list):
        raise TypeError("Invalid `json`: must be a dictionary or a list")

    new_json = {}
    for key, path in expression_map.items():
        if isinstance(path, str) and COALESCE_SEPARATOR in path:
            candidates = [c.strip() for c in path.split(COALESCE_SEPARATOR) if c.strip()]
            # "||" only means "try these in order" when at least one alternative is a
            # rooted JSONPath. Otherwise the value is ordinary text that happens to
            # contain the characters, and must be preserved as-is.
            if any(is_expression(candidate, expression_prefixes) for candidate in candidates):
                new_json[key] = _coalesce(json, candidates, expression_prefixes)
                continue
        if not is_jsonpath(path):
            new_json[key] = path
            continue
        try:
            found = parse(path).find(json)
            if not found:
                if use_expression_as_value and not path.startswith(expression_prefixes):
                    new_json[key] = path
                else:
                    new_json[key] = None
            elif len(found) == 1:
                new_json[key] = found[0].value
            else:
                new_json[key] = [v.value for v in found]
        except Exception as e:
            raise ValueError(f"Error in jsonpath during parsing: {e}")

    return new_json


def filter(json: dict, expression_filter: str):
    """
    Filter a JSON object based on a JSONPath expression.

    Args:
        json (dict): The input JSON object to be filtered.
        expression_filter (str): A JSONPath expression used to filter the JSON object.

    Returns:
        The filtered data, which can be a single value, a list of values, or None if no match is found.

    Raises:
        ValueError: If the filter is not a valid JSONPath expression or if there's an error in parsing.
    """
    if not expression_filter:
        return json
    if not is_jsonpath(expression_filter):
        raise ValueError(f"Invalid `expression_filter` {expression_filter}: must be a jsonpath")

    filtered_data = None
    try:
        value = parse(expression_filter).find(json)
        if value:
            filtered_data = [v.value for v in value]
            if len(filtered_data) == 1:
                filtered_data = filtered_data[0]
    except Exception as e:
        raise ValueError(f"Error in path during parsing with `expression_filter` {expression_filter}: {e}")

    return filtered_data


def filter_all(json: dict | list, expression_filter: str) -> list:
    """Filter a JSON object and return all matched values as a list.

    Unlike `filter`, this function always returns a list of matched values,
    making it possible to distinguish between "no match", "single match"
    (even if the value is a list or None), and "multiple matches" by checking len().

    Args:
        json: The input JSON object or list to be filtered.
        expression_filter: A JSONPath expression used to filter the JSON object.

    Returns:
        A list of all matched values (empty list if no matches found).

    Raises:
        ValueError: If the filter is not a valid JSONPath expression or if there's an error in parsing.
    """
    if not expression_filter:
        return [json]
    if not is_jsonpath(expression_filter):
        raise ValueError(f"Invalid `expression_filter` {expression_filter}: must be a jsonpath")

    try:
        matches = parse(expression_filter).find(json)
        return [m.value for m in matches]
    except Exception as e:
        raise ValueError(f"Error in path during parsing with `expression_filter` {expression_filter}: {e}")
