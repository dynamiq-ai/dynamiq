from jsonpath_ng import parse
from jsonpath_ng.exceptions import JsonPathParserError


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


def mapper(json: dict | list, expression_map: dict) -> dict:
    """
    Map values from a JSON object or list to a new dictionary based on a mapping configuration.

    Args:
        json (dict | list): The input JSON object or list to be mapped.
        expression_map (dict): A dictionary defining the mapping configuration.

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
        if not is_jsonpath(path):
            new_json[key] = path
            continue
        try:
            found = parse(path).find(json)
            if not found:
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
