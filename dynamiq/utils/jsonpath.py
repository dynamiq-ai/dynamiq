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


def mapper(json: dict | list, map: dict, node_id: str) -> dict:
    """
    Map values from a JSON object or list to a new dictionary based on a mapping configuration.

    Args:
        json (dict | list): The input JSON object or list to be mapped.
        map (dict): A dictionary defining the mapping configuration.
        node_id (str): An identifier for the current node being processed.

    Returns:
        dict: A new dictionary with mapped values according to the provided configuration.

    Raises:
        TypeError: If the map is not a dictionary or if the json is neither a dictionary nor a list.
        ValueError: If there's an error in JSONPath parsing.
    """
    if not map:
        return json
    if not isinstance(map, dict):
        raise TypeError(f"Invalid map of node {node_id}: map must be a dictionary")
    if not isinstance(json, dict) and not isinstance(json, list):
        raise TypeError(
            f"Invalid json of node {node_id}: json must be a dictionary or a list"
        )

    new_json = {}
    for key, path in map.items():
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
            raise ValueError(f"Error in jsonpath parsing of node {node_id}: {e}")

    return new_json


def filter(json: dict, filter: str, node_id: str):
    """
    Filter a JSON object based on a JSONPath expression.

    Args:
        json (dict): The input JSON object to be filtered.
        filter (str): A JSONPath expression used to filter the JSON object.
        node_id (str): An identifier for the current node being processed.

    Returns:
        The filtered data, which can be a single value, a list of values, or None if no match is found.

    Raises:
        ValueError: If the filter is not a valid JSONPath expression or if there's an error in parsing.
    """
    if not filter:
        return json
    if not is_jsonpath(filter):
        raise ValueError(f"Invalid filter of node {node_id}: filter must be a jsonpath")

    filtered_data = None
    try:
        value = parse(filter).find(json)
        if value:
            filtered_data = [v.value for v in value]
            if len(filtered_data) == 1:
                filtered_data = filtered_data[0]
        else:
            filtered_data = None
    except Exception as e:
        raise ValueError(f"Error in path parsing of node {node_id}: {e}")

    return filtered_data
