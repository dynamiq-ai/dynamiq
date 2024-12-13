import json
import re
from typing import Any


def extract_json_string(s: str) -> str | None:
    """Extract the first JSON object or array from the string by balancing brackets.

    Args:
        s (str): The input string containing the JSON object or array.

    Returns:
        Optional[str]: The extracted JSON string, or None if not found.
    """
    stack = []
    start = None
    in_string = False
    escape = False

    for i, char in enumerate(s):
        if char == '"' and not escape:
            in_string = not in_string
        elif char == "\\" and not escape:
            escape = True
            continue

        if not in_string:
            if char in "{[":
                if not stack:
                    start = i
                stack.append(char)
            elif char in "}]":
                if stack:
                    opening_bracket = stack.pop()
                    if (opening_bracket == "{" and char != "}") or (opening_bracket == "[" and char != "]"):
                        # Mismatched brackets
                        return None
                    if not stack and start is not None:
                        return s[start : i + 1]
                else:
                    # Unbalanced closing bracket
                    return None
        escape = False
    return None


def clean_json_string(json_str: str) -> str:
    """Clean the JSON string to make it parseable by the json module.

    Args:
        json_str (str): The JSON string to clean.

    Returns:
        str: The cleaned JSON string.
    """
    # Remove comments (single-line // and multi-line /* */)
    json_str = re.sub(r"//.*?$|/\*.*?\*/", "", json_str, flags=re.DOTALL | re.MULTILINE)

    # Replace single quotes with double quotes
    json_str = re.sub(r"(?<!\\)'", '"', json_str)

    # Replace Python literals with JSON equivalents
    json_str = re.sub(r"\bTrue\b", "true", json_str)
    json_str = re.sub(r"\bFalse\b", "false", json_str)
    json_str = re.sub(r"\bNone\b", "null", json_str)

    # Remove trailing commas
    json_str = re.sub(r",\s*(\]|\})", r"\1", json_str)

    return json_str


def parse_llm_json_output(response: str) -> dict[str, Any] | list[Any]:
    """Attempt to parse the received LLM output into a JSON object.

    Args:
        response (str): The raw output from the LLM.

    Returns:
        dict[str, Any] | list[Any]: The parsed JSON object.

    Raises:
        ValueError: If the output cannot be parsed into valid JSON.
    """
    try:
        # First, try to parse the received string directly.
        return json.loads(response)
    except json.JSONDecodeError:
        pass

    # Extract JSON string
    json_str = extract_json_string(response)
    if not json_str:
        raise ValueError(f"Response from LLM is not valid JSON: {response}")

    # Clean the JSON string
    cleaned_json_str = clean_json_string(json_str)

    # Try parsing the cleaned JSON string
    try:
        return json.loads(cleaned_json_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse JSON after corrections: {e}")
