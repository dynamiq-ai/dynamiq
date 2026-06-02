import json
import re
from typing import Any, Match


def extract_json_string(s: str) -> str | None:
    """
    Extract the first JSON object or array from the string by balancing brackets.
    The function looks for '{' or '[' and keeps track of nested brackets until
    they are balanced, returning the substring that contains the complete JSON.

    Args:
        s: The input string potentially containing a JSON object or array.

    Returns:
        The extracted JSON string if found and balanced, otherwise None.
    """
    bracket_stack: list[str] = []
    start_index: int | None = None
    in_string = False
    escape = False

    for i, char in enumerate(s):
        # Toggle in_string when encountering an unescaped double quote
        if char == '"' and not escape:
            in_string = not in_string
        elif char == "\\" and not escape:
            escape = True
            continue

        if not in_string:
            if char in "{[":
                if not bracket_stack:
                    start_index = i
                bracket_stack.append(char)
            elif char in "}]":
                if bracket_stack:
                    opening_bracket = bracket_stack.pop()
                    if (opening_bracket == "{" and char != "}") or (opening_bracket == "[" and char != "]"):
                        # Mismatched brackets
                        return None
                    # If stack is empty, we've balanced everything
                    if not bracket_stack and start_index is not None:
                        return s[start_index : i + 1]
                else:
                    # Found a closing bracket without a matching opener
                    return None
        escape = False

    # If brackets never fully balanced, return None
    return None


def parse_llm_json_output(response: str) -> dict[str, Any] | list[Any]:
    """
    Attempt to parse the received LLM output into a JSON object or array.
    If direct parsing fails, looks for the first balanced JSON substring,
    then tries corrections.

    Args:
        response: The raw output from the LLM.

    Returns:
        A Python dict or list representing the parsed JSON.

    Raises:
        ValueError: If the output cannot be parsed into valid JSON.
    """
    # Try directly parsing
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        pass

    # Attempt bracket extraction
    json_str = extract_json_string(response)
    if not json_str:
        raise ValueError(f"Response from LLM is not valid JSON: {response}")

    # Try parsing the extracted substring
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        pass

    # Clean and try again
    cleaned_json_str = clean_json_string(json_str)
    try:
        return json.loads(cleaned_json_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse JSON after corrections: {e}")


def _remove_comments_outside_strings(source: str) -> str:
    """
    Remove JavaScript/JSON style comments (//... or /*...*/) from a string,
    but only if they occur outside quoted string literals.

    Args:
        source: The raw JSON-like string containing comments.

    Returns:
        The string with comments removed where appropriate.
    """
    result: list[str] = []
    i = 0
    n = len(source)
    in_str = False
    str_delim: str | None = None

    while i < n:
        c = source[i]
        if not in_str:
            # Look for a string start or comment
            if c in ["'", '"']:
                in_str = True
                str_delim = c
                result.append(c)
                i += 1
            elif c == "/" and i + 1 < n:
                next_char = source[i + 1]
                if next_char == "/":
                    # Single-line comment: skip until newline
                    i += 2
                    while i < n and source[i] not in ("\r", "\n"):
                        i += 1
                elif next_char == "*":
                    # Multi-line comment: skip until "*/"
                    i += 2
                    while i + 1 < n and not (source[i] == "*" and source[i + 1] == "/"):
                        i += 1
                    i += 2  # Skip the closing '*/'
                else:
                    # Just a '/', not a comment
                    result.append(c)
                    i += 1
            else:
                # Normal character outside a string
                result.append(c)
                i += 1
        else:
            # Currently inside a string
            if c == str_delim:
                # Check if this quote is escaped
                if i > 0 and source[i - 1] == "\\":
                    # It's an escaped quote
                    result.append(c)
                    i += 1
                else:
                    # Closing the string
                    in_str = False
                    str_delim = None
                    result.append(c)
                    i += 1
            else:
                # Normal character inside the string
                result.append(c)
                i += 1

    return "".join(result)


def single_quoted_replacer(match: Match[str]) -> str:
    """
    A helper function for clean_json_string to replace single-quoted JSON-like
    string literals with double-quoted equivalents, preserving internal
    apostrophes.

    Args:
        match: The regular expression match object for a single-quoted string.

    Returns:
        The corresponding double-quoted string literal.
    """
    content = match.group(1)
    # Convert escaped \' to an actual apostrophe
    content = content.replace("\\'", "'")
    # Escape any double quotes inside
    content = content.replace('"', '\\"')
    return f'"{content}"'


def clean_json_string(json_str: str) -> str:
    """
    Clean a JSON-like string so that it can be parsed by the built-in `json` module.

    1. Remove single-line (//...) and multi-line (/*...*/) comments outside of
       string literals.
    2. Convert single-quoted string literals to double-quoted ones, preserving
       internal apostrophes.
    3. Replace Python-specific boolean/null literals (True, False, None) with their
       JSON equivalents (true, false, null).
    4. Remove trailing commas in objects and arrays.

    Args:
        json_str: The raw JSON-like string to clean.

    Returns:
        A cleaned JSON string suitable for json.loads.
    """
    # 1. Remove comments
    json_str = _remove_comments_outside_strings(json_str)

    # 2. Convert single‐quoted string literals -> double‐quoted
    pattern = r"'((?:\\'|[^'])*)'"
    json_str = re.sub(pattern, single_quoted_replacer, json_str)

    # 3. Replace Python-specific boolean/null with JSON equivalents
    json_str = re.sub(r"\bTrue\b", "true", json_str)
    json_str = re.sub(r"\bFalse\b", "false", json_str)
    json_str = re.sub(r"\bNone\b", "null", json_str)

    # 4. Remove trailing commas before a closing bracket or brace
    json_str = re.sub(r",\s*(\]|\})", r"\1", json_str)

    return json_str
