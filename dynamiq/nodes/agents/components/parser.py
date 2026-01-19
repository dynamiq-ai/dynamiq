"""Parsing logic for Agent LLM outputs in different formats."""

import json
import re

from dynamiq.nodes.agents.exceptions import ActionParsingException
from dynamiq.utils.logger import logger


def parse_default_thought(output: str) -> str:
    """
    Extracts thought from the output string in default inference mode format.

    Args:
        output: The LLM output string

    Returns:
        The extracted thought or empty string if not found
    """
    thought_match = re.search(
        r"Thought:\s*(.*?)Action",
        output,
        re.DOTALL,
    )

    if thought_match:
        return thought_match.group(1).strip()

    return ""


def parse_default_action(
    output: str, parallel_tool_calls_enabled: bool
) -> tuple[str | None, str | None, dict | list | None]:
    """
    Parses the action(s), input(s), and thought from the output string in default inference mode format.

    Supports both single tool actions and multiple sequential tool calls
    when multi-tool is enabled.

    Args:
        output: The output string from the LLM containing Thought, Action, and Action Input
        parallel_tool_calls_enabled: Whether parallel tool calls are enabled

    Returns:
        tuple: (thought, action_type, actions_data) where:
            - thought is the extracted reasoning
            - action_type is either a tool name (for single tool) or "multiple_tools" (for multiple tools)
            - actions_data is either a dict (for single tool) or a list of dicts (for multiple tools)

    Raises:
        ActionParsingException: If parsing fails or format is invalid
    """
    try:
        thought = parse_default_thought(output) or None

        actions = []
        action_input_matches = list(re.finditer(r"Action Input:", output))

        for input_match in action_input_matches:
            preceding_text = output[: input_match.start()]

            # Find the last "Action:" before this "Action Input:"
            # Match action name up to newline or end of string
            all_actions = list(re.finditer(r"Action:\s*(.+?)(?=\n|$)", preceding_text))

            if not all_actions:
                continue

            action_match = all_actions[-1]
            action_name = action_match.group(1).strip()

            # Extract JSON starting after "Action Input:"
            json_str_candidate = output[input_match.end() :].strip()

            # Remove markdown code block markers first
            for marker in ["```json", "```JSON", "```"]:
                json_str_candidate = json_str_candidate.replace(marker, "").strip()

            # Manual JSON extraction with brace counting to handle nested structures
            brace_count = 0
            json_end = 0
            in_string = False
            escape = False
            found_start = False
            start_idx = 0

            for j, char in enumerate(json_str_candidate):
                if not found_start:
                    if char == "{":
                        found_start = True
                        start_idx = j
                        brace_count = 1
                    elif not char.isspace():
                        # Non-whitespace character before '{' means invalid format
                        break
                    continue

                if char == '"' and not escape:
                    in_string = not in_string

                if char == "\\" and not escape:
                    escape = True
                else:
                    escape = False

                if not in_string:
                    if char == "{":
                        brace_count += 1
                    elif char == "}":
                        brace_count -= 1
                        if brace_count == 0:
                            json_end = j + 1
                            break

            if json_end > 0:
                raw_input = json_str_candidate[start_idx:json_end]

                try:
                    action_input = json.loads(raw_input)
                    actions.append({"tool_name": action_name, "tool_input": action_input})
                except json.JSONDecodeError as e:
                    raise ActionParsingException(
                        f"Invalid JSON in Action Input for {action_name}: {str(e)} : {raw_input}",
                        recoverable=True,
                    )

        if not actions:
            logger.info("No valid Action and Action Input pairs found in the output ")
            raise ActionParsingException(
                "No valid Action and Action Input pairs found in the output.",
                recoverable=True,
            )

        if not parallel_tool_calls_enabled or len(actions) == 1:
            action = actions[0]["tool_name"]
            action_input = actions[0]["tool_input"]
            return thought, action, action_input
        else:
            return thought, "multiple_tools", actions

    except Exception as e:
        logger.error(f"Error: {e}")
        if isinstance(e, ActionParsingException):
            raise
        raise ActionParsingException(
            f"Error parsing action(s): {str(e)}. "
            f"Please ensure the output follows the format 'Thought: <text> "
            f"Action: <action> Action Input: <valid JSON>' "
            f"{'with possible multiple Action/Action Input pairs.' if parallel_tool_calls_enabled else ''}",
            recoverable=True,
        )


def extract_default_final_answer(output: str) -> tuple[str, str]:
    """
    Extracts the final thought and answer from the output string in default inference mode format.

    Args:
        output: The LLM output string containing Thought and Answer

    Returns:
        tuple: (thought, answer) where both are strings (empty strings if not found)
    """
    match = re.search(r"Thought:\s*(.*?)\s*Answer:\s*(.*)", output, re.DOTALL)
    if match:
        thought = match.group(1).strip()
        answer = match.group(2).strip()
        return thought, answer
    else:
        return "", ""
