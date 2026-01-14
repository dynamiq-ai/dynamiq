"""Parsing logic for Agent LLM outputs in different formats."""

import json
import re

import regex

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

        action_pattern = r"Action:\s*(.*?)\nAction Input:\s*(\{(?:[^{}]|(?R))*\})"

        remaining_text = output
        actions = []

        while "Action:" in remaining_text:
            action_match = regex.search(action_pattern, remaining_text, re.DOTALL)
            if not action_match:
                break

            action_name = action_match.group(1).strip()
            raw_input = action_match.group(2).strip()

            for marker in ["```json", "```JSON", "```"]:
                raw_input = raw_input.replace(marker, "").strip()

            try:
                action_input = json.loads(raw_input.strip())
                actions.append({"tool_name": action_name, "tool_input": action_input})
            except json.JSONDecodeError as e:
                raise ActionParsingException(
                    f"Invalid JSON in Action Input for {action_name}: {str(e)} : {raw_input}",
                    recoverable=True,
                )

            end_pos = action_match.end()
            remaining_text = remaining_text[end_pos:]

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
