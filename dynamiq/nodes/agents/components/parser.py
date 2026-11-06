"""Parsing logic for Agent LLM outputs in different formats."""

import json
import re

from dynamiq.nodes.agents.exceptions import ActionParsingException
from dynamiq.utils.logger import logger

_ACTION_LABEL_RE = re.compile(r"^[ \t]*Action:", re.MULTILINE)
_ACTION_INPUT_LABEL_RE = re.compile(r"^[ \t]*Action Input:", re.MULTILINE)
_ANSWER_LABEL_RE = re.compile(r"^[ \t]*Answer:", re.MULTILINE)
_OBSERVATION_LABEL_RE = re.compile(r"^[ \t]*Observation:", re.MULTILINE)
_LABELS = ("Thought", "Action Input", "Action", "Final Answer", "Answer", "Output Files", "Observation")
# A label written in bold or italics ("**Action:**", "**Action**:"), after a heading mark ("### Action:"), or in
# another case ("action:"); and a label written as a heading alone on its line ("### Action").
_DECORATED_LABEL_RE = re.compile(
    rf"^([ \t]*)(?:#{{1,6}}[ \t]+)?([*_]{{1,2}})?({'|'.join(_LABELS)})\2?[ \t]*:(?:[*_]{{1,2}})?",
    re.IGNORECASE | re.MULTILINE,
)
_HEADING_LABEL_RE = re.compile(rf"^([ \t]*)#{{1,6}}[ \t]+({'|'.join(_LABELS)})[ \t]*$", re.IGNORECASE | re.MULTILINE)


def normalize_default_labels(output: str) -> str:
    """``output`` with its Thought/Action/Action Input/Answer labels in the plain form the parser reads."""

    def plain(indent: str, label: str) -> str:
        return indent + next(name for name in _LABELS if name.lower() == label.lower()) + ":"

    output = _HEADING_LABEL_RE.sub(lambda match: plain(match.group(1), match.group(2)), output)
    return _DECORATED_LABEL_RE.sub(lambda match: plain(match.group(1), match.group(3)), output)


def _default_action_label(output: str) -> re.Match | None:
    """The reply's ``Action:`` label when an ``Action Input:`` follows it, which makes it a tool call.

    Models write ``Action: None`` or ``Action: Final Answer`` before their answer; with no input to
    run, that line does not make the reply an action.
    """
    action = _ACTION_LABEL_RE.search(output)
    return action if action and _ACTION_INPUT_LABEL_RE.search(output, action.end()) else None


def has_default_action_input(output: str) -> bool:
    """Whether a DEFAULT-mode reply has reached an ``Action Input:`` label, after which text is the tool's input."""
    return _default_action_label(output) is not None


def is_default_final_answer(output: str) -> bool:
    """Whether a DEFAULT-mode reply is a final answer rather than an action.

    An ``Answer:`` label counts only when no tool call (an ``Action:`` with an ``Action Input:``) comes
    before it, so a tool input that contains the text "Answer:" (a FAQ, a test fixture) is still run as
    an action. A reply with neither label at the start of a line falls back to looking for "Answer:"
    anywhere, as before.
    """
    answer = _ANSWER_LABEL_RE.search(output)
    action = _default_action_label(output)
    if answer and action:
        return answer.start() < action.start()
    if answer or action:
        return bool(answer)
    return "Answer:" in output


def cut_default_reply_at_observation(output: str) -> str:
    """The reply up to an ``Observation:`` the model wrote after its action, which it must not write.

    ``Observation:`` is the DEFAULT mode's stop sequence, but reasoning models ignore stop sequences and
    go on to invent the tool result and a final answer.
    """
    action = _default_action_label(output)
    if not action:
        return output
    observation = _OBSERVATION_LABEL_RE.search(output, action.end())
    return output[: observation.start()].rstrip() if observation else output


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


def parse_default_action(output: str) -> tuple[str | None, str | None, dict | list | None]:
    """
    Parses the action(s), input(s), and thought from the output string in default inference mode format.

    Supports both single tool actions and multiple sequential tool calls
    when multi-tool is enabled.

    Args:
        output: The output string from the LLM containing Thought, Action, and Action Input
        parallel_tool_calls_enabled: Whether parallel tool calls are enabled

    Returns:
        tuple: (thought, action, action_input) where:
            - thought is the extracted reasoning
            - action is the tool name (first action if multiple are found)
            - action_input is the tool input dict

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
            all_actions = list(re.finditer(r"Action:\s*(.+?)(?=\n|$|Action)", preceding_text))

            if not all_actions:
                continue

            action_match = all_actions[-1]
            action_name = action_match.group(1).strip()

            json_str_candidate = output[input_match.end() :].strip()

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
                    # strict=False: models put raw newlines and tabs inside JSON strings (code, multi-line text).
                    action_input = json.loads(raw_input, strict=False)
                    actions.append({"tool_name": action_name, "tool_input": action_input})
                except json.JSONDecodeError as e:
                    raise ActionParsingException(
                        f"Invalid JSON in Action Input for {action_name}: {str(e)} : {raw_input}",
                        recoverable=True,
                    )
            else:
                raise ActionParsingException(
                    f"Could not parse JSON structure in Action Input for {action_name}. "
                    f"Expected JSON object starting with '{{' but found: "
                    f"{json_str_candidate[:100]}{'...' if len(json_str_candidate) > 100 else ''}",
                    recoverable=True,
                )

        if not actions:
            logger.info("No valid Action and Action Input pairs found in the output ")
            raise ActionParsingException(
                "No valid Action and Action Input pairs found in the output.",
                recoverable=True,
            )

        action = actions[0]["tool_name"]
        action_input = actions[0]["tool_input"]
        return thought, action, action_input

    except Exception as e:
        logger.warning(f"Agent reply could not be parsed: {e}")
        if isinstance(e, ActionParsingException):
            raise
        raise ActionParsingException(
            f"Error parsing action(s): {str(e)}. "
            f"Please ensure the output follows the format 'Thought: <text> "
            f"Action: <action> Action Input: <valid JSON>' ",
            recoverable=True,
        )


def extract_default_final_answer(output: str) -> tuple[str, str, str]:
    """
    Extracts the final thought, optional output files, and answer from
    the output string in default inference mode format.

    The expected format is::

        Thought: <reasoning>
        Output Files: <comma-separated paths>   (optional)
        Answer: <response>

    Args:
        output: The LLM output string containing Thought and Answer

    Returns:
        tuple: (thought, answer, output_files_raw) where all are strings
               (empty strings if not found). *output_files_raw* is the raw
               comma-separated value, or "" when the field is absent.
    """
    match = re.search(r"Thought:\s*(.*?)\s*Answer:\s*(.*)", output, re.DOTALL)
    if not match:
        # An answer without a thought is still the answer.
        bare = re.search(r"Answer:\s*(.*)", output, re.DOTALL)
        return "", (bare.group(1).strip() if bare else ""), ""

    raw_thought = match.group(1).strip()
    answer = match.group(2).strip()

    files_match = re.search(r"^(.*)\nOutput Files:\s*(.*)$", raw_thought, re.DOTALL)
    if files_match:
        thought = files_match.group(1).strip()
        output_files_raw = files_match.group(2).strip()
    else:
        thought = raw_thought
        output_files_raw = ""

    return thought, answer, output_files_raw
