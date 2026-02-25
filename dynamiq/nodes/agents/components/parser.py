"""Parsing logic for Agent LLM outputs (deprecated).

Agents use function calling only; tool schema is the single source of truth.
These helpers are kept for backward compatibility but raise if used.
"""


def parse_default_thought(output: str) -> str:
    """Deprecated. Use function calling mode; text parsing was removed."""
    raise DeprecationWarning(
        "parse_default_thought is deprecated. Agents use function calling only; "
        "do not rely on text/XML parsing."
    )


def parse_default_action(output: str) -> tuple[str | None, str | None, dict | list | None]:
    """Deprecated. Use function calling mode; text parsing was removed."""
    raise DeprecationWarning(
        "parse_default_action is deprecated. Agents use function calling only; "
        "do not rely on text/XML parsing."
    )


def extract_default_final_answer(output: str) -> tuple[str, str]:
    """Deprecated. Use function calling mode; text parsing was removed."""
    raise DeprecationWarning(
        "extract_default_final_answer is deprecated. Agents use function calling only; "
        "do not rely on text/XML parsing."
    )
