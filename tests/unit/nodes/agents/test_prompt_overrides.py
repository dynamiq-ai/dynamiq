"""Model-specific prompt overrides keep the rules the parser and the loop rely on."""

import logging

import pytest

from dynamiq.nodes.agents.prompts.overrides._sections import with_base_sections
from dynamiq.nodes.agents.prompts.react import instructions as base
from dynamiq.nodes.agents.prompts.registry import get_registry

# Per base prompt, the sections every override must have word for word.
SHARED_SECTIONS = {
    "REACT_BLOCK_INSTRUCTIONS_SINGLE": ("## Single Action Per Turn",),
    "REACT_BLOCK_XML_INSTRUCTIONS_SINGLE": ("## Single Action Per Turn", "## JSON Formatting Requirements"),
}

OVERRIDES = [
    (model, name, prompt)
    for model, prompts in get_registry()._registry.items()
    for name, prompt in prompts.items()
    if name in SHARED_SECTIONS
]


def section(prompt: str, heading: str) -> str:
    """The text from the heading's line to the next heading's line, cut with plain string indexes."""
    start = prompt.index(f"\n{heading}\n") + 1
    end = prompt.find("\n#", start)
    return prompt[start : end + 1] if end != -1 else prompt[start:]


@pytest.mark.parametrize("model, name, prompt", OVERRIDES, ids=[f"{m}:{n}" for m, n, _ in OVERRIDES])
def test_an_override_keeps_the_base_rules_for_one_action_per_turn(model, name, prompt):
    # The base XML rules forbid writing the Observation and a second block: what reasoning models,
    # which ignore stop sequences, need most. A copied override had fallen behind them.
    default = getattr(base, name)
    for heading in SHARED_SECTIONS[name]:
        assert default.splitlines().count(heading) == 1, heading
        assert section(default, heading) in prompt, (model, heading)


def test_gpt_and_gemini_overrides_are_registered():
    models = {model for model, _, _ in OVERRIDES}
    assert {"gpt-5.1", "gemini-3-pro-preview"} <= models


SINGLE_ACTION = ("## Single Action Per Turn",)


def test_a_heading_line_in_fenced_code_does_not_end_the_section():
    fenced = "```\n## Not a heading\n```\n"
    base_prompt = f"## Single Action Per Turn\n- New rule.\n{fenced}- Last rule.\n\n## File Handling\n- Base.\n"
    prompt = "## Single Action Per Turn\n- Old rule.\n\n## File Handling\n- Override.\n"

    assert with_base_sections(prompt, base_prompt, SINGLE_ACTION) == (
        f"## Single Action Per Turn\n- New rule.\n{fenced}- Last rule.\n\n## File Handling\n- Override.\n"
    )


def test_the_heading_in_a_fenced_example_is_not_taken_for_the_section():
    base_prompt = "## Single Action Per Turn\n- New rule.\n\n## File Handling\n"
    example = "Example:\n```\n## Single Action Per Turn\n- Example rule.\n```\n\n"
    prompt = f"{example}## Single Action Per Turn\n- Old rule.\n\n## File Handling\n"

    assert with_base_sections(prompt, base_prompt, SINGLE_ACTION) == (
        f"{example}## Single Action Per Turn\n- New rule.\n\n## File Handling\n"
    )


def test_a_level_one_section_after_the_section_is_kept():
    base_prompt = "## Single Action Per Turn\n- New rule.\n\n# Appendix\nBase appendix.\n"
    prompt = "## Single Action Per Turn\n- Old rule.\n\n# Appendix\nOverride appendix.\n"

    assert with_base_sections(prompt, base_prompt, SINGLE_ACTION) == (
        "## Single Action Per Turn\n- New rule.\n\n# Appendix\nOverride appendix.\n"
    )


@pytest.mark.parametrize(
    "base_prompt, prompt",
    [
        pytest.param(
            "## One Action Per Turn\n- New rule.\n",
            "## Single Action Per Turn\n- Old rule.\n",
            id="renamed in the base",
        ),
        pytest.param(
            "## Single Action Per Turn\n- New rule.\n",
            "Example:\n## Single Action Per Turn\n\n## Single Action Per Turn\n- Old rule.\n",
            id="twice in the override",
        ),
        pytest.param(
            "## Single Action Per Turn\n- New rule.\n",
            "## One Action Per Turn\n- Old rule.\n",
            id="missing from the override",
        ),
    ],
)
def test_a_heading_not_once_in_each_prompt_leaves_the_override_as_it_is_with_a_warning(base_prompt, prompt, caplog):
    # Raising instead would break importing the agents, which builds the overrides.
    with caplog.at_level(logging.WARNING):
        assert with_base_sections(prompt, base_prompt, SINGLE_ACTION) == prompt

    assert "'## Single Action Per Turn'" in caplog.text
