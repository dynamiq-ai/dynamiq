import pytest

from dynamiq.utils.jsonpath import COALESCE_SEPARATOR, is_expression, mapper

AGENT_RAN = {
    "agent": {"status": "success", "output": {"content": "AGENT ANSWER"}},
    "error": {"status": "skip", "output": None},
}
ERROR_RAN = {
    "agent": {"status": "skip", "output": None},
    "error": {"status": "success", "output": {"output": "BLOCKED"}},
}
BOTH_RAN = {
    "agent": {"status": "success", "output": {"content": "AGENT ANSWER"}},
    "error": {"status": "success", "output": {"output": "BLOCKED"}},
}
NEITHER_RAN = {
    "agent": {"status": "skip", "output": None},
    "error": {"status": "skip", "output": None},
}

COALESCE = "$.agent.output.content || $.error.output.output"


def test_separator_is_double_pipe():
    assert COALESCE_SEPARATOR == "||"


def test_coalesce_takes_first_available():
    assert mapper(AGENT_RAN, {"output": COALESCE}) == {"output": "AGENT ANSWER"}


def test_coalesce_falls_through_to_second():
    assert mapper(ERROR_RAN, {"output": COALESCE}) == {"output": "BLOCKED"}


def test_coalesce_order_is_priority():
    """When more than one alternative resolves, the leftmost wins."""
    assert mapper(BOTH_RAN, {"output": COALESCE}) == {"output": "AGENT ANSWER"}
    reversed_expr = "$.error.output.output || $.agent.output.content"
    assert mapper(BOTH_RAN, {"output": reversed_expr}) == {"output": "BLOCKED"}


def test_coalesce_returns_none_when_nothing_resolves():
    assert mapper(NEITHER_RAN, {"output": COALESCE}) == {"output": None}


def test_coalesce_literal_tail_is_used_as_default():
    expr = "$.agent.output.content || $.error.output.output || no result"
    assert mapper(NEITHER_RAN, {"output": expr}) == {"output": "no result"}


def test_coalesce_literal_tail_ignored_when_a_path_resolves():
    expr = "$.agent.output.content || no result"
    assert mapper(AGENT_RAN, {"output": expr}) == {"output": "AGENT ANSWER"}


def test_coalesce_supports_more_than_two_alternatives():
    expr = "$.a.output.v || $.b.output.v || $.c.output.v"
    data = {"a": {"output": None}, "b": {"output": None}, "c": {"output": {"v": "C"}}}
    assert mapper(data, {"output": expr}) == {"output": "C"}


def test_coalesce_tolerates_whitespace():
    assert mapper(ERROR_RAN, {"output": "$.agent.output.content||$.error.output.output"}) == {"output": "BLOCKED"}
    assert mapper(ERROR_RAN, {"output": "$.agent.output.content   ||   $.error.output.output"}) == {"output": "BLOCKED"}


def test_coalesce_skips_empty_segments():
    assert mapper(ERROR_RAN, {"output": "$.agent.output.content || || $.error.output.output"}) == {"output": "BLOCKED"}


def test_coalesce_does_not_leak_the_expression_as_a_value():
    """A failed lookup must never surface the JSONPath expression itself as data."""
    result = mapper(NEITHER_RAN, {"output": COALESCE})
    assert COALESCE_SEPARATOR not in str(result["output"])
    assert "$." not in str(result["output"])


@pytest.mark.parametrize(
    "literal",
    [
        "a || b",
        "true || false",
        "yes||no",
        "Answer: A || B",
        "0 || 1",
        "x||y||z",
        "|| leading",
        "trailing ||",
        "spaced  ||  out",
        "||",
    ],
)
def test_literal_values_containing_the_separator_are_preserved(literal):
    """A value is only coalesced when an alternative is a rooted JSONPath.

    A bare word parses as a valid JSONPath, so without the rooting check plain text
    such as "a || b" would resolve to None and silently lose the user's value.
    """
    assert mapper(AGENT_RAN, {"output": literal}) == {"output": literal}


def test_is_expression_requires_a_root():
    assert is_expression("$.agent.output.content") is True
    assert is_expression("@.content") is True
    # Parses as a JSONPath but is not rooted, so it is a literal here.
    assert is_expression("agent") is False
    assert is_expression("a") is False
    assert is_expression("no result") is False
    # Never raises, whatever it is given.
    assert is_expression("{{ template }}") is False
    assert is_expression("$.broken[") is False
    assert is_expression(None) is False


def test_coalesce_needs_only_one_rooted_alternative():
    result = mapper(AGENT_RAN, {"output": f"$.missing.output.x {COALESCE_SEPARATOR} plain words"})

    assert result == {"output": "plain words"}


def test_plain_jsonpath_is_unchanged():
    assert mapper(AGENT_RAN, {"output": "$.agent.output.content"}) == {"output": "AGENT ANSWER"}


def test_plain_jsonpath_missing_still_returns_none():
    assert mapper(ERROR_RAN, {"output": "$.agent.output.content"}) == {"output": None}


def test_plain_literal_is_unchanged():
    assert mapper(AGENT_RAN, {"output": "just some text"}) == {"output": "just some text"}


def test_mixed_selector_keys():
    """Coalesced and plain entries can live side by side in one selector."""
    selector = {"output": COALESCE, "literal": "hello", "direct": "$.agent.output.content"}
    assert mapper(AGENT_RAN, selector) == {
        "output": "AGENT ANSWER",
        "literal": "hello",
        "direct": "AGENT ANSWER",
    }


def test_coalesce_returns_list_when_one_alternative_matches_many():
    data = {"a": {"output": {"v": [1, 2]}}, "items": [{"v": 1}, {"v": 2}]}
    assert mapper(data, {"output": "$.items[*].v || $.a.output.v"}) == {"output": [1, 2]}
