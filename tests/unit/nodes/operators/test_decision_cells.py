import pytest

from dynamiq.nodes.operators.decision_table import (
    coerce_value,
    compile_condition,
    parse_literal,
    read_literal,
    split_alternatives,
)


def matches(cell: str, value, column_type: str = "Any") -> bool:
    return compile_condition(cell, column_type, "cell")(coerce_value(value, column_type))


@pytest.mark.parametrize(
    ("text", "literal"),
    [
        ("700", 700),
        ("-0.8", -0.8),
        ("FHA", "FHA"),
        ("true", True),
        ("false", False),
        ('"700"', "700"),
        ("'a, b'", "a, b"),
        ("", None),
        ("   ", None),
    ],
)
def test_parse_literal(text, literal):
    assert parse_literal(text) == literal


def test_split_alternatives_keeps_quoted_commas():
    assert split_alternatives("\"a, b\", c, 'd,e'") == ['"a, b"', " c", " 'd,e'"]


@pytest.mark.parametrize("cell", ["", "   ", "*"])
@pytest.mark.parametrize("value", [None, 0, "", "anything", False])
def test_empty_cell_and_star_match_anything(cell, value):
    assert matches(cell, value)


@pytest.mark.parametrize(
    ("cell", "value", "expected"),
    [
        ("700", 700, True),
        ("700", 700.0, True),
        ("700", "700", False),
        ("FHA", "FHA", True),
        ("FHA", "fha", False),
        ("true", True, True),
        ("true", 1, False),
        ("1", True, False),
        ('"700"', "700", True),
        ("= FHA", "FHA", True),
        ("== FHA", "FHA", True),
        (">= 620", 620, True),
        (">= 620", 619, False),
        ("> 0.8", 0.81, True),
        ("<= -5", -5, True),
        ("< 1", 1, False),
        ("!= VA", "FHA", True),
        ("!= VA", "VA", False),
        (">= 2024-01-01", "2024-06-30", True),
        ("> 10", "abc", False),
    ],
)
def test_comparisons(cell, value, expected):
    assert matches(cell, value) is expected


@pytest.mark.parametrize(
    ("cell", "value", "expected"),
    [
        ("[620..680]", 620, True),
        ("[620..680]", 680, True),
        ("[620..680]", 681, False),
        ("(0..1]", 0, False),
        ("(0..1]", 1, True),
        ("[0.5 .. 0.8)", 0.8, False),
        ("[0.5 .. 0.8)", 0.5, True),
        ("[1..2]", "1", False),
        ("[1..2]", True, False),
    ],
)
def test_ranges(cell, value, expected):
    assert matches(cell, value) is expected


@pytest.mark.parametrize(
    ("cell", "value", "expected"),
    [
        ("FHA, VA, USDA", "VA", True),
        ("FHA, VA, USDA", "Conventional", False),
        ('1, 2, "3"', 2, True),
        ('1, 2, "3"', "3", True),
        ('1, 2, "3"', 3, False),
    ],
)
def test_lists(cell, value, expected):
    assert matches(cell, value) is expected


@pytest.mark.parametrize("cell", ["700", "!= 700", "> 1", "[1..2]", "1, 2"])
def test_missing_value_matches_only_an_empty_cell(cell):
    assert not matches(cell, None)


def test_column_type_reads_the_value_and_the_cell():
    assert matches(">= 650", "700", "int")
    assert not matches(">= 650", "abc", "int")
    assert not matches(">= 650", True, "int")
    assert matches("700", 700, "string")
    assert matches("true", "TRUE", "bool")
    assert not matches("true", "yes", "bool")
    assert matches("1", 1.0, "float")


@pytest.mark.parametrize(
    ("cell", "column_type", "message"),
    [
        (">=", "Any", ">= needs a value to compare with"),
        ("[10..1]", "Any", "runs backwards"),
        ("FHA, , VA", "Any", "cannot contain an empty alternative"),
        ("abc", "int", "expected a number"),
        ("[1..2]", "string", "needs a numeric column"),
        ("yes", "bool", "expected true or false"),
    ],
)
def test_malformed_cells_are_configuration_errors(cell, column_type, message):
    with pytest.raises(ValueError, match=message):
        compile_condition(cell, column_type, "cell")


@pytest.mark.parametrize(
    ("text", "column_type", "literal"),
    [
        ("0.25", "int", 0.25),
        ("TRUE", "bool", True),
        ("700", "string", "700"),
        ('"quoted"', "string", "quoted"),
        ("700", "Any", 700),
        ("approve", "Any", "approve"),
    ],
)
def test_output_literals_follow_the_column_type(text, column_type, literal):
    assert read_literal(text, column_type, "cell") == literal


def test_a_quote_opens_an_alternative_only_at_its_start():
    assert split_alternatives("O'Brien, Smith") == ["O'Brien", " Smith"]
    assert split_alternatives("Smith, O'Brien") == ["Smith", " O'Brien"]
    assert split_alternatives('"a, b", c') == ['"a, b"', " c"]
    matches = compile_condition("O'Brien, Smith", "string", "cell")
    assert matches("O'Brien") and matches("Smith") and not matches("O'Brien, Smith")
