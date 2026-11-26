from enum import IntEnum

import numpy as np
import pytest

from dynamiq.storages.vector.milvus.filter import Filter


def test_build_filter_expression_with_none():
    with pytest.raises(TypeError):
        filter_instance = Filter(None)
        filter_instance.build_filter_expression()


def test_build_filter_expression_with_empty_dict():
    filter_instance = Filter({})
    with pytest.raises(ValueError, match="Invalid filter structure"):
        filter_instance.build_filter_expression()


def test_build_filter_expression_with_simple_comparison():
    filter_term = {"field": "age", "operator": "==", "value": 30}
    filter_instance = Filter(filter_term)
    expression = filter_instance.build_filter_expression()
    assert expression == "age == 30"


def test_build_filter_expression_with_string_value():
    filter_term = {"field": "name", "operator": "==", "value": "John"}
    filter_instance = Filter(filter_term)
    expression = filter_instance.build_filter_expression()
    assert expression == 'name == "John"'


def test_build_filter_expression_with_unsupported_operator():
    filter_term = {"field": "age", "operator": "%%", "value": 30}
    filter_instance = Filter(filter_term)
    with pytest.raises(ValueError, match="Unsupported comparison operator: %%"):
        filter_instance.build_filter_expression()


def test_build_filter_expression_with_logical_operator():
    filter_term = {
        "operator": "AND",
        "conditions": [{"field": "age", "operator": ">", "value": 20}, {"field": "age", "operator": "<", "value": 30}],
    }
    filter_instance = Filter(filter_term)
    expression = filter_instance.build_filter_expression()
    assert expression == "(age > 20 and age < 30)"


def test_build_filter_expression_with_nested_logical_operators():
    filter_term = {
        "operator": "OR",
        "conditions": [
            {
                "operator": "AND",
                "conditions": [
                    {"field": "age", "operator": ">=", "value": 20},
                    {"field": "age", "operator": "<=", "value": 30},
                ],
            },
            {"field": "name", "operator": "==", "value": "Alice"},
        ],
    }
    filter_instance = Filter(filter_term)
    expression = filter_instance.build_filter_expression()
    assert expression == '((age >= 20 and age <= 30) or name == "Alice")'


def test_build_filter_expression_with_in_operator():
    filter_term = {"field": "status", "operator": "in", "value": ["active", "pending"]}
    filter_instance = Filter(filter_term)
    expression = filter_instance.build_filter_expression()
    assert expression == 'status in ["active", "pending"]'


def test_build_filter_expression_with_not_in_operator():
    filter_term = {"field": "status", "operator": "not in", "value": ["inactive", "banned"]}
    filter_instance = Filter(filter_term)
    expression = filter_instance.build_filter_expression()
    assert expression == 'status not in ["inactive", "banned"]'


def test_build_filter_expression_with_contains_any_operator():
    filter_term = {"field": "tags", "operator": "contains_any", "value": ["ai", "ml"]}
    filter_instance = Filter(filter_term)
    expression = filter_instance.build_filter_expression()
    assert expression == 'json_contains_any(tags, ["ai", "ml"])'


def test_build_filter_expression_with_contains_any_non_list_raises():
    filter_term = {"field": "tags", "operator": "contains_any", "value": "ai"}
    filter_instance = Filter(filter_term)
    with pytest.raises(ValueError, match="must be a list"):
        filter_instance.build_filter_expression()


def test_build_filter_expression_with_invalid_structure():
    filter_term = {"operator": "AND", "field": "age", "value": 30}
    filter_instance = Filter(filter_term)
    with pytest.raises(ValueError, match="Unsupported comparison operator: AND"):
        filter_instance.build_filter_expression()


@pytest.mark.parametrize(
    "operator, value, expected",
    [
        ("==", 'a" or id > 0 or tenant == "', 'tenant == "a\\" or id > 0 or tenant == \\""'),
        ("==", "back\\slash", 'tenant == "back\\\\slash"'),
        ("in", ["a", '") or true or ("'], 'tenant in ["a", "\\") or true or (\\""]'),
        ("contains_any", ['"]) or true or (['], 'json_contains_any(tenant, ["\\"]) or true or (["])'),
        ("==", True, "tenant == True"),
        ("<", 1.5, "tenant < 1.5"),
        ("==", IntEnum("Level", "LOW HIGH").HIGH, "tenant == 2"),
        ("<", np.float64(0.5), "tenant < 0.5"),
        ("in", [np.int64(3), np.float32(0.25)], "tenant in [3, 0.25]"),
    ],
)
def test_build_filter_expression_escapes_values(operator, value, expected):
    expression = Filter({"field": "tenant", "operator": operator, "value": value}).build_filter_expression()
    assert expression == expected


@pytest.mark.parametrize("value", [None, {"a": 1}, float("nan"), np.float64("inf"), ["a", None]])
def test_build_filter_expression_rejects_unsupported_values(value):
    operator = "in" if isinstance(value, list) else "=="
    with pytest.raises(ValueError, match="Unsupported filter value"):
        Filter({"field": "tenant", "operator": operator, "value": value}).build_filter_expression()


@pytest.mark.parametrize(
    "field",
    [
        "info",
        '$meta["my-key"]',
        'info["k"]',
        "info[0]",
        '$meta["dc:title"]',
        '$meta["source/url"]',
        '$meta["назва"]',
        '$meta["say \\"hi\\""]',
    ],
)
def test_build_filter_expression_accepts_field_paths(field):
    assert Filter({"field": field, "operator": "==", "value": 1}).build_filter_expression() == f"{field} == 1"


@pytest.mark.parametrize(
    "field",
    [
        "id > 0 or tenant",
        'info["k"] or true',
        'info["k\\"] or true',
        '$meta["a"] or $meta["b"]',
        "",
        1,
    ],
)
def test_build_filter_expression_rejects_invalid_fields(field):
    with pytest.raises(ValueError, match="Invalid filter field"):
        Filter({"field": field, "operator": "==", "value": "x"}).build_filter_expression()
