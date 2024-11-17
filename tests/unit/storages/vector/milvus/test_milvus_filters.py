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
    assert expression == "status in ['active', 'pending']"


def test_build_filter_expression_with_not_in_operator():
    filter_term = {"field": "status", "operator": "not in", "value": ["inactive", "banned"]}
    filter_instance = Filter(filter_term)
    expression = filter_instance.build_filter_expression()
    assert expression == "status not in ['inactive', 'banned']"


def test_build_filter_expression_with_invalid_structure():
    filter_term = {"operator": "AND", "field": "age", "value": 30}
    filter_instance = Filter(filter_term)
    with pytest.raises(ValueError, match="Unsupported comparison operator: AND"):
        filter_instance.build_filter_expression()
