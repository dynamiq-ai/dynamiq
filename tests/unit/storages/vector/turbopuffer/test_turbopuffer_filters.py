from datetime import datetime
from decimal import Decimal
from enum import Enum

import numpy as np
import pytest

from dynamiq.storages.vector.exceptions import VectorStoreFilterException
from dynamiq.storages.vector.turbopuffer.filters import (
    FILTER_ALL,
    FILTER_NONE,
    MATCH_ALL,
    MATCH_NONE,
    combine_and,
    convert_filters,
    referenced_fields,
    to_turbopuffer_filter,
)


class Color(Enum):
    RED = "red"


SCHEMA = {
    "id": {"type": "string"},
    "s": {"type": "string", "filterable": True},
    "num": {"type": "int", "filterable": True},
    "tags": {"type": "[]string", "filterable": True},
    "content": {"type": "string", "filterable": False, "full_text_search": {"tokenizer": "word_v4"}},
}


@pytest.mark.parametrize(
    "filters, expected",
    [
        ({"field": "s", "operator": "==", "value": "a"}, ["s", "Eq", "a"]),
        ({"field": "s", "operator": "==", "value": None}, ["s", "Eq", None]),
        ({"field": "s", "operator": "!=", "value": "a"}, ["s", "NotEq", "a"]),
        ({"field": "num", "operator": ">", "value": 5}, ["num", "Gt", 5]),
        ({"field": "num", "operator": ">=", "value": 5}, ["num", "Gte", 5]),
        ({"field": "num", "operator": "<", "value": 5}, ["And", [["num", "NotEq", None], ["num", "Lt", 5]]]),
        ({"field": "num", "operator": "<=", "value": 5}, ["And", [["num", "NotEq", None], ["num", "Lte", 5]]]),
        ({"field": "num", "operator": ">", "value": None}, MATCH_NONE),
        ({"field": "s", "operator": "in", "value": ["a", "b"]}, ["s", "In", ["a", "b"]]),
        ({"field": "s", "operator": "in", "value": []}, MATCH_NONE),
        ({"field": "s", "operator": "not in", "value": ["a"]}, ["s", "NotIn", ["a"]]),
        ({"field": "s", "operator": "contains_any", "value": ["a"]}, ["s", "In", ["a"]]),
        ({"field": "s", "operator": "contains_all", "value": ["a", "a"]}, ["s", "Eq", "a"]),
        ({"field": "s", "operator": "contains_all", "value": ["a", "b"]}, MATCH_NONE),
        ({"field": "metadata.s", "operator": "==", "value": "a"}, ["s", "Eq", "a"]),
    ],
)
def test_convert_scalar_conditions(filters, expected):
    assert convert_filters(filters, SCHEMA) == expected


@pytest.mark.parametrize(
    "filters, expected",
    [
        ({"field": "tags", "operator": "==", "value": "x"}, ["tags", "Contains", "x"]),
        ({"field": "tags", "operator": "!=", "value": "x"}, ["tags", "NotContains", "x"]),
        ({"field": "tags", "operator": "in", "value": ["x", "y"]}, ["tags", "ContainsAny", ["x", "y"]]),
        ({"field": "tags", "operator": "not in", "value": ["x"]}, ["tags", "NotContainsAny", ["x"]]),
        ({"field": "tags", "operator": "contains_any", "value": ["x"]}, ["tags", "ContainsAny", ["x"]]),
        (
            {"field": "tags", "operator": "contains_all", "value": ["x", "y"]},
            ["And", [["tags", "Contains", "x"], ["tags", "Contains", "y"]]],
        ),
        ({"field": "tags", "operator": "contains_all", "value": []}, MATCH_ALL),
    ],
)
def test_convert_array_conditions(filters, expected):
    assert convert_filters(filters, SCHEMA) == expected


@pytest.mark.parametrize(
    "filters, expected",
    [
        ({"field": "missing", "operator": "==", "value": "a"}, MATCH_NONE),
        ({"field": "missing", "operator": "==", "value": None}, MATCH_ALL),
        ({"field": "missing", "operator": "!=", "value": "a"}, MATCH_ALL),
        ({"field": "missing", "operator": "<", "value": 1}, MATCH_NONE),
        ({"field": "missing", "operator": "not in", "value": ["a"]}, MATCH_ALL),
        ({"field": "missing", "operator": "contains_any", "value": ["a"]}, MATCH_NONE),
    ],
)
def test_convert_missing_attribute(filters, expected):
    assert convert_filters(filters, SCHEMA) == expected


@pytest.mark.parametrize("operator, value", [("==", "a"), ("!=", "a"), ("in", ["a"]), ("not in", ["a"])])
def test_convert_rejects_unfilterable_attribute(operator, value):
    with pytest.raises(VectorStoreFilterException, match="not filterable"):
        convert_filters({"field": "content", "operator": operator, "value": value}, SCHEMA)


def test_referenced_fields():
    filters = {
        "operator": "AND",
        "conditions": [
            {"field": "metadata.s", "operator": "==", "value": "a"},
            {"operator": "NOT", "conditions": [{"field": "num", "operator": "<", "value": 1}]},
        ],
    }

    assert referenced_fields(filters) == {"s", "num"}
    assert referenced_fields({"file_id": ["f1"]}) == {"file_id"}
    assert referenced_fields(None) == set()


@pytest.mark.parametrize(
    "filters, expected",
    [
        ({"field": "num", "operator": "==", "value": "5"}, ["num", "Eq", 5]),
        ({"field": "num", "operator": "==", "value": "five"}, MATCH_NONE),
        ({"field": "num", "operator": "!=", "value": "five"}, MATCH_ALL),
        ({"field": "num", "operator": "in", "value": ["1", "x"]}, ["num", "In", [1]]),
        ({"field": "s", "operator": "==", "value": 5}, ["s", "Eq", "5"]),
        ({"field": "s", "operator": "==", "value": datetime(2024, 1, 1)}, ["s", "Eq", "2024-01-01T00:00:00"]),
        ({"field": "s", "operator": "==", "value": Color.RED}, ["s", "Eq", "red"]),
        ({"field": "num", "operator": "!=", "value": Decimal(3)}, ["num", "NotEq", 3]),
        ({"field": "num", "operator": "==", "value": np.int64(3)}, ["num", "Eq", 3]),
        ({"field": "num", "operator": "in", "value": [Decimal(1), np.float64(2.0)]}, ["num", "In", [1, 2]]),
    ],
)
def test_convert_coerces_values_to_attribute_type(filters, expected):
    assert convert_filters(filters, SCHEMA) == expected


def test_convert_logical_conditions():
    filters = {
        "operator": "AND",
        "conditions": [
            {"field": "tags", "operator": "contains_any", "value": ["public"]},
            {
                "operator": "OR",
                "conditions": [
                    {"field": "s", "operator": "==", "value": "a"},
                    {"field": "missing", "operator": "==", "value": "b"},
                ],
            },
            {"operator": "NOT", "conditions": [{"field": "num", "operator": "==", "value": 1}]},
        ],
    }

    assert convert_filters(filters, SCHEMA) == [
        "And",
        [["tags", "ContainsAny", ["public"]], ["s", "Eq", "a"], ["Not", ["num", "Eq", 1]]],
    ]


def test_convert_and_with_unmatchable_condition():
    filters = {
        "operator": "AND",
        "conditions": [
            {"field": "s", "operator": "==", "value": "a"},
            {"field": "missing", "operator": "==", "value": "b"},
        ],
    }

    assert convert_filters(filters, SCHEMA) is MATCH_NONE


def test_convert_simple_filters():
    assert convert_filters({"file_id": ["f1", "f2"], "s": "a"}) == [
        "And",
        [["file_id", "In", ["f1", "f2"]], ["s", "Eq", "a"]],
    ]


def test_convert_without_schema_trusts_attributes():
    assert convert_filters({"field": "anything", "operator": "==", "value": 1}) == ["anything", "Eq", 1]


def test_convert_empty_filters():
    assert convert_filters(None) is MATCH_ALL
    assert convert_filters({}) is MATCH_ALL


@pytest.mark.parametrize(
    "filters",
    [
        "not a dict",
        {"field": "s", "operator": "in", "value": "a"},
        {"field": "s", "operator": "~", "value": "a"},
        {"operator": "AND", "conditions": [{"field": "s", "value": "a"}]},
        {"operator": "XOR", "conditions": []},
    ],
)
def test_convert_invalid_filters(filters):
    with pytest.raises(VectorStoreFilterException):
        convert_filters(filters, SCHEMA)


def test_combine_and():
    assert combine_and(MATCH_ALL, None) is MATCH_ALL
    assert combine_and(["s", "Eq", "a"], MATCH_ALL) == ["s", "Eq", "a"]
    assert combine_and(["s", "Eq", "a"], MATCH_NONE) is MATCH_NONE
    assert combine_and(["s", "Eq", "a"], ["id", "Gt", "x"]) == ["And", [["s", "Eq", "a"], ["id", "Gt", "x"]]]


def test_to_turbopuffer_filter():
    assert to_turbopuffer_filter(MATCH_ALL) == FILTER_ALL
    assert to_turbopuffer_filter(MATCH_NONE) == FILTER_NONE
    assert to_turbopuffer_filter(["s", "Eq", "a"]) == ["s", "Eq", "a"]
