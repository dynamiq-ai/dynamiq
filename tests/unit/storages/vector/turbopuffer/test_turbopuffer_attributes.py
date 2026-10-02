from datetime import date, datetime
from decimal import Decimal
from enum import Enum

import pytest

from dynamiq.storages.vector.turbopuffer.attributes import (
    coerce_value,
    infer_type,
    longest_string_bytes,
    normalize_value,
)


class Color(Enum):
    RED = "red"


class TextObject(str):
    """A str subclass, like the text objects PDF parsers return."""


@pytest.mark.parametrize(
    "value, expected",
    [
        (None, None),
        ("text", "text"),
        (TextObject("pdf text"), "pdf text"),
        (True, True),
        (3, 3),
        (3.5, 3.5),
        (float("nan"), None),
        (2**70, str(2**70)),
        (Decimal("1.5"), 1.5),
        (Color.RED, "red"),
        (datetime(2024, 1, 2, 3, 4, 5), "2024-01-02T03:04:05"),
        (date(2024, 1, 2), "2024-01-02"),
        (b"bytes", "bytes"),
        ({"w": 612, "h": 792}, '{"w": 612, "h": 792}'),
        ([], []),
        (["a", None, "b"], ["a", "b"]),
        ([1, 2], [1, 2]),
        ([1, 2.5], [1.0, 2.5]),
        (["x", 1, True], ["x", "1", "true"]),
        ([{"a": 1}], '[{"a": 1}]'),
        ((1, 2), [1, 2]),
    ],
)
def test_normalize_value(value, expected):
    assert normalize_value(value) == expected


@pytest.mark.parametrize(
    "value, expected",
    [
        ("text", "string"),
        (True, "bool"),
        (3, "int"),
        (3.0, "float"),
        ([], "[]string"),
        (["a"], "[]string"),
        ([1], "[]int"),
        ([1.5], "[]float"),
        ([True], "[]bool"),
    ],
)
def test_infer_type(value, expected):
    assert infer_type(value) == expected


@pytest.mark.parametrize(
    "value, attribute_type, expected",
    [
        (5, "string", "5"),
        (True, "string", "true"),
        (["a", 1], "string", '["a", 1]'),
        ("2", "int", 2),
        (2.0, "int", 2),
        (7.5, "int", None),
        (True, "int", None),
        ("x", "int", None),
        (-1, "uint", None),
        (3, "float", 3.0),
        ("3.5", "float", 3.5),
        ("not-a-number", "float", None),
        ("TRUE", "bool", True),
        ("yes", "bool", None),
        ("2024-01-01T00:00:00Z", "datetime", "2024-01-01T00:00:00Z"),
        ("a", "[]string", ["a"]),
        ([1, "x"], "[]int", [1]),
        (["public"], "[]unknown", ["public"]),
        ("public", "[]unknown", None),
        ({"w": 1.0}, "{}f16", None),
    ],
)
def test_coerce_value(value, attribute_type, expected):
    assert coerce_value(value, attribute_type) == expected


def test_longest_string_bytes():
    assert longest_string_bytes("é" * 3) == 6
    assert longest_string_bytes(["a", "bbb"]) == 3
    assert longest_string_bytes(5) == 0
