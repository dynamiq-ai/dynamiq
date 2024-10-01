import pytest
from qdrant_client.http import models

from dynamiq.storages.vector.exceptions import VectorStoreFilterException as FilterError
from dynamiq.storages.vector.qdrant.filters import (
    _build_eq_condition,
    _build_gt_condition,
    _build_gte_condition,
    _build_in_condition,
    _build_lt_condition,
    _build_lte_condition,
    _build_ne_condition,
    _build_nin_condition,
    _parse_comparison_operation,
    build_filters_for_repeated_operators,
    convert_filters_to_qdrant,
    is_iso_datetime_string,
)


def test_convert_filters_to_qdrant_with_none():
    assert convert_filters_to_qdrant(None) is None


def test_convert_filters_to_qdrant_with_models_filter():
    filter_term = models.Filter(
        must=[models.FieldCondition(key="metadata.field", match=models.MatchValue(value="test"))]
    )
    result = convert_filters_to_qdrant(filter_term)
    assert result.model_dump() == filter_term.model_dump()


def test_convert_filters_to_qdrant_with_empty_dict():
    assert convert_filters_to_qdrant({}) is None


def test_convert_filters_to_qdrant_with_logical_operators():
    filter_term = [
        {"operator": "AND", "conditions": [{"operator": "==", "field": "field1", "value": "value1"}]},
        {"operator": "OR", "conditions": [{"operator": "==", "field": "field2", "value": "value2"}]},
    ]
    result = convert_filters_to_qdrant(filter_term)
    assert isinstance(result, models.Filter)


def test_convert_filters_to_qdrant_with_comparison_operators():
    filter_term = {"operator": "==", "field": "field", "value": "value"}
    result = convert_filters_to_qdrant(filter_term)
    expected = models.Filter(must=[models.FieldCondition(key="metadata.field", match=models.MatchValue(value="value"))])
    assert result.model_dump() == expected.model_dump()


def test_convert_filters_to_qdrant_with_invalid_operator():
    filter_term = {"operator": "INVALID", "field": "field", "value": "value"}
    with pytest.raises(FilterError):
        convert_filters_to_qdrant(filter_term)


def test_build_filters_for_repeated_operators():
    must_clauses = [
        [models.Filter(must=[models.FieldCondition(key="metadata.field", match=models.MatchValue(value="test"))])]
    ]
    should_clauses = []
    must_not_clauses = []
    qdrant_filter = []
    result = build_filters_for_repeated_operators(must_clauses, should_clauses, must_not_clauses, qdrant_filter)
    assert isinstance(result, list)
    assert len(result) == 1
    assert isinstance(result[0], models.Filter)


def test_parse_comparison_operation():
    result = _parse_comparison_operation("==", "metadata.field", "value")
    expected = [models.FieldCondition(key="metadata.field", match=models.MatchValue(value="value"))]
    assert isinstance(result, list)
    assert result[0].model_dump() == expected[0].model_dump()


def test_build_eq_condition():
    result = _build_eq_condition("metadata.field", "value")
    expected = models.FieldCondition(key="metadata.field", match=models.MatchValue(value="value"))
    assert result.model_dump() == expected.model_dump()


def test_build_in_condition():
    result = _build_in_condition("metadata.field", ["value1", "value2"])
    expected = models.Filter(
        should=[
            models.FieldCondition(key="metadata.field", match=models.MatchText(text="value1")),
            models.FieldCondition(key="metadata.field", match=models.MatchText(text="value2")),
        ]
    )
    assert result.model_dump() == expected.model_dump()


def test_build_ne_condition():
    result = _build_ne_condition("metadata.field", "value")
    expected = models.Filter(
        must_not=[models.FieldCondition(key="metadata.field", match=models.MatchText(text="value"))]
    )
    assert result.model_dump() == expected.model_dump()


def test_build_nin_condition():
    result = _build_nin_condition("metadata.field", ["value1", "value2"])
    expected = models.Filter(
        must_not=[
            models.FieldCondition(
                key="metadata.field",
                match=models.MatchValue(value="value1"),
                range=None,
                geo_bounding_box=None,
                geo_radius=None,
                geo_polygon=None,
                values_count=None,
            ),
            models.FieldCondition(
                key="metadata.field",
                match=models.MatchValue(value="value2"),
                range=None,
                geo_bounding_box=None,
                geo_radius=None,
                geo_polygon=None,
                values_count=None,
            ),
        ],
        min_should=None,
        must=None,
        should=None,
    )
    print("test_build_nin_condition actual:", result.model_dump())
    print("test_build_nin_condition expected:", expected.model_dump())
    assert result.model_dump() == expected.model_dump()


def test_build_lt_condition():
    result = _build_lt_condition("metadata.field", 10)
    expected = models.FieldCondition(key="metadata.field", range=models.Range(lt=10))
    assert result.model_dump() == expected.model_dump()


def test_build_lte_condition():
    result = _build_lte_condition("metadata.field", 10)
    expected = models.FieldCondition(key="metadata.field", range=models.Range(lte=10))
    assert result.model_dump() == expected.model_dump()


def test_build_gt_condition():
    result = _build_gt_condition("metadata.field", 10)
    expected = models.FieldCondition(key="metadata.field", range=models.Range(gt=10))
    assert result.model_dump() == expected.model_dump()


def test_build_gte_condition():
    result = _build_gte_condition("metadata.field", 10)
    expected = models.FieldCondition(key="metadata.field", range=models.Range(gte=10))
    assert result.model_dump() == expected.model_dump()


def test_is_datetime_string_with_valid_datetime():
    assert is_iso_datetime_string("2023-01-01T00:00:00")


def test_is_datetime_string_with_invalid_datetime():
    assert not is_iso_datetime_string("invalid-datetime")
