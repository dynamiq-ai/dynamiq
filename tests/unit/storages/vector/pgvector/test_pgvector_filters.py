import pytest

from dynamiq.storages.vector.exceptions import VectorStoreFilterException
from dynamiq.storages.vector.pgvector.filters import _contains_any, _convert_filters_to_query

EXPECTED_QUERY = (
    "EXISTS (SELECT 1 FROM jsonb_array_elements_text("
    "CASE WHEN jsonb_typeof(metadata->%s) = 'array' THEN metadata->%s ELSE '[]'::jsonb END"
    ") AS elem WHERE elem = ANY(%s::text[]))"
)


def test_contains_any_builds_jsonb_membership_query():
    query, params = _contains_any("metadata.tags", ["ai", "ml"])
    assert query == EXPECTED_QUERY
    assert params == ["tags", "tags", ["ai", "ml"]]


def test_contains_any_coerces_values_to_text_for_numeric_arrays():
    _, params = _contains_any("metadata.codes", [1, 2])
    assert params == ["codes", "codes", ["1", "2"]]


def test_contains_any_requires_list():
    with pytest.raises(VectorStoreFilterException, match="must be a list"):
        _contains_any("metadata.tags", "ai")


def test_contains_any_rejects_non_metadata_field():
    # Only metadata list keys are supported; anything else fails loudly (and keeps the SQL
    # a static, injection-free string).
    with pytest.raises(VectorStoreFilterException, match="only supported on metadata list fields"):
        _contains_any("content", ["ai"])


def test_convert_filters_to_query_with_contains_any():
    filters = {"field": "tags", "operator": "contains_any", "value": ["ai", "ml"]}
    where_clause, params = _convert_filters_to_query(filters)
    assert "jsonb_array_elements_text" in where_clause.as_string(None)
    assert params == ("tags", "tags", ["ai", "ml"])


@pytest.mark.parametrize(
    "operator, value, expected",
    [
        ("in", [1, 2], "(metadata->>'page')::int = ANY(%s::int[])"),
        ("in", [1.5, 2.5], "(metadata->>'page')::float = ANY(%s::float[])"),
        ("in", ["a", "b"], "(metadata->>'page')::text = ANY(%s::text[])"),
        ("not in", [1, 2], "(metadata->>'page')::int IS NULL OR (metadata->>'page')::int != ALL(%s::int[])"),
    ],
)
def test_in_operators_cast_array_to_field_type(operator, value, expected):
    query, params = _convert_filters_to_query({"field": "metadata.page", "operator": operator, "value": value})
    assert expected in repr(query)
    assert params == (value,)
