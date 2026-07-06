import pytest
from weaviate.collections.classes.filters import _FilterValue

from dynamiq.storages.vector.exceptions import VectorStoreFilterException
from dynamiq.storages.vector.weaviate.filters import _contains_all, _contains_any, convert_filters


def test_contains_any_builds_filter():
    result = _contains_any("tags", ["ai", "ml"])
    assert isinstance(result, _FilterValue)
    assert result.value == ["ai", "ml"]
    assert result.operator.value == "ContainsAny"


def test_contains_all_builds_filter():
    result = _contains_all("tags", ["ai", "ml"])
    assert isinstance(result, _FilterValue)
    assert result.value == ["ai", "ml"]
    assert result.operator.value == "ContainsAll"


@pytest.mark.parametrize("builder", [_contains_any, _contains_all])
def test_contains_operators_require_list(builder):
    with pytest.raises(VectorStoreFilterException, match="must be a list"):
        builder("tags", "ai")


@pytest.mark.parametrize("operator", ["contains_any", "contains_all"])
def test_convert_filters_with_contains_operators(operator):
    filters = {"field": "metadata.tags", "operator": operator, "value": ["ai", "ml"]}
    assert convert_filters(filters) is not None


def test_convert_filters_contains_any_inside_logical():
    filters = {
        "operator": "AND",
        "conditions": [
            {"field": "year", "operator": ">=", "value": 2020},
            {"field": "tags", "operator": "contains_all", "value": ["ai", "ml"]},
        ],
    }
    assert convert_filters(filters) is not None


@pytest.mark.parametrize("operator", ["contains_any", "contains_all"])
def test_not_wrapping_contains_raises_clear_error(operator):
    # `contains_*` have no invertible counterpart in Weaviate; a NOT around them must
    # raise a clear VectorStoreFilterException, not a KeyError.
    filters = {
        "operator": "NOT",
        "conditions": [
            {"field": "tags", "operator": operator, "value": ["ai", "ml"]},
        ],
    }
    with pytest.raises(VectorStoreFilterException, match="cannot be used inside a 'NOT' filter"):
        convert_filters(filters)
