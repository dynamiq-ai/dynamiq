import pytest

from dynamiq.storages.vector.exceptions import VectorStoreFilterException
from dynamiq.storages.vector.pinecone.filters import _contains_any, _normalize_filters


def test_contains_any_builds_filter():
    assert _contains_any("tags", ["ai", "ml"]) == {"tags": {"$in": ["ai", "ml"]}}


def test_contains_any_requires_list():
    with pytest.raises(VectorStoreFilterException, match="must be a list"):
        _contains_any("tags", "ai")


def test_contains_any_rejects_unsupported_types():
    with pytest.raises(VectorStoreFilterException, match="Unsupported type"):
        _contains_any("tags", [{"a": 1}])


def test_normalize_filters_with_contains_any():
    filters = {"field": "metadata.tags", "operator": "contains_any", "value": ["ai", "ml"]}
    assert _normalize_filters(filters) == {"tags": {"$in": ["ai", "ml"]}}


def test_normalize_filters_contains_any_inside_logical():
    filters = {
        "operator": "AND",
        "conditions": [
            {"field": "year", "operator": ">=", "value": 2020},
            {"field": "tags", "operator": "contains_any", "value": ["ai", "ml"]},
        ],
    }
    assert _normalize_filters(filters) == {"$and": [{"year": {"$gte": 2020}}, {"tags": {"$in": ["ai", "ml"]}}]}
