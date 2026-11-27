"""Pinecone and Weaviate search filters must treat list values as membership.

InMemory, PostgreSQL, DynamoDB and Qdrant read a list filter value as "the stored value is one of these".
Pinecone passed the raw dict to the vector store, which rejects it, and Weaviate compared with "==".
"""

from types import SimpleNamespace

import pytest

from dynamiq.memory.backends.pinecone import Pinecone
from dynamiq.memory.backends.weaviate import Weaviate
from dynamiq.storages.vector.pinecone.filters import _normalize_filters


def _pinecone(filters):
    # The method reads no instance state, so no Pinecone client is needed.
    return Pinecone._prepare_filters(SimpleNamespace(), filters)


def _weaviate(filters):
    backend = SimpleNamespace(_vector_store=SimpleNamespace(is_valid_property_name=lambda name: True))
    return Weaviate._prepare_filters(backend, filters)


def test_pinecone_list_filter_uses_membership():
    prepared = _pinecone({"tags": ["a", "b"], "user_id": "u1"})

    assert _normalize_filters(prepared) == {"$and": [{"tags": {"$in": ["a", "b"]}}, {"user_id": {"$eq": "u1"}}]}


def test_weaviate_list_filter_uses_membership():
    assert _weaviate({"tags": ["a", "b"], "user_id": "u1"}) == {
        "operator": "AND",
        "conditions": [
            {"field": "tags", "operator": "in", "value": ["a", "b"]},
            {"field": "user_id", "operator": "==", "value": "u1"},
        ],
    }


@pytest.mark.parametrize("prepare", [_pinecone, _weaviate])
@pytest.mark.parametrize(
    "filters",
    [
        {"operator": "AND", "conditions": [{"field": "user_id", "operator": "==", "value": "u1"}]},
        {"field": "tags", "operator": "in", "value": ["a", "b"]},
    ],
)
def test_structured_filters_pass_through(prepare, filters):
    assert prepare(filters) == filters
