from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from dynamiq.storages.vector.exceptions import VectorStoreException
from dynamiq.storages.vector.weaviate.weaviate import WeaviateVectorStore


def _make_store():
    store = WeaviateVectorStore.__new__(WeaviateVectorStore)
    store.content_key = "content"
    store._collection = MagicMock()
    store.write_documents = MagicMock()
    return store


def test_replace_document_metadata_single():
    store = _make_store()
    store._collection.query.fetch_object_by_id.return_value = SimpleNamespace(
        properties={"content": "c", "old": "v"}, vector={"default": [0.1, 0.2]}
    )

    store.replace_document_metadata("d1", {"tags": ["x"]})

    doc = store.write_documents.call_args[0][0][0]
    assert doc.id == "d1"
    assert doc.content == "c"  # content preserved
    assert doc.embedding == [0.1, 0.2]  # vector preserved
    assert doc.metadata == {"tags": ["x"]}  # literal replace


def test_replace_document_metadata_multiple_ids():
    store = _make_store()
    store._collection.query.fetch_object_by_id.return_value = SimpleNamespace(
        properties={"content": "c"}, vector={"default": [0.1]}
    )

    store.replace_document_metadata(["d1", "d2"], {"category": "final"})

    docs = store.write_documents.call_args[0][0]
    assert {d.id for d in docs} == {"d1", "d2"}
    assert all(d.metadata == {"category": "final"} for d in docs)


def test_replace_document_metadata_not_found():
    store = _make_store()
    store._collection.name = "Test"
    store._collection.query.fetch_object_by_id.return_value = None
    with pytest.raises(VectorStoreException):
        store.replace_document_metadata("missing", {"tags": ["x"]})
    store.write_documents.assert_not_called()
