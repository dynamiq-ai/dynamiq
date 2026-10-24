"""ChromaVectorStore against chromadb 1.x behavior, with the client mocked."""

from unittest.mock import MagicMock

import numpy as np
import pytest
from chromadb.errors import NotFoundError

from dynamiq.storages.vector.chroma.chroma import ChromaVectorStore


@pytest.fixture
def store_and_collection():
    client = MagicMock()
    collection = client.get_collection.return_value
    collection.query.return_value = {
        "ids": [[]],
        "documents": [[]],
        "metadatas": [[]],
        "embeddings": [[]],
        "distances": [[]],
    }
    return ChromaVectorStore(client=client, index_name="docs"), collection


def test_search_with_empty_filters_sends_no_filter(store_and_collection):
    store, collection = store_and_collection

    store.search_embeddings(query_embeddings=[[0.1, 0.2]], top_k=1, filters={})

    assert collection.query.call_args.kwargs["where"] is None
    assert collection.query.call_args.kwargs["where_document"] is None


def test_content_filter_sends_no_metadata_filter(store_and_collection):
    store, collection = store_and_collection

    store.delete_documents_by_filters({"operator": "==", "field": "content", "value": "tea"})

    assert collection.delete.call_args.kwargs == {"ids": None, "where": None, "where_document": {"$contains": "tea"}}


def test_create_if_not_exist_creates_a_missing_collection():
    client = MagicMock()
    client.get_collection.side_effect = NotFoundError("Collection [docs] does not exist")

    ChromaVectorStore(client=client, index_name="docs", create_if_not_exist=True)

    client.create_collection.assert_called_once_with(name="docs")


def test_documents_without_metadata_and_numpy_embeddings_are_read(store_and_collection):
    store, collection = store_and_collection
    collection.query.return_value = {
        "ids": [["a"]],
        "documents": [["tea"]],
        "metadatas": [[None]],
        "embeddings": [np.array([[0.1, 0.2]])],
        "distances": [[0.5]],
    }
    collection.get.return_value = {
        "ids": ["a"],
        "documents": ["tea"],
        "metadatas": [None],
        "embeddings": np.array([[0.1, 0.2]]),
    }

    [[found]] = store.search_embeddings(query_embeddings=[[0.1, 0.2]], top_k=1)
    [fetched] = store.get_documents_by_id(["a"], include_embeddings=True)

    assert found.metadata == fetched.metadata == {}
    assert fetched.embedding == [0.1, 0.2]
