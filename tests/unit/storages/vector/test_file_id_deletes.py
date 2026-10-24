"""Deleting by file id matches the file id in each document's metadata, exactly."""

from unittest.mock import MagicMock

import pytest

from dynamiq.storages.vector import ElasticsearchVectorStore, OpenSearchVectorStore

STORES = [
    pytest.param(ElasticsearchVectorStore, id="elasticsearch"),
    pytest.param(OpenSearchVectorStore, id="opensearch"),
]


def deleted_query(call):
    return call.kwargs["query"] if "query" in call.kwargs else call.kwargs["body"]["query"]


@pytest.mark.parametrize("store_cls", STORES)
def test_delete_documents_by_file_ids_matches_the_metadata_file_id_in_batches(store_cls):
    client = MagicMock()
    store = store_cls(client=client, index_name="docs")

    store.delete_documents_by_file_ids(["a-1", "b-2", "c-3"], batch_size=2)

    assert [deleted_query(call) for call in client.delete_by_query.call_args_list] == [
        {"bool": {"must": [{"terms": {"metadata.file_id.keyword": ["a-1", "b-2"]}}]}},
        {"bool": {"must": [{"terms": {"metadata.file_id.keyword": ["c-3"]}}]}},
    ]


@pytest.mark.parametrize("store_cls", STORES)
def test_delete_documents_by_file_id_matches_the_metadata_file_id(store_cls):
    client = MagicMock()
    store = store_cls(client=client, index_name="docs")

    store.delete_documents_by_file_id("a-1")

    assert deleted_query(client.delete_by_query.call_args) == {
        "bool": {"must": [{"terms": {"metadata.file_id.keyword": ["a-1"]}}]}
    }
