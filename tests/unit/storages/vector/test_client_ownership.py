"""A vector store closes only the client it created: a client passed in can be shared through ConnectionManager."""

import gc
from unittest.mock import MagicMock

import pytest

from dynamiq.connections import AWSOpenSearch, Elasticsearch, PostgreSQL
from dynamiq.storages.vector import ElasticsearchVectorStore, OpenSearchVectorStore, PGVectorStore

STORES = [
    pytest.param(ElasticsearchVectorStore, Elasticsearch, {"index_name": "docs"}, id="elasticsearch"),
    pytest.param(OpenSearchVectorStore, AWSOpenSearch, {"index_name": "docs"}, id="opensearch"),
    pytest.param(PGVectorStore, PostgreSQL, {"table_name": "docs"}, id="pgvector"),
]


@pytest.fixture(autouse=True)
def skip_pgvector_type_registration(monkeypatch):
    monkeypatch.setattr("dynamiq.storages.vector.pgvector.pgvector.register_vector", lambda conn: None)


@pytest.mark.parametrize("store_cls, connection_cls, params", STORES)
def test_deleting_a_store_leaves_a_client_passed_in_open(store_cls, connection_cls, params):
    client = MagicMock(closed=False)

    store_cls(client=client, **params)
    gc.collect()

    client.close.assert_not_called()


@pytest.mark.parametrize("store_cls, connection_cls, params", STORES)
def test_deleting_a_store_closes_the_client_it_created(store_cls, connection_cls, params):
    connection = MagicMock(spec=connection_cls)
    connection.connect.return_value.closed = False

    store_cls(connection=connection, **params)
    gc.collect()

    connection.connect.return_value.close.assert_called_once()
