import time
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock

import numpy as np
import pytest

from dynamiq.storages.vector import PGVectorStore
from dynamiq.storages.vector.pgvector.pgvector import PGVectorIndexMethod, PGVectorVectorFunction


def test_stores_sharing_a_connection_register_the_vector_types_once(monkeypatch):
    connection = MagicMock(closed=False)
    connection.adapters.types.get.return_value = None
    registrations = []

    def register_vector(conn):
        registrations.append(conn)
        time.sleep(0.05)
        conn.adapters.types.get.return_value = object()

    monkeypatch.setattr("dynamiq.storages.vector.pgvector.pgvector.register_vector", register_vector)

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda _: PGVectorStore(client=connection, table_name="docs"), range(8)))

    assert registrations == [connection]


def test_searches_sharing_a_connection_each_query_with_their_own_index_setting(monkeypatch):
    monkeypatch.setattr("dynamiq.storages.vector.pgvector.pgvector.register_vector", lambda conn: None)
    connection = MagicMock(closed=False)
    store = PGVectorStore(
        client=connection,
        table_name="docs",
        dimension=2,
        index_method=PGVectorIndexMethod.HNSW,
        vector_function=PGVectorVectorFunction.L2_DISTANCE,
    )
    current, queried = {}, []

    def execute(sql, *args, **kwargs):
        current["ef_search"] = int(sql.rsplit(" ", 1)[1])
        time.sleep(0.1)  # room for another search to set its own value

    connection.execute.side_effect = execute
    cursor = connection.cursor.return_value.__enter__.return_value
    cursor.execute.side_effect = lambda *args, **kwargs: queried.append(current["ef_search"]) or cursor
    cursor.fetchall.return_value = []

    with ThreadPoolExecutor(max_workers=2) as pool:
        large = pool.submit(store._embedding_retrieval, [0.1, 0.2], top_k=100)
        time.sleep(0.05)
        small = pool.submit(store._embedding_retrieval, [0.1, 0.2], top_k=5)
        large.result(), small.result()

    assert queried == [200, 40]  # HNSW returns at most ef_search rows, so 40 would cut the first search short


def _store_capturing_sql(monkeypatch):
    monkeypatch.setattr("dynamiq.storages.vector.pgvector.pgvector.register_vector", lambda conn: None)
    connection = MagicMock(closed=False)
    store = PGVectorStore(client=connection, table_name="docs", dimension=2)
    cursor = connection.cursor.return_value.__enter__.return_value
    cursor.reset_mock()  # drop the table check the constructor ran
    cursor.fetchall.return_value = []
    return store, cursor


@pytest.mark.parametrize(
    "retrieve",
    [
        lambda s, ck, ek: s._embedding_retrieval([0.1, 0.2], content_key=ck, embedding_key=ek),
        lambda s, ck, ek: s._keyword_retrieval("q", content_key=ck, embedding_key=ek),
        lambda s, ck, ek: s._hybrid_retrieval("q", [0.1, 0.2], alpha=0.5, content_key=ck, embedding_key=ek),
    ],
)
def test_retrieval_quotes_content_and_embedding_keys(monkeypatch, retrieve):
    store, cursor = _store_capturing_sql(monkeypatch)
    content_key = 'content", (SELECT 1) AS "x'
    embedding_key = 'embedding" <-> NULL) --'

    retrieve(store, content_key, embedding_key)

    sql = cursor.execute.call_args.args[0].as_string(None)
    assert '"content"", (SELECT 1) AS ""x"' in sql
    assert content_key not in sql
    assert embedding_key not in sql.replace('""', "")


def test_embedding_retrieval_renders_query_embedding_as_literal(monkeypatch):
    store, cursor = _store_capturing_sql(monkeypatch)

    store._embedding_retrieval([0.1, np.float32(0.5)])

    sql = cursor.execute.call_args.args[0].as_string(None)
    assert "1 - (\"embedding\" <=> '[0.1,0.5]') AS score" in sql


@pytest.mark.parametrize("element", ["0.2]'::vector) --", True, None])
def test_embedding_retrieval_rejects_non_numeric_embedding(monkeypatch, element):
    store, cursor = _store_capturing_sql(monkeypatch)

    with pytest.raises(ValueError, match="only numbers"):
        store._embedding_retrieval([0.1, element])
    cursor.execute.assert_not_called()
