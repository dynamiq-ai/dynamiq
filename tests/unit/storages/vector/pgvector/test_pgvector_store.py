import time
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock

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
