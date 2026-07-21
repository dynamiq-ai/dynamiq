"""Unit tests for KnowledgeGraphChunkWriter: chunk→entity-tag recovery + chunk write (no DB)."""

from dynamiq.connections import PostgreSQL
from dynamiq.nodes.extractors import KnowledgeGraphChunkWriter, KnowledgeGraphWriter
from dynamiq.storages.graph.postgres import PostgresGraphStore
from dynamiq.types import Document

from .test_entity_extractor import StubLLM


class _FakeCursor:
    def __init__(self, store):
        self.store = store

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def execute(self, sql, params=None):
        pass

    def executemany(self, sql, rows):
        self.store.many.append((" ".join(sql.split()), list(rows)))


class _FakeClient:
    def __init__(self):
        self.many = []

    def cursor(self):
        return _FakeCursor(self)


def test_chunk_writer_tags_chunks_with_resolved_entities(monkeypatch):
    # super().execute() (the graph write) returns RESOLVED relationships carrying source_doc_id +
    # resolved endpoints — the chunk writer recovers each chunk's entity ids from those.
    resolved_rels = [
        {"type": "WORKS_AT", "start_identity": "uuid-jane", "end_identity": "uuid-acme",
         "properties": {"source_doc_id": "c1"}},
        {"type": "USES", "start_identity": "uuid-acme", "end_identity": "uuid-helios",
         "properties": {"source_doc_id": "c2"}},
    ]
    monkeypatch.setattr(KnowledgeGraphWriter, "execute", lambda self, i, config=None, **k: {"relationships": resolved_rels})

    writer = KnowledgeGraphChunkWriter(
        llm=StubLLM(), connection=PostgreSQL(host="x"), is_postponed_component_init=True
    )
    fake = _FakeClient()
    writer._graph_store = PostgresGraphStore.__new__(PostgresGraphStore)  # bypass __init__/DB
    writer._graph_store.client = fake
    writer._graph_store.embedding_dimension = 1536

    docs = [
        Document(id="c1", content="Jane works at Acme.", embedding=[0.1, 0.2],
                 metadata={"allowed_principals": ["group:a"]}),
        Document(id="c2", content="Acme uses Helios.", embedding=[0.3, 0.4],
                 metadata={"allowed_principals": ["group:b"]}),
    ]
    out = writer.execute(KnowledgeGraphChunkWriter.input_schema(documents=docs))

    assert out["chunks_written"] == 2
    _, rows = next((s, r) for s, r in fake.many if "kg_chunk" in s)
    by_id = {r[0]: r for r in rows}
    # c1's chunk is tagged with the entities of its facts (jane, acme); ACL carried from metadata.
    assert by_id["c1"][3] == ["uuid-acme", "uuid-jane"] and by_id["c1"][4] == ["group:a"]
    assert by_id["c2"][3] == ["uuid-acme", "uuid-helios"] and by_id["c2"][4] == ["group:b"]


def test_chunk_writer_noop_on_non_postgres(monkeypatch):
    monkeypatch.setattr(KnowledgeGraphWriter, "execute", lambda self, i, config=None, **k: {"relationships": []})
    from dynamiq.connections import Neo4j

    writer = KnowledgeGraphChunkWriter(llm=StubLLM(), connection=Neo4j(uri="bolt://x", username="u", password="p"),
                                       is_postponed_component_init=True)

    class _NotPostgres:
        pass

    writer._graph_store = _NotPostgres()
    out = writer.execute(KnowledgeGraphChunkWriter.input_schema(documents=[Document(content="x")]))
    assert "chunks_written" not in out  # behaves like the parent on non-Postgres backends
