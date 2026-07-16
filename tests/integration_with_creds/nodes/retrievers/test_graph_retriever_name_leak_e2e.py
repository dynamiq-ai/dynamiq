"""E2E regression for the shared-node NAME leak (case B2) against a real Neo4j.

Entity nodes are shared/merged and carry a single, first-writer-wins ``name``; all ACL lives on edges.
Before the fix, the retriever rendered endpoint names by dereferencing that shared node, so a caller
entitled only to a *public* edge could see a name written by a *differently-scoped* document (e.g. a
secret-encoded variant). The fix snapshots each edge's endpoint names onto the edge itself
(``src_name``/``dst_name``), which inherit the edge's ACL, and the retriever renders from those.

This test builds the situation deterministically at the store level: ONE shared organisation node whose
stored name is the SECRET variant (first/only writer), referenced by two USES edges with DIFFERENT
per-document name snapshots and DIFFERENT ``allowed_principals``. It then asserts a ``group:public``
caller renders the edge's own ``Acme`` snapshot and NEVER the shared node's ``Acme Secret Division`` name
nor the secret system. Seeding is by ``entity_ids`` (anchored) so the assertion is LLM-free and stable.

Requires NEO4J_URI / NEO4J_USERNAME / NEO4J_PASSWORD and OPENAI_API_KEY. Skipped otherwise. Wipes the
graph first — run serially against a dedicated test instance, not a shared one.
"""

import os

import pytest

from dynamiq.connections import Neo4j as Neo4jConnection
from dynamiq.connections import OpenAI as OpenAIConnection
from dynamiq.nodes.embedders import OpenAIDocumentEmbedder, OpenAITextEmbedder
from dynamiq.nodes.knowledge_graphs import KnowledgeGraphRetriever, KnowledgeGraphWriter, Ontology
from dynamiq.nodes.knowledge_graphs.entity_extractor import ENTITY_EMBEDDING_VECTOR_INDEX
from dynamiq.nodes.knowledge_graphs.retriever import GraphRetrieverInputSchema
from dynamiq.nodes.llms.openai import OpenAI
from dynamiq.storages.graph.neo4j import Neo4jGraphStore

ORG_ID = "org-1"
NODE_NAME = "Acme Secret Division"  # the shared node's stored (secret-encoded) name — MUST NOT leak
PUBLIC_NAME = "Acme"  # the public edge's own name snapshot — what a public caller should see
SYS_PUBLIC = "Helios"  # group:public may see this
SYS_SECRET = "Borealis"  # group:secret only
GROUP_PUBLIC = "group:public"
GROUP_SECRET = "group:secret"

ONTOLOGY = Ontology(entity_types=["Organization", "System"], relationship_types=["USES"])


@pytest.fixture(scope="module")
def graph_connection():
    if not (os.getenv("OPENAI_API_KEY") and os.getenv("NEO4J_URI")):
        pytest.skip("OPENAI_API_KEY and NEO4J_URI required; skipping credentials-required test.")
    return Neo4jConnection()


@pytest.fixture(scope="module")
def shared_node_graph(graph_connection):
    """Wipe, then write ONE shared org node + two USES edges with distinct name snapshots and ACLs."""
    store = Neo4jGraphStore(connection=graph_connection, client=graph_connection.connect())
    store.run_cypher("MATCH (n) DETACH DELETE n")  # clean slate

    nodes = [
        {"labels": ["Organization", "Entity"], "id": ORG_ID, "name": NODE_NAME, "properties": {}},
        {"labels": ["System", "Entity"], "id": "sys-pub", "name": SYS_PUBLIC, "properties": {}},
        {"labels": ["System", "Entity"], "id": "sys-sec", "name": SYS_SECRET, "properties": {}},
    ]
    # Both edges start at the SAME shared org node; each carries its own per-document name snapshot + ACL.

    def _edge(dst, src_name, dst_name, principal, doc_id):
        return {
            "type": "USES",
            "start_label": "Organization",
            "end_label": "System",
            "start_identity": ORG_ID,
            "end_identity": dst,
            "src_name": src_name,
            "dst_name": dst_name,
            "identity_keys": ["source_doc_id"],
            "properties": {
                "allowed_principals": [principal], "source_doc_id": doc_id,
            },
        }

    relationships = [
        _edge("sys-pub", PUBLIC_NAME, SYS_PUBLIC, GROUP_PUBLIC, "docP"),
        _edge("sys-sec", NODE_NAME, SYS_SECRET, GROUP_SECRET, "docS"),
    ]
    store.write_graph(nodes=nodes, relationships=relationships)
    yield
    store.close()


def _facts_for(graph_connection, principals):
    retriever = KnowledgeGraphRetriever(
        connection=graph_connection,
        llm=OpenAI(connection=OpenAIConnection(), model="gpt-4o-mini", temperature=0),
        ontology=ONTOLOGY,
        filters={"field": "allowed_principals", "operator": "contains_any", "value": principals},
    )
    retriever.init_components()
    try:
        # Anchor by id (skips LLM entity extraction) so the assertion is deterministic.
        out = retriever.execute(GraphRetrieverInputSchema(query="What does the org use?", entity_ids=[ORG_ID]))
        return out["content"]
    finally:
        retriever._graph_store.close()


def test_public_caller_sees_edge_snapshot_not_shared_node_name(shared_node_graph, graph_connection):
    content = _facts_for(graph_connection, [GROUP_PUBLIC])
    assert SYS_PUBLIC in content, f"public caller should see its own system {SYS_PUBLIC}: {content}"
    assert PUBLIC_NAME in content, f"public caller should render its edge's own name snapshot: {content}"
    # The shared node's stored (secret-encoded) name must never reach a public-only caller.
    assert NODE_NAME not in content, f"LEAK: shared-node name {NODE_NAME!r} surfaced to public caller: {content}"
    assert SYS_SECRET not in content, f"LEAK: public caller must not see secret system {SYS_SECRET}: {content}"


def test_secret_caller_sees_its_own_edge(shared_node_graph, graph_connection):
    content = _facts_for(graph_connection, [GROUP_SECRET])
    assert SYS_SECRET in content
    assert SYS_PUBLIC not in content, f"LEAK: secret caller must not see public system {SYS_PUBLIC}: {content}"


def test_unknown_principal_sees_nothing(shared_node_graph, graph_connection):
    content = _facts_for(graph_connection, ["group:nobody"])
    assert SYS_PUBLIC not in content and SYS_SECRET not in content, f"unknown principal must see nothing: {content}"


# --- Semantic (embedding) seeding -------------------------------------------------------------------
# When the writer is given an `entity_embedder`, each entity's NAME is embedded and a Neo4j vector index
# is created; a retriever with a matching `text_embedder` then seeds entry entities by embedding
# similarity. This proves a PARAPHRASED seed ("automaker") reaches an entity named "Car Manufacturer" —
# a match the full-text/CONTAINS name path cannot make.

VEC_ORG_NAME = "Car Manufacturer"
VEC_SYS_NAME = "Assembly Robotics"
VEC_DOC = "docV"


@pytest.fixture(scope="module")
def embedded_graph(graph_connection):
    """Wipe, then ingest ONE org->system fact through the writer WITH an entity embedder (real OpenAI).

    Exercises the real write path: ``_embed_nodes`` (name embedding) + ``_ensure_entity_vector_index``.
    """
    store = Neo4jGraphStore(connection=graph_connection, client=graph_connection.connect())
    store.run_cypher("MATCH (n) DETACH DELETE n")  # clean slate

    nodes = [
        {
            "labels": ["Organization", "Entity"],
            "id": "vec-org",
            "name": VEC_ORG_NAME,
            "properties": {},
        },
        {"labels": ["System", "Entity"], "id": "vec-sys", "name": VEC_SYS_NAME, "properties": {}},
    ]
    relationships = [
        {
            "type": "USES",
            "start_label": "Organization",
            "end_label": "System",
            "start_identity": "vec-org",
            "end_identity": "vec-sys",
            "src_name": VEC_ORG_NAME,
            "dst_name": VEC_SYS_NAME,
            "identity_keys": ["source_doc_id"],
            "properties": {
                "allowed_principals": [GROUP_PUBLIC],
                "source_doc_id": VEC_DOC,
            },
        }
    ]

    writer = KnowledgeGraphWriter(
        connection=graph_connection,
        entity_embedder=OpenAIDocumentEmbedder(connection=OpenAIConnection()),
    )
    writer.init_components()
    try:
        writer.execute(KnowledgeGraphWriter.input_schema(nodes=nodes, relationships=relationships))
    finally:
        writer._graph_store.close()
    yield
    store.close()


def test_vector_index_created_and_entities_embedded(embedded_graph, graph_connection):
    store = Neo4jGraphStore(connection=graph_connection, client=graph_connection.connect())
    try:
        rows, _, _ = store.run_cypher(
            "SHOW INDEXES YIELD name, type WHERE name = $n AND type = 'VECTOR' RETURN count(*) AS c",
            parameters={"n": ENTITY_EMBEDDING_VECTOR_INDEX},
        )
        assert store.format_records(rows)[0]["c"] == 1, "entity vector index should have been created"
        embedded, _, _ = store.run_cypher("MATCH (n:Entity) WHERE n.embedding IS NOT NULL RETURN count(n) AS c")
        assert store.format_records(embedded)[0]["c"] >= 2, "entity nodes should carry an embedding"
    finally:
        store.close()


def test_paraphrased_query_seeds_via_vector_similarity(embedded_graph, graph_connection):
    # "automaker" never appears as a stored name; only a semantic (vector) seed can reach "Car Manufacturer".
    retriever = KnowledgeGraphRetriever(
        connection=graph_connection,
        llm=OpenAI(connection=OpenAIConnection(), model="gpt-4o-mini", temperature=0),
        text_embedder=OpenAITextEmbedder(connection=OpenAIConnection()),
        ontology=ONTOLOGY,
        filters={"field": "allowed_principals", "operator": "contains_any", "value": [GROUP_PUBLIC]},
    )
    retriever.init_components()
    assert retriever._use_vector, "vector seeding should be active (embedder set + vector index present)"
    try:
        # `entities` skips LLM extraction so the seed term is fixed; it is embedded and vector-matched.
        out = retriever.execute(GraphRetrieverInputSchema(query="What does the automaker use?", entities=["automaker"]))
        assert VEC_SYS_NAME in out["content"], f"paraphrased vector seed should reach the fact: {out['content']}"
    finally:
        retriever._graph_store.close()
