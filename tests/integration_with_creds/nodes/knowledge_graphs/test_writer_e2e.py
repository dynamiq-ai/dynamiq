"""E2E for the WRITE stage: what ``KnowledgeGraphWriter`` persists into a real Neo4j.

This file asserts on the GRAPH STATE (Cypher counts / index metadata), not on retrieval — that is the line
that separates it from ``test_retrieval_e2e.py`` and ``test_acl_e2e.py``. It covers:

  * entity RESOLUTION — the same entity named across documents resolves to ONE node (dedup by name), while
    each document's assertion stays its own edge (per-document provenance);
  * DELETION — ``delete_documents`` removes exactly one document's edges and sweeps entities left orphaned;
  * EMBEDDING WRITE — with an ``entity_embedder`` the writer stores an embedding on each entity and creates
    the Neo4j vector index (the substrate the retriever's semantic seeding relies on).

Requires ``OPENAI_API_KEY`` and ``NEO4J_URI`` /``NEO4J_USERNAME`` /``NEO4J_PASSWORD``; skipped otherwise. In
CI a local ``neo4j:5-community`` container supplies Neo4j (docker-compose ``neo4j`` service) — no Aura creds.
Fixtures wipe the graph before writing and are ordered so each section's wipe follows the previous section's
assertions; run serially against a dedicated test instance.
"""

import os

import pytest

from dynamiq.connections import Neo4j as Neo4jConnection
from dynamiq.connections import OpenAI as OpenAIConnection
from dynamiq.nodes.embedders import OpenAIDocumentEmbedder
from dynamiq.nodes.knowledge_graphs import KnowledgeGraphEntityExtractor, KnowledgeGraphWriter, Ontology
from dynamiq.nodes.knowledge_graphs.entity_extractor import ENTITY_EMBEDDING_VECTOR_INDEX
from dynamiq.nodes.llms.openai import OpenAI
from dynamiq.storages.graph.neo4j import Neo4jGraphStore
from dynamiq.types import Document

ONTOLOGY = Ontology(entity_types=["Organization", "System"], relationship_types=["USES"])


def _llm():
    return OpenAI(connection=OpenAIConnection(), model="gpt-4o-mini", temperature=0)


@pytest.fixture(scope="module")
def graph_connection():
    if not (os.getenv("OPENAI_API_KEY") and os.getenv("NEO4J_URI")):
        pytest.skip("OPENAI_API_KEY and NEO4J_URI required; skipping credentials-required test.")
    return Neo4jConnection()


def _counter(store):
    def _count(cypher, **params):
        rows, _, _ = store.run_cypher(cypher, parameters=params)
        return store.format_records(rows)[0]["c"]

    return _count


# --- 1. Entity resolution / dedup --------------------------------------------------------------------
# Two documents both mention "Acme"; the writer must resolve them to ONE organization node (identity is
# uuid5(label:normalized_name), so a second document about the same entity adopts the same node) while each
# document's "Acme uses <system>" claim stays its own edge.

ORG = "Acme"


@pytest.fixture(scope="module")
def resolved_graph(graph_connection):
    driver = graph_connection.connect()
    with driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n").consume()  # clean slate
    driver.close()

    extractor = KnowledgeGraphEntityExtractor(llm=_llm(), ontology=ONTOLOGY)
    extractor.init_components()
    writer = KnowledgeGraphWriter(connection=graph_connection)
    writer.init_components()
    # Two SEPARATE ingestion events about the same org -> exercises cross-write resolution, not just
    # in-one-call dedup.
    for text, principal in (
        (f"{ORG} uses a system called Helios.", "group:a"),
        (f"{ORG} uses a system called Borealis.", "group:b"),
    ):
        extraction = extractor.execute(
            KnowledgeGraphEntityExtractor.input_schema(
                documents=[Document(content=text, metadata={"allowed_principals": [principal]})]
            )
        )
        writer.execute(
            KnowledgeGraphWriter.input_schema(nodes=extraction["nodes"], relationships=extraction["relationships"])
        )
    yield
    writer.graph_store.close()


def test_same_entity_across_documents_resolves_to_one_node(resolved_graph, graph_connection):
    store = Neo4jGraphStore(connection=graph_connection, client=graph_connection.connect())
    try:
        count = _counter(store)
        orgs = count(
            "MATCH (n) WHERE toLower(n.name) CONTAINS toLower($x) RETURN count(n) AS c",
            x=ORG,
        )
        assert orgs == 1, f"'{ORG}' from two documents should resolve to ONE node, found {orgs}"
        # the two distinct systems are two nodes, and each document's claim is its own edge
        assert count("MATCH ()-[r:USES]->() RETURN count(r) AS c") == 2, "each document's fact should be its own edge"
    finally:
        store.close()


# --- 2. Deletion (store-level write -> deterministic) ------------------------------------------------

FILTER_ORG = "org-w"


def _org_uses(store, edges):
    """Write one Acme org + a System per edge spec (system_name, extra_props, doc_id) with USES edges."""
    nodes = [
        {"labels": ["Organization", "Entity"], "id": FILTER_ORG, "name": "Acme"}
    ]
    relationships = []
    for i, (sys_name, extra, doc_id) in enumerate(edges):
        sid = f"sys-{i}"
        nodes.append(
            {"labels": ["System", "Entity"], "id": sid, "name": sys_name}
        )
        relationships.append(
            {
                "type": "USES",
                "start_node": {"label": "Organization", "id": FILTER_ORG, "name": "Acme"},
                "end_node": {"label": "System", "id": sid, "name": sys_name},
                "identity_keys": ["source_doc_id"],
                "properties": {
                    "source_doc_id": doc_id,
                    **extra,
                },
            }
        )
    store.write_graph(nodes=nodes, relationships=relationships)


def test_delete_documents_removes_only_that_documents_edges(graph_connection):
    """``delete_documents`` removes every edge a document asserted (matched by ``source_doc_id``) and sweeps
    any entity left with no remaining edge — without touching another document's edges."""
    store = Neo4jGraphStore(connection=graph_connection, client=graph_connection.connect())
    store.run_cypher("MATCH (n) DETACH DELETE n")  # clean slate
    try:
        _org_uses(store, [("Helios", {}, "docA"), ("Borealis", {}, "docB")])

        writer = KnowledgeGraphWriter(connection=graph_connection)
        writer.init_components()
        try:
            deleted = writer.delete_documents(["docA"])
        finally:
            writer.graph_store.close()
        assert deleted["relationships_deleted"] == 1, deleted

        count = _counter(store)
        assert count("MATCH ()-[r:USES]->() RETURN count(r) AS c") == 1, "only docB's edge should remain"
        assert count("MATCH (n) WHERE n.name = 'Borealis' RETURN count(n) AS c") == 1, "docB's system must remain"
        assert count("MATCH (n) WHERE n.name = 'Helios' RETURN count(n) AS c") == 0, "orphaned Helios must be swept"
    finally:
        store.close()


# --- 3. Embedding write (entity_embedder) ------------------------------------------------------------
# With an entity_embedder the writer embeds each entity name and creates the Neo4j vector index — the
# substrate the retriever's semantic seeding relies on (asserted end-to-end in test_retrieval_e2e.py).

VEC_ORG_NAME = "Car Manufacturer"
VEC_SYS_NAME = "Assembly Robotics"


@pytest.fixture(scope="module")
def embedded_graph(graph_connection):
    store = Neo4jGraphStore(connection=graph_connection, client=graph_connection.connect())
    store.run_cypher("MATCH (n) DETACH DELETE n")  # clean slate

    nodes = [
        {"labels": ["Organization", "Entity"], "id": "vec-org", "name": VEC_ORG_NAME},
        {"labels": ["System", "Entity"], "id": "vec-sys", "name": VEC_SYS_NAME},
    ]
    relationships = [
        {
            "type": "USES",
            "start_node": {"label": "Organization", "id": "vec-org", "name": VEC_ORG_NAME},
            "end_node": {"label": "System", "id": "vec-sys", "name": VEC_SYS_NAME},
            "identity_keys": ["source_doc_id"],
            "properties": {
                "allowed_principals": ["group:public"],
                "source_doc_id": "docV",
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
        writer.graph_store.close()
    yield
    store.close()


def test_vector_index_created_and_entities_embedded(embedded_graph, graph_connection):
    store = Neo4jGraphStore(connection=graph_connection, client=graph_connection.connect())
    try:
        count = _counter(store)
        indexes = count(
            "SHOW INDEXES YIELD name, type WHERE name = $n AND type = 'VECTOR' RETURN count(*) AS c",
            n=ENTITY_EMBEDDING_VECTOR_INDEX,
        )
        assert indexes == 1, "entity vector index should have been created"
        embedded = count("MATCH (n:Entity) WHERE n.embedding IS NOT NULL RETURN count(n) AS c")
        assert embedded >= 2, "entity nodes should carry an embedding"
    finally:
        store.close()
