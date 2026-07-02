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
from dynamiq.nodes.extractors import Ontology
from dynamiq.nodes.llms.openai import OpenAI
from dynamiq.nodes.retrievers import GraphRetriever
from dynamiq.nodes.retrievers.graph import GraphRetrieverInputSchema
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
        {"labels": ["Organization", "Entity"], "identity_key": "id",
         "properties": {"id": ORG_ID, "name": NODE_NAME}},
        {"labels": ["System", "Entity"], "identity_key": "id",
         "properties": {"id": "sys-pub", "name": SYS_PUBLIC}},
        {"labels": ["System", "Entity"], "identity_key": "id",
         "properties": {"id": "sys-sec", "name": SYS_SECRET}},
    ]
    # Both edges start at the SAME shared org node; each carries its own per-document name snapshot + ACL.

    def _edge(dst, src_name, dst_name, principal, doc_id):
        return {
            "type": "USES", "start_label": "Organization", "end_label": "System",
            "start_identity": ORG_ID, "end_identity": dst,
            "start_identity_key": "id", "end_identity_key": "id",
            "identity_keys": ["source_doc_id"],
            "properties": {
                "src_name": src_name, "dst_name": dst_name,
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
    retriever = GraphRetriever(
        connection=graph_connection,
        llm=OpenAI(connection=OpenAIConnection(), model="gpt-4o-mini", temperature=0),
        ontology=ONTOLOGY,
        filters={"allowed_principals": {"$intersects": principals}},
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
