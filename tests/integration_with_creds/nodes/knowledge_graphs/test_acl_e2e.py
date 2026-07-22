"""E2E for the SECURITY guarantee: edge-level ACL enforcement on ``KnowledgeGraphRetriever`` against Neo4j.

ACL lives on edges (``allowed_principals``); the retriever compiles a locked ``contains_any`` filter into the
traversal server-side (not LLM-written Cypher). This file asserts the promise that retrieval NEVER leaks
across principals. It covers:

  * ISOLATION — a caller sees its own facts and never another principal's; unknown principals see nothing
    (default-deny); the SAME fact asserted by two differently-scoped documents keeps BOTH ACLs;
  * NAME LEAK — endpoint names are rendered from each edge's own snapshot (``src_name``/``dst_name``), so a
    public-only caller never sees a name written onto the shared entity node by a differently-scoped doc;
  * NO WIDENING — the node's locked ``filters`` always apply; caller-supplied input ``filters`` are AND-ed on
    top and can only further NARROW, never widen past the ACL.

Requires ``OPENAI_API_KEY`` and ``NEO4J_URI`` /``NEO4J_USERNAME`` /``NEO4J_PASSWORD``; skipped otherwise. In
CI a local ``neo4j:5-community`` container supplies Neo4j (docker-compose ``neo4j`` service) — no Aura creds.
Fixtures wipe the graph and are ordered so each section's wipe follows the previous section's assertions
(entity names are NOT randomly suffixed — the extraction LLM normalizes suffixes off, making them useless
for isolation; a per-run wipe is the isolation mechanism). Run serially against a dedicated test instance.
"""

import os

import pytest

from dynamiq.connections import Neo4j as Neo4jConnection
from dynamiq.connections import OpenAI as OpenAIConnection
from dynamiq.nodes.embedders import OpenAIDocumentEmbedder, OpenAITextEmbedder
from dynamiq.nodes.knowledge_graphs import (
    KnowledgeGraphEntityExtractor,
    KnowledgeGraphRetriever,
    KnowledgeGraphWriter,
    Ontology,
)
from dynamiq.nodes.knowledge_graphs.retriever import GraphRetrieverInputSchema
from dynamiq.nodes.llms.openai import OpenAI
from dynamiq.storages.graph.neo4j import Neo4jGraphStore
from dynamiq.types import Document

# The extractor still needs the ontology to constrain extraction; the retriever no longer does — it seeds
# on explicit entities/entity_ids + a text_embedder, so it needs neither an llm nor an ontology.
ONTOLOGY = Ontology(entity_types=["Organization", "System"], relationship_types=["USES"])


def _llm():
    return OpenAI(connection=OpenAIConnection(), model="gpt-4o-mini", temperature=0)


def _doc_embedder():
    return OpenAIDocumentEmbedder(connection=OpenAIConnection())


def _text_embedder():
    return OpenAITextEmbedder(connection=OpenAIConnection())


@pytest.fixture(scope="module")
def graph_connection():
    if not (os.getenv("OPENAI_API_KEY") and os.getenv("NEO4J_URI")):
        pytest.skip("OPENAI_API_KEY and NEO4J_URI required; skipping credentials-required test.")
    return Neo4jConnection()


# --- Isolation: a caller sees its own edges and nothing else -----------------------------------------

ORG = "Acme"
SYS_A = "Helios"  # only group:a may see this
SYS_B = "Borealis"  # only group:b may see this
SHARED = "Mercury"  # the SAME fact asserted by two docs with different ACLs (merge-bug regression)
GROUP_A = "group:a"
GROUP_B = "group:b"


@pytest.fixture(scope="module")
def ingested(graph_connection):
    """Wipe the graph, then ingest the ACL-scoped documents once for the module."""
    driver = graph_connection.connect()
    with driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n").consume()  # clean slate -> isolation without name suffixes
    driver.close()

    extractor = KnowledgeGraphEntityExtractor(llm=_llm(), ontology=ONTOLOGY)
    extractor.init_components()
    writer = KnowledgeGraphWriter(connection=graph_connection, entity_embedder=_doc_embedder())
    writer.init_components()
    docs = [
        Document(
            content=f"{ORG} uses a system called {SYS_A}.",
            metadata={"allowed_principals": [GROUP_A]},
        ),
        Document(
            content=f"{ORG} uses a system called {SYS_B}.",
            metadata={"allowed_principals": [GROUP_B]},
        ),
        # The SAME fact asserted by two documents with DIFFERENT ACLs — must stay two edges, not merge.
        Document(
            content=f"{ORG} uses a system called {SHARED}.",
            metadata={"allowed_principals": [GROUP_A]},
        ),
        Document(
            content=f"{ORG} uses a system called {SHARED}.",
            metadata={"allowed_principals": [GROUP_B]},
        ),
    ]
    extraction = extractor.execute(KnowledgeGraphEntityExtractor.input_schema(documents=docs))
    result = writer.execute(
        KnowledgeGraphWriter.input_schema(
            nodes=extraction["nodes"],
            relationships=extraction["relationships"],
        )
    )
    assert result["relationships_created"] is not None
    yield
    writer._graph_store.close()


def _facts_by_name(graph_connection, principals, **kwargs):
    # ACL is expressed as a LOCKED filter via the contains_any operator (node config, not input). Seed by an
    # explicit `entities` name + a text_embedder (vector seeding) so the retriever needs no llm/ontology.
    retriever = KnowledgeGraphRetriever(
        connection=graph_connection,
        text_embedder=_text_embedder(),
        filters={
            "field": "allowed_principals",
            "operator": "contains_any",
            "value": principals,
        },
        **kwargs,
    )
    retriever.init_components()
    try:
        return retriever.execute(GraphRetrieverInputSchema(query=f"What systems does {ORG} use?", entities=[ORG]))[
            "content"
        ]
    finally:
        retriever._graph_store.close()


def test_principal_a_sees_only_a(ingested, graph_connection):
    content = _facts_by_name(graph_connection, [GROUP_A])
    assert SYS_A in content, f"group:a should see its own system {SYS_A}: {content}"
    assert SYS_B not in content, f"LEAK: group:a must not see group:b's system {SYS_B}: {content}"


def test_principal_b_sees_only_b(ingested, graph_connection):
    content = _facts_by_name(graph_connection, [GROUP_B])
    assert SYS_B in content
    assert SYS_A not in content, f"LEAK: group:b must not see group:a's system {SYS_A}: {content}"


def test_default_deny_hides_everything_for_unknown_principal(ingested, graph_connection):
    content = _facts_by_name(graph_connection, ["group:nobody"])
    assert SYS_A not in content and SYS_B not in content, f"unknown principal must see nothing: {content}"


def test_same_fact_from_two_docs_keeps_both_acls(ingested, graph_connection):
    # The SAME fact ("{ORG} uses {SHARED}") was asserted by a group:a doc AND a group:b doc. With the merge
    # bug the two collapse into one edge and the last writer's ACL wins — so one group would lose access.
    # With per-document edges, BOTH groups see it. unknown still sees nothing.
    assert SHARED in _facts_by_name(graph_connection, [GROUP_A]), "group:a lost a fact it asserted (ACL overwrite)"
    assert SHARED in _facts_by_name(graph_connection, [GROUP_B]), "group:b lost a fact it asserted (ACL overwrite)"
    assert SHARED not in _facts_by_name(graph_connection, ["group:nobody"]), "unknown principal must not see SHARED"


# --- Name provenance: endpoint names render from the edge snapshot, not the shared node --------------

ORG_ID = "org-1"
NODE_NAME = "Acme Secret Division"  # the shared node's stored (secret-encoded) name — MUST NOT leak
PUBLIC_NAME = "Acme"  # the public edge's own name snapshot — what a public caller should see
SYS_PUBLIC = "Helios"  # group:public may see this
SYS_SECRET = "Borealis"  # group:secret only
GROUP_PUBLIC = "group:public"
GROUP_SECRET = "group:secret"


@pytest.fixture(scope="module")
def shared_node_graph(graph_connection):
    """Wipe, then write ONE shared org node + two USES edges with distinct name snapshots and ACLs."""
    store = Neo4jGraphStore(connection=graph_connection, client=graph_connection.connect())
    store.run_cypher("MATCH (n) DETACH DELETE n")  # clean slate

    nodes = [
        {"labels": ["Organization", "Entity"], "id": ORG_ID, "name": NODE_NAME},
        {"labels": ["System", "Entity"], "id": "sys-pub", "name": SYS_PUBLIC},
        {"labels": ["System", "Entity"], "id": "sys-sec", "name": SYS_SECRET},
    ]

    def _edge(dst, src_name, dst_name, principal, doc_id):
        return {
            "type": "USES",
            "start_node": {"label": "Organization", "id": ORG_ID, "name": src_name},
            "end_node": {"label": "System", "id": dst, "name": dst_name},
            "identity_keys": ["source_doc_id"],
            "properties": {
                "allowed_principals": [principal],
                "source_doc_id": doc_id,
            },
        }

    relationships = [
        _edge("sys-pub", PUBLIC_NAME, SYS_PUBLIC, GROUP_PUBLIC, "docP"),
        _edge("sys-sec", NODE_NAME, SYS_SECRET, GROUP_SECRET, "docS"),
    ]
    store.write_graph(nodes=nodes, relationships=relationships)
    yield
    store.close()


def _facts_by_id(graph_connection, principals):
    retriever = KnowledgeGraphRetriever(
        connection=graph_connection,
        text_embedder=_text_embedder(),
        filters={
            "field": "allowed_principals",
            "operator": "contains_any",
            "value": principals,
        },
    )
    retriever.init_components()
    try:
        # Anchor by id (skips LLM entity extraction) so the assertion is deterministic.
        out = retriever.execute(GraphRetrieverInputSchema(query="What does the org use?", entity_ids=[ORG_ID]))
        return out["content"]
    finally:
        retriever._graph_store.close()


def test_public_caller_sees_edge_snapshot_not_shared_node_name(shared_node_graph, graph_connection):
    content = _facts_by_id(graph_connection, [GROUP_PUBLIC])
    assert SYS_PUBLIC in content, f"public caller should see its own system {SYS_PUBLIC}: {content}"
    assert PUBLIC_NAME in content, f"public caller should render its edge's own name snapshot: {content}"
    # The shared node's stored (secret-encoded) name must never reach a public-only caller.
    assert NODE_NAME not in content, f"LEAK: shared-node name {NODE_NAME!r} surfaced to public caller: {content}"
    assert SYS_SECRET not in content, f"LEAK: public caller must not see secret system {SYS_SECRET}: {content}"


def test_secret_caller_sees_its_own_edge(shared_node_graph, graph_connection):
    content = _facts_by_id(graph_connection, [GROUP_SECRET])
    assert SYS_SECRET in content
    assert SYS_PUBLIC not in content, f"LEAK: secret caller must not see public system {SYS_PUBLIC}: {content}"


def test_name_leak_unknown_principal_sees_nothing(shared_node_graph, graph_connection):
    content = _facts_by_id(graph_connection, ["group:nobody"])
    assert SYS_PUBLIC not in content and SYS_SECRET not in content, f"unknown principal must see nothing: {content}"


# --- No widening: caller filters can only narrow within the locked ACL -------------------------------

FILTER_ORG = "org-acl"


def _write_tiered(store, edges):
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


def _retrieve(graph_connection, *, locked, input_filters=None):
    retriever = KnowledgeGraphRetriever(connection=graph_connection, text_embedder=_text_embedder(), filters=locked)
    retriever.init_components()
    try:
        return retriever.execute(
            GraphRetrieverInputSchema(
                query="What does the org use?",
                entity_ids=[FILTER_ORG],
                filters=input_filters,
            )
        )["content"]
    finally:
        retriever._graph_store.close()


def test_input_filter_only_narrows_within_the_locked_acl(graph_connection):
    """The node's ``filters`` are LOCKED; input ``filters`` are AND-ed on top and can only further restrict.
    Two systems share the same ACL but differ by a ``tier`` property: the locked ACL alone returns both;
    adding an input filter ``tier == "low"`` returns only the low one (never widens the ACL)."""
    store = Neo4jGraphStore(connection=graph_connection, client=graph_connection.connect())
    store.run_cypher("MATCH (n) DETACH DELETE n")  # clean slate
    try:
        _write_tiered(
            store,
            [
                ("Helios", {"allowed_principals": ["group:x"], "tier": "low"}, "docL"),
                (
                    "Borealis",
                    {"allowed_principals": ["group:x"], "tier": "high"},
                    "docH",
                ),
            ],
        )
        acl = {
            "field": "allowed_principals",
            "operator": "contains_any",
            "value": ["group:x"],
        }

        both = _retrieve(graph_connection, locked=acl)
        assert "Helios" in both and "Borealis" in both, f"locked ACL alone should return both systems: {both}"

        narrowed = _retrieve(
            graph_connection,
            locked=acl,
            input_filters={"field": "tier", "operator": "==", "value": "low"},
        )
        assert "Helios" in narrowed, f"input filter tier==low should keep the low-tier system: {narrowed}"
        assert "Borealis" not in narrowed, f"input filter should exclude the high-tier system: {narrowed}"
    finally:
        store.close()
