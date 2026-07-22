"""E2E for the RETRIEVE stage: what ``KnowledgeGraphRetriever`` returns from a real Neo4j.

This file asserts on the RETRIEVED FACTS — the retriever's functional capability, independent of access
control (that is ``test_acl_e2e.py``). It covers the two behaviours that only exist once entities/edges are
embedded (writer's ``entity_embedder`` + retriever's ``text_embedder``):

  * SEMANTIC SEEDING — a paraphrased query term reaches an entity stored under a different name, via the
    entity vector index (a match the full-text/CONTAINS name path can't make);
  * MULTI-HOP EDGE RANKING — at a hub with several branches, per-hop cosine ranking + a tight ``beam_width``
    keep only the branch relevant to the query and cut the rest.

Requires ``OPENAI_API_KEY`` and ``NEO4J_URI`` /``NEO4J_USERNAME`` /``NEO4J_PASSWORD``; skipped otherwise. In
CI a local ``neo4j:5-community`` container supplies Neo4j (docker-compose ``neo4j`` service) — no Aura creds.
Fixtures wipe the graph before ingesting and are ordered so each section's wipe follows the previous
section's assertions; run serially against a dedicated test instance.
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

EMBED_MODEL = "text-embedding-3-small"
GROUP_PUBLIC = "group:public"


def _llm():
    return OpenAI(connection=OpenAIConnection(), model="gpt-4o-mini", temperature=0)


def _doc_embedder():
    return OpenAIDocumentEmbedder(connection=OpenAIConnection(), model=EMBED_MODEL)


def _text_embedder():
    return OpenAITextEmbedder(connection=OpenAIConnection(), model=EMBED_MODEL)


@pytest.fixture(scope="module")
def graph_connection():
    if not (os.getenv("OPENAI_API_KEY") and os.getenv("NEO4J_URI")):
        pytest.skip("OPENAI_API_KEY and NEO4J_URI required; skipping credentials-required test.")
    return Neo4jConnection()


# --- 1. Multi-hop edge ranking -----------------------------------------------------------------------
# A hub (DataPlatform) depends on three databases in three regions (analytics / billing / search). The
# query asks only about analytics; with beam_width=1 the retriever returns the analytics branch and cuts
# the billing and search branches (reachable but lower-ranked against the query embedding).

PLATFORM = "DataPlatform"
RELEVANT_DB, RELEVANT_REGION = (
    "Borealis",
    "Ireland",
)  # the analytics branch the query is about
IRRELEVANT = [
    ("Ledger", "Virginia"),
    ("Sphinx", "Singapore"),
]  # billing + search branches

MULTIHOP_CORPUS = [
    f"{PLATFORM} depends on {RELEVANT_DB}, its customer analytics database.",
    f"The {RELEVANT_DB} database is hosted in the {RELEVANT_REGION} region.",
    f"{PLATFORM} depends on {IRRELEVANT[0][0]}, its billing and payments database.",
    f"The {IRRELEVANT[0][0]} database is hosted in the {IRRELEVANT[0][1]} region.",
    f"{PLATFORM} depends on {IRRELEVANT[1][0]}, its search indexing database.",
    f"The {IRRELEVANT[1][0]} database is hosted in the {IRRELEVANT[1][1]} region.",
]
MULTIHOP_QUESTION = f"In which region is {PLATFORM}'s customer analytics data stored?"
MULTIHOP_ONTOLOGY = Ontology(
    entity_types=["Project", "Database", "Region"],
    relationship_types=["DEPENDS_ON", "HOSTED_IN"],
)


@pytest.fixture(scope="module")
def multihop_graph(graph_connection):
    """Wipe, extract, and write the DataPlatform corpus WITH an entity_embedder — once, shared by depth cases."""
    driver = graph_connection.connect()
    with driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n").consume()  # clean slate -> deterministic isolation
    driver.close()

    extractor = KnowledgeGraphEntityExtractor(llm=_llm(), ontology=MULTIHOP_ONTOLOGY)
    extractor.init_components()
    writer = KnowledgeGraphWriter(connection=graph_connection, entity_embedder=_doc_embedder())
    writer.init_components()
    extraction = extractor.execute(
        KnowledgeGraphEntityExtractor.input_schema(documents=[Document(content=t) for t in MULTIHOP_CORPUS])
    )
    writer.execute(
        KnowledgeGraphWriter.input_schema(
            nodes=extraction["nodes"],
            relationships=extraction["relationships"],
        )
    )
    yield
    writer.graph_store.close()


def _multihop_retrieve(graph_connection, max_hops):
    """Retrieve the analytics facts at a given traversal depth (query-seeded, tight beam)."""
    retriever = KnowledgeGraphRetriever(
        connection=graph_connection,
        text_embedder=_text_embedder(),
        max_hops=max_hops,
        beam_width=1,  # keep only the most query-relevant edge per hop -> forces the branch choice
        top_k=6,
    )
    retriever.init_components()
    try:
        # No llm, no entities -> the whole question is embedded and seeds the entry entities (and ranks
        # each hop's edges). The beam then keeps only the analytics branch.
        return retriever.execute(GraphRetrieverInputSchema(query=MULTIHOP_QUESTION))["content"].lower()
    finally:
        retriever.graph_store.close()


@pytest.fixture(scope="module")
def multihop_facts(multihop_graph, graph_connection):
    """Facts at max_hops=2 — the full chain DataPlatform -> Borealis -> Ireland."""
    return _multihop_retrieve(graph_connection, max_hops=2)


@pytest.fixture(scope="module")
def multihop_facts_single_hop(multihop_graph, graph_connection):
    """Facts at max_hops=1 — a single hop can't cross the DB -> Region edge."""
    return _multihop_retrieve(graph_connection, max_hops=1)


def test_multihop_beam_picks_relevant_branch_and_ignores_the_rest(multihop_facts):
    # the analytics branch is found end-to-end (its database AND its region)...
    assert RELEVANT_DB.lower() in multihop_facts, multihop_facts
    assert RELEVANT_REGION.lower() in multihop_facts, multihop_facts
    # ...and the billing + search branches — reachable but off-topic — are cut by the embedding-ranked beam
    leaked = [name for pair in IRRELEVANT for name in pair if name.lower() in multihop_facts]
    assert not leaked, f"irrelevant branches leaked past the beam: {leaked}\n{multihop_facts}"


def test_single_hop_cannot_reach_the_region_two_hops_away(multihop_facts_single_hop):
    # One hop from the DataPlatform hub reaches its analytics database...
    assert RELEVANT_DB.lower() in multihop_facts_single_hop, multihop_facts_single_hop
    # ...but the region is a SECOND hop away (Borealis -[HOSTED_IN]-> Ireland), so max_hops=1 never reaches
    # it — this is the gap the multi-hop beam closes (asserted by the max_hops=2 test above).
    assert RELEVANT_REGION.lower() not in multihop_facts_single_hop, multihop_facts_single_hop


# --- 2. Semantic seeding -----------------------------------------------------------------------------
# The writer embeds entity NAMES; a paraphrased seed term is matched to the stored name by vector
# similarity. With unrelated distractor entities present and vector_top_k=1, the seed must pick the RIGHT
# entity out of several — a discrimination the full-text/CONTAINS name path cannot make.

VEC_TARGET_ORG = "Car Manufacturer"
VEC_TARGET_SYS = "Assembly Robotics"  # only the automaker uses this
VEC_DISTRACTORS = [
    ("Bakery", "Dough Mixer"),
    ("Law Firm", "Case Tracker"),
]  # reachable but semantically far


@pytest.fixture(scope="module")
def embedded_graph(graph_connection):
    """Wipe, then ingest the target org->system fact PLUS unrelated distractor facts through the writer WITH
    an entity embedder (real OpenAI), so semantic seeding has to CHOOSE among several stored entities."""
    store = Neo4jGraphStore(connection=graph_connection, client=graph_connection.connect())
    store.run_cypher("MATCH (n) DETACH DELETE n")  # clean slate

    nodes, relationships = [], []
    for i, (org, sys) in enumerate([(VEC_TARGET_ORG, VEC_TARGET_SYS), *VEC_DISTRACTORS]):
        oid, sid = f"org-{i}", f"sys-{i}"
        nodes.append(
            {"labels": ["Organization", "Entity"], "id": oid, "name": org}
        )
        nodes.append(
            {"labels": ["System", "Entity"], "id": sid, "name": sys}
        )
        relationships.append(
            {
                "type": "USES",
                "start_node": {"label": "Organization", "id": oid, "name": org},
                "end_node": {"label": "System", "id": sid, "name": sys},
                "identity_keys": ["source_doc_id"],
                "properties": {
                    "allowed_principals": [GROUP_PUBLIC],
                    "source_doc_id": f"doc-{i}",
                },
            }
        )
    writer = KnowledgeGraphWriter(connection=graph_connection, entity_embedder=_doc_embedder())
    writer.init_components()
    try:
        writer.execute(KnowledgeGraphWriter.input_schema(nodes=nodes, relationships=relationships))
    finally:
        writer.graph_store.close()
    yield
    store.close()


def test_paraphrased_query_seeds_the_right_entity_via_vector_similarity(embedded_graph, graph_connection):
    # "automaker" never appears as a stored name; only a semantic (vector) seed can reach "Car Manufacturer"
    # — and with vector_top_k=1 it must pick it OVER the bakery/law-firm distractors.
    retriever = KnowledgeGraphRetriever(
        connection=graph_connection,
        text_embedder=_text_embedder(),
        vector_top_k=1,  # seed only the single nearest entity -> forces the semantic choice, not "seed them all"
        filters={
            "field": "allowed_principals",
            "operator": "contains_any",
            "value": [GROUP_PUBLIC],
        },
    )
    retriever.init_components()
    assert retriever._use_vector, "vector seeding should be active (embedder set + vector index present)"
    try:
        # `entities` skips LLM extraction so the seed term is fixed; it is embedded and vector-matched.
        out = retriever.execute(GraphRetrieverInputSchema(query="What does the automaker use?", entities=["automaker"]))
        content = out["content"]
        assert VEC_TARGET_SYS in content, f"paraphrased vector seed should reach the target fact: {content}"
        leaked = [sys for _, sys in VEC_DISTRACTORS if sys in content]
        assert not leaked, f"vector seed picked unrelated entities instead of the automaker: {leaked}\n{content}"
    finally:
        retriever.graph_store.close()
