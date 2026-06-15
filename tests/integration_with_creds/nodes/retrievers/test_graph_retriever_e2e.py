"""E2E ACL test for GraphRetriever against a real Neo4j (+ OPENAI for extraction).

Ingests two documents about the same organisation but with DIFFERENT ``allowed_principals``, then
asserts that retrieving as one principal returns that principal's facts and NEVER leaks the other's —
the core promise of edge-level ACL enforcement compiled server-side (not LLM-written Cypher).

Requires NEO4J_URI / NEO4J_USERNAME / NEO4J_PASSWORD and OPENAI_API_KEY. Skipped otherwise. The test
is robust to pre-existing graph data: it ingests two distinctively-named systems and only asserts on
the visibility of those two names, so it does not clear the database.
"""

import os
import uuid

import pytest

from dynamiq.connections import Neo4j as Neo4jConnection
from dynamiq.connections import OpenAI as OpenAIConnection
from dynamiq.nodes.extractors import KnowledgeGraphWriter, Ontology
from dynamiq.nodes.llms.openai import OpenAI
from dynamiq.nodes.retrievers import GraphRetriever
from dynamiq.nodes.retrievers.graph import GraphRetrieverInputSchema
from dynamiq.types import Document

# Unique per run so concurrent/leftover data cannot interfere with the assertions.
RUN = uuid.uuid4().hex[:8]
ORG = f"Acme-{RUN}"
SYS_A = f"Helios-{RUN}"  # only group:a may see this
SYS_B = f"Borealis-{RUN}"  # only group:b may see this
GROUP_A = "group:a"
GROUP_B = "group:b"

ONTOLOGY = Ontology(entity_types=["Organization", "System"], relationship_types=["USES"])


@pytest.fixture(scope="module")
def graph_connection():
    if not (os.getenv("OPENAI_API_KEY") and os.getenv("NEO4J_URI")):
        pytest.skip("OPENAI_API_KEY and NEO4J_URI required; skipping credentials-required test.")
    return Neo4jConnection()


@pytest.fixture(scope="module")
def ingested(graph_connection):
    """Ingest the two ACL-scoped documents once for the module."""
    writer = KnowledgeGraphWriter(
        llm=OpenAI(connection=OpenAIConnection(), model="gpt-4o-mini", temperature=0),
        connection=graph_connection,
        ontology=ONTOLOGY,
    )
    writer.init_components()
    docs = [
        Document(content=f"{ORG} uses a system called {SYS_A}.", metadata={"allowed_principals": [GROUP_A]}),
        Document(content=f"{ORG} uses a system called {SYS_B}.", metadata={"allowed_principals": [GROUP_B]}),
    ]
    result = writer.execute(KnowledgeGraphWriter.input_schema(documents=docs))
    assert result["relationships_created"] is not None
    yield
    writer._graph_store.close()


def _facts_for(graph_connection, principals, **kwargs):
    # ACL is expressed as a LOCKED filter via the $intersects operator (node config, not input).
    retriever = GraphRetriever(
        connection=graph_connection,
        filters={"allowed_principals": {"$intersects": principals}},
        **kwargs,
    )
    retriever.init_components()
    try:
        out = retriever.execute(GraphRetrieverInputSchema(query=f"What systems does {ORG} use?"))
        return out["content"]
    finally:
        retriever._graph_store.close()


def test_principal_a_sees_only_a(ingested, graph_connection):
    content = _facts_for(graph_connection, [GROUP_A])
    assert SYS_A in content, f"group:a should see its own system {SYS_A}: {content}"
    assert SYS_B not in content, f"LEAK: group:a must not see group:b's system {SYS_B}: {content}"


def test_principal_b_sees_only_b(ingested, graph_connection):
    content = _facts_for(graph_connection, [GROUP_B])
    assert SYS_B in content
    assert SYS_A not in content, f"LEAK: group:b must not see group:a's system {SYS_A}: {content}"


def test_default_deny_hides_everything_for_unknown_principal(ingested, graph_connection):
    content = _facts_for(graph_connection, ["group:nobody"])
    assert SYS_A not in content and SYS_B not in content, f"unknown principal must see nothing: {content}"
