"""Answer questions using BOTH the vector store and the knowledge graph (GraphRAG).

An Agent is given three tools and decides which to use (or several) per question:

  - VectorStoreRetriever       → semantic search over the Qdrant vector store (unstructured context).
  - KnowledgeGraphRetriever    → bounded, ACL-filtered facts from the Neo4j knowledge graph.
  - CypherExecutor             → raw Cypher queries against the Neo4j knowledge graph (power tool).

Run ``kg_ingestion.py`` first to populate both stores, then run this script.

Two modes:
  - ``python kg_question_answering.py ["your question"]`` — the GraphRAG agent answers a question.
  - ``python kg_question_answering.py --demo-acl``       — a self-contained demonstration that
    KnowledgeGraphRetriever WITHHOLDS facts from a caller whose principals don't match an edge's ACL (access-control
    denial). It seeds its OWN access-scoped edges and deletes them afterwards, so it needs a running Neo4j
    but NOT the ingested demo data.

Requirements (same as kg_ingestion.py):
  - OPENAI_API_KEY (embeddings + the agent LLM).
  - A running Neo4j with NEO4J_URI / NEO4J_USERNAME / NEO4J_PASSWORD set.
  - The local on-disk Qdrant produced by kg_ingestion.py (QDRANT_PATH below) — QA mode only.
"""

from qdrant_client import QdrantClient

from dynamiq import Workflow
from dynamiq.connections import Neo4j as Neo4jConnection
from dynamiq.connections import OpenAI as OpenAIConnection
from dynamiq.flows import Flow
from dynamiq.nodes.agents import Agent
from dynamiq.nodes.embedders import OpenAITextEmbedder
from dynamiq.nodes.knowledge_graph import KnowledgeGraphRetriever, Ontology
from dynamiq.nodes.knowledge_graph.retriever import GraphRetrieverInputSchema
from dynamiq.nodes.llms.openai import OpenAI
from dynamiq.nodes.retrievers import QdrantDocumentRetriever, VectorStoreRetriever
from dynamiq.nodes.tools import CypherExecutor
from dynamiq.nodes.types import InferenceMode
from dynamiq.runnables import RunnableConfig, RunnableStatus
from dynamiq.storages.graph.neo4j import Neo4jGraphStore
from dynamiq.storages.vector.qdrant.qdrant import QdrantVectorStore
from dynamiq.utils.logger import logger

# Must match kg_ingestion.py.
QDRANT_PATH = "./.qdrant_kg_demo2"
INDEX_NAME = "kg_demo"
EMBEDDING_MODEL = "text-embedding-3-small"

# Caller's access principals — chosen by the workflow AUTHOR (here, a constant), NOT by the agent.
# Only graph edges whose `allowed_principals` intersect this list are visible to graph-retriever.
# This is the trust boundary: the LLM supplies the question, the author supplies who-is-asking.
PRINCIPALS = ["group:public"]

AGENT_ROLE = """You answer questions about the ingested content using three tools:
- graph-retriever: bounded, access-controlled facts from the knowledge graph for a question. PREFER this
  for relationship/structured questions ("how is X connected to Y", "what does X use").
- vector-search: unstructured context retrieved from documents. Use for descriptive "what is / explain".
- knowledge-graph (Cypher): the power tool for custom graph queries when graph-retriever is not enough.
  Introspect the schema first, then ALWAYS query actual nodes/relationships — never answer from the
  schema alone.

Combine evidence when helpful, and cite which tool gave you each fact."""

# Default suits the kg_ingestion.py demo data; pass your own question as the first CLI argument,
# e.g.:  python kg_question_answering.py "What are the main components?"
DEFAULT_QUESTION = "Who is the CIO of Acme Capital, and what AI system does the firm use? How are they connected?"


def build_workflow() -> Workflow:
    openai_connection = OpenAIConnection()

    # Explicit on-disk client (same as kg_ingestion.py): QdrantVectorStore otherwise eagerly
    # connects to a *server* and ignores `path`. force_disable_check_same_thread is required
    # because the Flow runs the retriever on a worker thread over SQLite-backed local Qdrant.
    vector_store = QdrantVectorStore(
        client=QdrantClient(path=QDRANT_PATH, force_disable_check_same_thread=True),
        index_name=INDEX_NAME,
        create_if_not_exist=False,
        dimension=1536,
    )

    # Tool 1: semantic search over the vector store.
    vector_tool = VectorStoreRetriever(
        name="vector-search",
        text_embedder=OpenAITextEmbedder(connection=openai_connection, model=EMBEDDING_MODEL),
        document_retriever=QdrantDocumentRetriever(vector_store=vector_store),
        top_k=4,
    )

    # Tool 2: bounded, ACL-filtered graph context. The `filters` are LOCKED node config (not on the
    # tool's input schema), so the agent cannot drop or widen them — the controlled alternative to
    # LLM-written Cypher. Filters use the same structured grammar as the vector-store retrievers; ACL is
    # expressed via the `contains_any` operator on the edge ACL property.
    graph_tool = KnowledgeGraphRetriever(
        name="graph-retriever",
        connection=Neo4jConnection(),
        # Same entity types the graph was ingested with, so the question is parsed for those kinds.
        llm=OpenAI(connection=openai_connection, model="gpt-4o-mini", temperature=0),
        ontology=Ontology(
            entity_types=["Person", "Organization", "System", "Event", "Location"],
            relationship_types=["WORKS_AT", "USES", "PRESENTED", "PRESENTED_AT", "LOCATED_IN"],
        ),
        filters={"field": "allowed_principals", "operator": "contains_any", "value": PRINCIPALS},
        top_k=20,
    )

    # Tool 3: Cypher power tool for queries graph-retriever can't express.
    cypher_tool = CypherExecutor(name="knowledge-graph", connection=Neo4jConnection())

    agent = Agent(
        name="graphrag-agent",
        id="graphrag_agent",
        llm=OpenAI(connection=openai_connection, model="gpt-4o-mini", temperature=0.0, max_tokens=4000),
        role=AGENT_ROLE,
        tools=[vector_tool, graph_tool, cypher_tool],
        inference_mode=InferenceMode.XML,
        max_loops=12,
    )

    return Workflow(flow=Flow(nodes=[agent]))


# ---------------------------------------------------------------------------
# ACL denial demonstration (`--demo-acl`)
# ---------------------------------------------------------------------------
# Self-contained proof that KnowledgeGraphRetriever hides facts a caller isn't entitled to. It seeds two
# access-scoped edges from ONE org (a public system + a restricted system), then runs the SAME anchored
# query as three callers. The only thing that changes between runs is the caller's principals (the locked
# ACL filter), so any difference in what comes back is the access control at work. Uses uniquely-prefixed
# ids and deletes exactly those afterwards — it never touches the ingested demo graph.
_ACL_DEMO_ORG = "acl-demo-org"
_ACL_DEMO_SYS_PUBLIC = "acl-demo-sys-public"
_ACL_DEMO_SYS_RESTRICTED = "acl-demo-sys-restricted"
_ACL_DEMO_IDS = [_ACL_DEMO_ORG, _ACL_DEMO_SYS_PUBLIC, _ACL_DEMO_SYS_RESTRICTED]
_ACL_DEMO_PUBLIC_SYSTEM = "HeliosDemo"  # reachable only via the group:public edge
_ACL_DEMO_RESTRICTED_SYSTEM = "BorealisDemo"  # reachable only via the group:restricted edge


def _seed_acl_demo_edges(store: Neo4jGraphStore) -> None:
    """Write one org node and two USES edges to it, each carrying its own single-principal ACL."""
    nodes = [
        {"labels": ["Organization", "Entity"], "id": _ACL_DEMO_ORG, "name": "AcmeDemo", "properties": {}},
        {
            "labels": ["System", "Entity"],
            "id": _ACL_DEMO_SYS_PUBLIC,
            "name": _ACL_DEMO_PUBLIC_SYSTEM,
            "properties": {},
        },
        {
            "labels": ["System", "Entity"],
            "id": _ACL_DEMO_SYS_RESTRICTED,
            "name": _ACL_DEMO_RESTRICTED_SYSTEM,
            "properties": {},
        },
    ]

    def _edge(dst: str, dst_name: str, principal: str, doc_id: str) -> dict:
        # ACL lives on the EDGE (allowed_principals); src/dst names are per-edge snapshots.
        return {
            "type": "USES",
            "start_label": "Organization",
            "end_label": "System",
            "start_identity": _ACL_DEMO_ORG,
            "end_identity": dst,
            "src_name": "AcmeDemo",
            "dst_name": dst_name,
            "identity_keys": ["source_doc_id"],
            "properties": {
                "allowed_principals": [principal],
                "source_doc_id": doc_id,
            },
        }

    store.write_graph(
        nodes=nodes,
        relationships=[
            _edge(_ACL_DEMO_SYS_PUBLIC, _ACL_DEMO_PUBLIC_SYSTEM, "group:public", "acl-demo-docP"),
            _edge(_ACL_DEMO_SYS_RESTRICTED, _ACL_DEMO_RESTRICTED_SYSTEM, "group:restricted", "acl-demo-docS"),
        ],
    )


def _delete_acl_demo_edges(store: Neo4jGraphStore) -> None:
    """Remove ONLY the seeded demo nodes/edges (leaves the ingested graph untouched)."""
    store.run_cypher("MATCH (n:Entity) WHERE n.id IN $ids DETACH DELETE n", parameters={"ids": _ACL_DEMO_IDS})


def _graph_facts_for(openai_connection: OpenAIConnection, principals: list[str]) -> str:
    """Run graph-retriever anchored on the demo org for a caller with ``principals``; return its facts.

    Anchored by ``entity_ids`` so there's NO LLM entity-extraction randomness — the caller's principals
    (the locked ACL filter) are the only variable across the three runs.
    """
    retriever = KnowledgeGraphRetriever(
        name="graph-retriever",
        connection=Neo4jConnection(),
        llm=OpenAI(connection=openai_connection, model="gpt-4o-mini", temperature=0),
        ontology=Ontology(entity_types=["Organization", "System"], relationship_types=["USES"]),
        filters={"field": "allowed_principals", "operator": "contains_any", "value": principals},
    )
    retriever.init_components()
    try:
        out = retriever.execute(GraphRetrieverInputSchema(query="What does the org use?", entity_ids=[_ACL_DEMO_ORG]))
        return out["content"]
    finally:
        retriever._graph_store.close()


def _require(condition: bool, message: str) -> None:
    """Fail loudly if an ACL invariant is violated (a plain raise, so it survives ``python -O``)."""
    if not condition:
        raise RuntimeError(message)


def demonstrate_acl_denial() -> None:
    """Prove access-control denial: same query, three callers, only the entitled ones see each fact."""
    openai_connection = OpenAIConnection()
    graph_connection = Neo4jConnection()
    store = Neo4jGraphStore(connection=graph_connection, client=graph_connection.connect())
    try:
        _seed_acl_demo_edges(store)

        public_facts = _graph_facts_for(openai_connection, ["group:public"])
        restricted_facts = _graph_facts_for(openai_connection, ["group:restricted"])
        nobody_facts = _graph_facts_for(openai_connection, ["group:nobody"])

        logger.info("\n=== ACL denial demonstration (identical query, different callers) ===")
        logger.info(f"caller [group:public] sees -> {public_facts!r}")
        logger.info(f"caller [group:restricted] sees -> {restricted_facts!r}")
        logger.info(f"caller [group:nobody] sees -> {nobody_facts!r}")

        # Each caller sees ONLY the system its principal is entitled to; the unknown caller sees neither.
        _require(
            _ACL_DEMO_PUBLIC_SYSTEM in public_facts and _ACL_DEMO_RESTRICTED_SYSTEM not in public_facts,
            f"public caller should see only the public system: {public_facts!r}",
        )
        _require(
            _ACL_DEMO_RESTRICTED_SYSTEM in restricted_facts and _ACL_DEMO_PUBLIC_SYSTEM not in restricted_facts,
            f"restricted caller should see only the restricted system: {restricted_facts!r}",
        )
        _require(
            _ACL_DEMO_PUBLIC_SYSTEM not in nobody_facts and _ACL_DEMO_RESTRICTED_SYSTEM not in nobody_facts,
            f"unknown principal must be DENIED all facts: {nobody_facts!r}",
        )
        logger.info("Verified: relevant facts exist, but each caller retrieved only what its ACL allows. ✅")
    finally:
        _delete_acl_demo_edges(store)
        store.close()


if __name__ == "__main__":
    import sys

    args = sys.argv[1:]
    if args and args[0] == "--demo-acl":
        demonstrate_acl_denial()
        sys.exit(0)

    question = args[0] if args else DEFAULT_QUESTION
    workflow = build_workflow()

    try:
        result = workflow.run(
            input_data={"input": question},
            config=RunnableConfig(request_timeout=180),
        )

        if result.status != RunnableStatus.SUCCESS:
            raise RuntimeError(f"QA failed: {result.status} / {result.output}")

        answer = result.output["graphrag_agent"]["output"]["content"]
        logger.info(f"\nQuestion: {question}\n\nAnswer:\n{answer}")
    finally:
        # Close the agent tools' clients: the Neo4j Bolt driver (avoids the unclosed-driver
        # ResourceWarning) and the on-disk Qdrant client (releases the SQLite lock).
        for node in workflow.flow.nodes:
            for tool in getattr(node, "tools", []) or []:
                client = getattr(tool, "client", None)
                if client is not None and hasattr(client, "close"):
                    client.close()
                retriever = getattr(tool, "document_retriever", None)
                vector_store = getattr(retriever, "vector_store", None)
                if vector_store is not None and getattr(vector_store, "_client", None) is not None:
                    vector_store._client.close()
