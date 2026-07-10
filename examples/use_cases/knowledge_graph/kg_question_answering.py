"""Answer questions using BOTH the vector store and the knowledge graph (GraphRAG).

An Agent is given two tools and decides which to use (or both) per question:

  - VectorStoreRetriever  → semantic search over the Qdrant vector store (unstructured context).
  - CypherExecutor        → Cypher queries against the Neo4j knowledge graph (entities/relationships).

Run ``kg_ingestion.py`` first to populate both stores, then run this script.

Requirements (same as kg_ingestion.py):
  - OPENAI_API_KEY (embeddings + the agent LLM).
  - A running Neo4j with NEO4J_URI / NEO4J_USERNAME / NEO4J_PASSWORD set.
  - The local on-disk Qdrant produced by kg_ingestion.py (QDRANT_PATH below).
"""

from qdrant_client import QdrantClient

from dynamiq import Workflow
from dynamiq.connections import Neo4j as Neo4jConnection
from dynamiq.connections import OpenAI as OpenAIConnection
from dynamiq.flows import Flow
from dynamiq.nodes.agents import Agent
from dynamiq.nodes.embedders import OpenAITextEmbedder
from dynamiq.nodes.knowledge_graph import KnowledgeGraphRetriever, Ontology
from dynamiq.nodes.llms.openai import OpenAI
from dynamiq.nodes.retrievers import QdrantDocumentRetriever, VectorStoreRetriever
from dynamiq.nodes.tools import CypherExecutor
from dynamiq.nodes.types import InferenceMode
from dynamiq.runnables import RunnableConfig, RunnableStatus
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
    # LLM-written Cypher. ACL is expressed via the $intersects operator on the edge ACL property.
    graph_tool = KnowledgeGraphRetriever(
        name="graph-retriever",
        connection=Neo4jConnection(),
        # Same entity types the graph was ingested with, so the question is parsed for those kinds.
        llm=OpenAI(connection=openai_connection, model="gpt-4o-mini", temperature=0),
        ontology=Ontology(
            entity_types=["Person", "Organization", "System", "Event", "Location"],
            relationship_types=["WORKS_AT", "USES", "PRESENTED", "PRESENTED_AT", "LOCATED_IN"],
        ),
        filters={"allowed_principals": {"$intersects": PRINCIPALS}},
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


if __name__ == "__main__":
    import sys

    question = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_QUESTION
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
