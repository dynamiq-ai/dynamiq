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

from dotenv import load_dotenv

load_dotenv()

from dynamiq import Workflow
from dynamiq.connections import Neo4j as Neo4jConnection
from dynamiq.connections import OpenAI as OpenAIConnection
from dynamiq.flows import Flow
from dynamiq.nodes.agents import Agent
from dynamiq.nodes.embedders import OpenAITextEmbedder
from dynamiq.nodes.llms.openai import OpenAI
from dynamiq.nodes.retrievers import QdrantDocumentRetriever, VectorStoreRetriever
from dynamiq.nodes.tools import CypherExecutor
from dynamiq.nodes.types import InferenceMode
from dynamiq.runnables import RunnableConfig, RunnableStatus
from dynamiq.storages.vector.qdrant.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

from dynamiq.utils.logger import logger

# Must match kg_ingestion.py.
QDRANT_PATH = "./.qdrant_kg_demo2"
INDEX_NAME = "kg_demo"
EMBEDDING_MODEL = "text-embedding-3-small"

AGENT_ROLE = """You answer questions about firms, people, and AI systems using two tools:
- a vector-search tool for unstructured context retrieved from documents, and
- a Cypher tool for querying a Neo4j knowledge graph of entities and their relationships.

Strategy:
1. Use the Cypher tool's introspect mode first to learn the graph schema (labels/relationship types)
   before writing queries.
2. Use the knowledge graph for structured/relationship questions (who works where, what connects to what).
3. Use the vector-search tool for descriptive or "what is / explain" questions.
4. Combine evidence from both when helpful, and cite which tool gave you each fact."""

QUESTION = "Who is the CIO of Acme Capital, and what AI system does the firm use? How are they connected?"


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

    # Tool 2: Cypher queries over the knowledge graph.
    cypher_tool = CypherExecutor(name="knowledge-graph", connection=Neo4jConnection())

    agent = Agent(
        name="graphrag-agent",
        id="graphrag_agent",
        llm=OpenAI(connection=openai_connection, model="gpt-4o-mini", temperature=0.0, max_tokens=4000),
        role=AGENT_ROLE,
        tools=[vector_tool, cypher_tool],
        inference_mode=InferenceMode.XML,
        max_loops=12,
    )

    return Workflow(flow=Flow(nodes=[agent]))


if __name__ == "__main__":
    workflow = build_workflow()

    result = workflow.run(
        input_data={"input": QUESTION},
        config=RunnableConfig(request_timeout=180),
    )

    assert result.status == RunnableStatus.SUCCESS, f"QA failed: {result.status} / {result.output}"

    answer = result.output["graphrag_agent"]["output"]["content"]
    logger.info(f"\nQuestion: {QUESTION}\n\nAnswer:\n{answer}")
