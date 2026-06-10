"""Ingest documents into BOTH a vector store and a knowledge graph in one workflow.

This shows the vector store and the knowledge graph as two INDEPENDENT branches of the same
ingestion flow (they are not bundled into one node):

    input.documents ─┬─► OpenAIDocumentEmbedder ─► QdrantDocumentWriter      (vector store)
                     └─► EntityExtractor (LLM)   ─► Neo4jGraphWriter         (knowledge graph)

Run this first, then run ``kg_question_answering.py`` to query both stores.

Requirements:
  - OPENAI_API_KEY (used for embeddings + entity extraction).
  - A running Neo4j. Locally:
        docker run -p 7687:7687 -p 7474:7474 -e NEO4J_AUTH=neo4j/password neo4j:5
    then set NEO4J_URI=bolt://localhost:7687, NEO4J_USERNAME=neo4j, NEO4J_PASSWORD=password.
  - Qdrant runs locally on disk (no server needed) under ``QDRANT_PATH`` below.
"""

from dotenv import load_dotenv

load_dotenv()

from dynamiq import Workflow
from dynamiq.connections import Neo4j as Neo4jConnection
from dynamiq.connections import OpenAI as OpenAIConnection
from dynamiq.flows import Flow
from dynamiq.nodes.embedders import OpenAIDocumentEmbedder
from dynamiq.nodes.extractors import EntityExtractor
from dynamiq.nodes.llms.openai import OpenAI
from dynamiq.nodes.node import InputTransformer, NodeDependency
from dynamiq.nodes.writers import Neo4jGraphWriter, QdrantDocumentWriter
from dynamiq.runnables import RunnableConfig, RunnableStatus
from dynamiq.storages.vector.qdrant.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from dynamiq.types import Document
from dynamiq.utils.logger import logger

# Local on-disk Qdrant — shared by the ingestion and QA scripts (no server needed).
QDRANT_PATH = "./.qdrant_kg_demo2"
INDEX_NAME = "kg_demo"
EMBEDDING_MODEL = "text-embedding-3-small"  # 1536 dims, matches Qdrant store default

DOCS = [
    Document(
        content=(
            "Acme Capital is a hedge fund based in New York. Jane Doe is the Chief Investment Officer "
            "at Acme Capital. The firm uses an agentic AI system called Helios for trade research."
        )
    ),
    Document(
        content=(
            "Helios was built on top of large language models and integrates web search tools. "
            "Jane Doe presented Helios at the 2025 Quant Summit in London."
        )
    ),
]


def build_workflow() -> Workflow:
    openai_connection = OpenAIConnection()

    # A single Qdrant store instance, reused by the writer (and later by the retriever).
    # NOTE: pass an explicit on-disk client. QdrantVectorStore eagerly builds a *server*
    # connection (localhost:6333) whenever no client is supplied, which ignores `path` and
    # fails with "Connection refused" when no Qdrant server is running. Handing it a local
    # QdrantClient(path=...) keeps everything on disk, no server needed.
    #
    # force_disable_check_same_thread=True: on-disk Qdrant is backed by SQLite, but the Flow
    # executes nodes on a worker-thread pool. Without this flag the writer intermittently hits
    # "SQLite objects created in a thread can only be used in that same thread".
    vector_store = QdrantVectorStore(
        client=QdrantClient(path=QDRANT_PATH, force_disable_check_same_thread=True),
        index_name=INDEX_NAME,
        create_if_not_exist=True,
        dimension=1536,
    )

    # ---- Vector branch: embed documents and write them to Qdrant ----
    document_embedder = OpenAIDocumentEmbedder(
        id="document_embedder",
        connection=openai_connection,
        model=EMBEDDING_MODEL,
    )
    vector_writer = QdrantDocumentWriter(
        id="vector_writer",
        vector_store=vector_store,
        depends=[NodeDependency(document_embedder)],
        input_transformer=InputTransformer(
            selector={"documents": f"$.{document_embedder.id}.output.documents"}
        ),
    )

    # ---- Graph branch: extract entities/relationships and write them to Neo4j ----
    extractor = EntityExtractor(
        id="entity_extractor",
        llm=OpenAI(connection=openai_connection, model="gpt-4o-mini", temperature=0.0, max_tokens=4000),
    )
    graph_writer = Neo4jGraphWriter(
        id="graph_writer",
        connection=Neo4jConnection(),
        depends=[NodeDependency(extractor)],
        input_transformer=InputTransformer(
            selector={
                "nodes": f"$.{extractor.id}.output.nodes",
                "relationships": f"$.{extractor.id}.output.relationships",
            }
        ),
    )

    # Both branches read the same workflow input ("documents").
    document_embedder.input_transformer = InputTransformer(selector={"documents": "$.documents"})
    extractor.input_transformer = InputTransformer(selector={"documents": "$.documents"})

    return Workflow(flow=Flow(nodes=[document_embedder, vector_writer, extractor, graph_writer]))


if __name__ == "__main__":
    workflow = build_workflow()

    try:
        result = workflow.run(
            input_data={"documents": DOCS},
            config=RunnableConfig(request_timeout=120),
        )

        assert result.status == RunnableStatus.SUCCESS, f"Ingestion failed: {result.status} / {result.output}"

        vector_out = result.output["vector_writer"]["output"]
        graph_out = result.output["graph_writer"]["output"]
        logger.info(f"Vector store: upserted {vector_out.get('upserted_count')} documents to Qdrant ('{INDEX_NAME}').")
        logger.info(
            f"Knowledge graph: created {graph_out.get('nodes_created')} nodes and "
            f"{graph_out.get('relationships_created')} relationships in Neo4j."
        )
        logger.info("--- Ingestion complete. Run kg_question_answering.py next. ---")
    finally:
        # Explicitly close stores so resources are released cleanly:
        #  - Neo4j: avoids the "unclosed BoltDriver" ResourceWarning.
        #  - Qdrant: local on-disk mode only FLUSHES to disk on close(), so without this the
        #    upserted vectors are lost when the process exits and the QA script finds nothing.
        for node in workflow.flow.nodes:
            graph_store = getattr(node, "_graph_store", None)
            if graph_store is not None:
                graph_store.close()
            vector_store = getattr(node, "vector_store", None)
            if vector_store is not None and getattr(vector_store, "_client", None) is not None:
                vector_store._client.close()
