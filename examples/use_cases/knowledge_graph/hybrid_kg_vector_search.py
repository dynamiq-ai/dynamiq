"""
Hybrid search that combines a vector store with per-chunk knowledge-graph context (the "Onyx" pattern).

Idea
----
Chunk first, THEN extract a knowledge graph per chunk. ``KnowledgeGraphWriter`` (an ``EntityExtractor``
subclass) does two things in one pass: it WRITES the extracted graph to the graph store (Neo4j here),
and it returns each chunk tagged with the RESOLVED, unique ids of the entities it mentions
(``kg_entity_ids`` metadata) — the SAME chunk that gets embedded. Those chunks go straight into the
normal vector store. At query time the hybrid retriever vector-searches passages, then seeds the graph by
the ``kg_entity_ids`` of those passages (unique, variant-proof) to pull the related facts.

So one node links the knowledge graph to the vector store: the graph is queryable on its own, and each
chunk carries the exact entity ids needed to expand it into facts at query time.

The dense embedding stays on clean chunk content — entity names already appear in the content, so the
keyword side of hybrid ranks on them; ``kg_entity_ids`` is the precise link into the graph.

Pipeline:
    Input(docs, query)
        └─> splitter ─> kg_writer ─> doc_embedder ─> pgvector_writer ─> hybrid_retriever ─> Output
                          (kg_writer also writes Neo4j)        hybrid_retriever fuses a vector retriever
                                                               (pgvector) + a graph retriever (Neo4j)

Prerequisites
-------------
1. Local Postgres with pgvector (the writer auto-creates the extension):

       docker run -d --name dynamiq-pgvector \
         -e POSTGRES_PASSWORD=password -e POSTGRES_DB=db \
         -p 5432:5432 pgvector/pgvector:pg16

2. Local Neo4j (the knowledge graph store):

       docker run -d --name dynamiq-neo4j \
         -e NEO4J_AUTH=neo4j/password -p 7474:7474 -p 7687:7687 neo4j:5

3. OPENAI_API_KEY in your environment / .env (LLM extraction + embeddings).

Run:  python examples/use_cases/knowledge_graph/hybrid_kg_vector_search.py
"""

import os

from dynamiq import Workflow
from dynamiq.connections import Neo4j as Neo4jConnection
from dynamiq.connections import OpenAI as OpenAIConnection
from dynamiq.connections import PostgreSQL as PostgreSQLConnection
from dynamiq.flows import Flow
from dynamiq.nodes.embedders.openai import OpenAIDocumentEmbedder, OpenAITextEmbedder
from dynamiq.nodes.extractors import EntityExtractor, KnowledgeGraphWriter, Ontology
from dynamiq.nodes.llms.openai import OpenAI
from dynamiq.nodes.node import InputTransformer, NodeDependency
from dynamiq.nodes.retrievers.graph import GraphRetriever
from dynamiq.nodes.retrievers.hybrid_graph_vector import HybridGraphVectorRetriever
from dynamiq.nodes.retrievers.pgvector import PGVectorDocumentRetriever
from dynamiq.nodes.retrievers.retriever import VectorStoreRetriever
from dynamiq.nodes.splitters.document import DocumentSplitBy, DocumentSplitter
from dynamiq.nodes.utils.utils import Input, Output
from dynamiq.nodes.writers.pgvector import PGVectorDocumentWriter
from dynamiq.runnables import RunnableConfig
from dynamiq.types import Document

TABLE = "demo_kg_hybrid"
DIM = 1536  # text-embedding-3-small

ONTOLOGY = Ontology(
    entity_types=["Person", "Organization", "City", "Technology"],
    relationship_types=["WORKS_AT", "LOCATED_IN", "USES", "FOUNDED"],
)


def build_workflow() -> Workflow:
    pg = PostgreSQLConnection()
    openai = OpenAIConnection()
    neo4j = Neo4jConnection(
        uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        username=os.getenv("NEO4J_USERNAME", "neo4j"),
        password=os.getenv("NEO4J_PASSWORD", "password"),
    )

    entry = Input(id="entry")

    # 1. Chunk FIRST — the chunk is the unit we extract from AND embed (so the attached graph is relevant).
    splitter = DocumentSplitter(
        id="splitter",
        split_by=DocumentSplitBy.SENTENCE,
        split_length=3,
        split_overlap=1,
        depends=[NodeDependency(node=entry)],
        input_transformer=InputTransformer(selector={"documents": "$.entry.output.documents"}),
    )

    # 2a. Per-chunk KG extraction (LLM). Separate node so it can be parallelized; emits {nodes,
    #     relationships, documents}.
    entity_extractor = EntityExtractor(
        id="entity_extractor",
        llm=OpenAI(connection=openai, model="gpt-4o-mini"),
        ontology=ONTOLOGY,
        depends=[NodeDependency(node=splitter)],
        input_transformer=InputTransformer(selector={"documents": "$.splitter.output.documents"}),
    )

    # 2b. Resolve duplicates + WRITE to Neo4j. Also returns the SAME chunks, tagged with kg_entity_ids
    #     (the resolved, unique ids each chunk mentions) for the vector store.
    kg_writer = KnowledgeGraphWriter(
        id="kg_writer",
        connection=neo4j,
        depends=[NodeDependency(node=entity_extractor)],
        input_transformer=InputTransformer(
            selector={
                "nodes": "$.entity_extractor.output.nodes",
                "relationships": "$.entity_extractor.output.relationships",
                "documents": "$.entity_extractor.output.documents",
            }
        ),
    )

    # 3. Embed CLEAN chunk content (no graph text folded in — entity names already live in the content).
    doc_embedder = OpenAIDocumentEmbedder(
        id="doc_embedder",
        model="text-embedding-3-small",
        depends=[NodeDependency(node=kg_writer)],
        input_transformer=InputTransformer(selector={"documents": "$.kg_writer.output.documents"}),
    )

    # 4. Persist content + embedding + kg_* metadata into pgvector.
    writer = PGVectorDocumentWriter(
        id="writer",
        connection=pg,
        table_name=TABLE,
        dimension=DIM,
        create_if_not_exist=True,
        depends=[NodeDependency(node=doc_embedder)],
        input_transformer=InputTransformer(selector={"documents": "$.doc_embedder.output.documents"}),
    )

    # 5. The hybrid retriever COMPOSES two retrievers (not added to the flow — bundled like an embedder):
    #    a vector retriever over the enriched pgvector chunks, and a graph retriever over Neo4j. It vector-
    #    searches passages, seeds the graph with the entity ids those passages mention (kg_entity_ids),
    #    pulls the related facts, and returns one sectioned "## Passages / ## Facts" context.
    vector_retriever = VectorStoreRetriever(
        id="vector_retriever",
        text_embedder=OpenAITextEmbedder(connection=openai, model="text-embedding-3-small"),
        document_retriever=PGVectorDocumentRetriever(connection=pg, table_name=TABLE, dimension=DIM, top_k=3),
        alpha=0.5,  # dense + BM25 keyword (entity names live in the content)
    )
    graph_retriever = GraphRetriever(id="graph_retriever", connection=neo4j, max_depth=1, top_k=10)

    hybrid_retriever = HybridGraphVectorRetriever(
        id="hybrid_retriever",
        vector_retriever=vector_retriever,
        graph_retriever=graph_retriever,
        depends=[NodeDependency(node=writer)],  # retrieve only after the chunks are persisted
        input_transformer=InputTransformer(selector={"query": "$.entry.output.query"}),
    )

    exit_node = Output(
        id="exit",
        depends=[NodeDependency(node=hybrid_retriever)],
        input_transformer=InputTransformer(
            selector={
                "context": "$.hybrid_retriever.output.content",
                "results": "$.hybrid_retriever.output.documents",
            }
        ),
    )

    return Workflow(
        flow=Flow(
            nodes=[entry, splitter, entity_extractor, kg_writer, doc_embedder, writer, hybrid_retriever, exit_node]
        )
    )


def main() -> None:
    docs = [
        Document(
            content=(
                "Jane Doe works at Acme Corp, a company located in Berlin. "
                "Acme Corp uses PostgreSQL for its data platform. "
                "The Eiffel Tower is a wrought-iron landmark located in Paris."
            )
        ),
    ]

    result = build_workflow().run_sync(
        input_data={"documents": docs, "query": "Where is Jane employed and what tech do they use?"},
        config=RunnableConfig(callbacks=[]),
    )

    print("status:", result.status)
    output = result.output["exit"]["output"]
    # The fused, structured context an agent/LLM would consume directly.
    print("\n=== structured context ===\n")
    print(output["context"])
    # The same content as documents, each tagged origin = "passage" | "fact".
    print("\n=== documents ===")
    for doc in output["results"]:
        md = doc.get("metadata") or {}
        line = f"  [{md.get('origin')}] {doc['content']}"
        if md.get("origin") == "passage":
            line += f"   kg_entity_ids={md.get('kg_entity_ids')}"
        print(line)


if __name__ == "__main__":
    main()
