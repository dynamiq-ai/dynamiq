"""Full ingestion workflow: documentation URL -> chunks -> vector store + knowledge graph.

Pipeline (one workflow, five nodes). Each branch gets its own chunk granularity — splitting is
free, so neither branch has to compromise:

    fetched HTML ─► HTMLConverter ─┬─► fine splitter (12 sent.)  ─► OpenAIDocumentEmbedder ─► QdrantDocumentWriter
                                   └─► coarse splitter (50 sent.) ─► KnowledgeGraphWriter (extract + resolve + Neo4j)

- HTMLConverter turns the fetched page into a Document (HTML stripped to text).
- The FINE splitter feeds embeddings: focused ~350-token chunks retrieve precisely.
- The COARSE splitter feeds extraction: multi-paragraph context so relationships that span
  paragraphs are visible to the LLM, and the per-call prompt overhead is paid fewer times.
- Source metadata (``source_url``) is copied onto every chunk by both splitters and ends up
  stamped on every graph edge as provenance.

Usage:
    python kg_url_ingestion.py <documentation-url>

Run it, then use ``kg_question_answering.py`` to query both stores.

Requirements:
  - OPENAI_API_KEY (embeddings + entity extraction).
  - A running Neo4j. Locally:
        docker run -p 7687:7687 -p 7474:7474 -e NEO4J_AUTH=neo4j/password neo4j:5
    then set NEO4J_URI=bolt://localhost:7687, NEO4J_USERNAME=neo4j, NEO4J_PASSWORD=password.
  - Qdrant runs locally on disk (no server needed) under ``QDRANT_PATH`` below.
"""

import sys
from io import BytesIO

import requests
from qdrant_client import QdrantClient
from tqdm import tqdm

from dynamiq import Workflow
from dynamiq.callbacks import BaseCallbackHandler
from dynamiq.connections import Neo4j as Neo4jConnection
from dynamiq.connections import OpenAI as OpenAIConnection
from dynamiq.flows import Flow
from dynamiq.nodes.converters import HTMLConverter
from dynamiq.nodes.embedders import OpenAIDocumentEmbedder
from dynamiq.nodes.extractors import KnowledgeGraphWriter, Ontology
from dynamiq.nodes.llms.openai import OpenAI
from dynamiq.nodes.node import InputTransformer, NodeDependency
from dynamiq.nodes.splitters.document import DocumentSplitter
from dynamiq.nodes.writers import QdrantDocumentWriter
from dynamiq.runnables import RunnableConfig, RunnableStatus
from dynamiq.storages.vector.qdrant.qdrant import QdrantVectorStore
from dynamiq.utils.logger import logger

DEFAULT_URL = "https://en.wikipedia.org/wiki/Knowledge_graph"

# Same local stores as kg_ingestion.py so kg_question_answering.py can query the result.
QDRANT_PATH = "./.qdrant_kg_demo2"
INDEX_NAME = "kg_demo"
EMBEDDING_MODEL = "text-embedding-3-small"  # 1536 dims, matches Qdrant store default

# Ontology tuned for technical documentation. No triples: any relationship (of an allowed type)
# between allowed entity types is permitted — docs are too varied for a strict pattern list.
# Adjust the types to the documentation you ingest. An ontology is required (no free-form mode).
ONTOLOGY = Ontology(
    entity_types=["Product", "Component", "Feature", "Concept", "Technology", "Person", "Organization"],
    relationship_types=["HAS_FEATURE", "PART_OF", "USES", "INTEGRATES_WITH", "DEVELOPED_BY", "RELATED_TO"],
)

KG_LLM_ID = "kg_extraction_llm"


class IngestionProgress(BaseCallbackHandler):
    """tqdm progress bar over the per-chunk extraction LLM calls.

    The KnowledgeGraphWriter runs its embedded LLM once per chunk, and every nested node run
    fires the node callbacks — so the bar's total comes from the kg_splitter's output size, and
    each finished extraction LLM call advances it by one.
    """

    def __init__(self, llm_node_id: str, splitter_node_id: str):
        self.llm_node_id = llm_node_id
        self.splitter_node_id = splitter_node_id
        self.bar: tqdm | None = None

    def on_node_end(self, serialized, output_data, **kwargs):
        node_id = serialized.get("id")
        if node_id == self.splitter_node_id:
            self.bar = tqdm(total=len(output_data.get("documents", [])), desc="KG extraction", unit="chunk")
        elif node_id == self.llm_node_id and self.bar is not None:
            self.bar.update(1)

    def on_workflow_end(self, serialized, output_data, **kwargs):
        if self.bar is not None:
            self.bar.close()

    on_workflow_error = on_workflow_end


def build_workflow() -> Workflow:
    openai_connection = OpenAIConnection()

    # See kg_ingestion.py for why the on-disk client + force_disable_check_same_thread are needed.
    vector_store = QdrantVectorStore(
        client=QdrantClient(path=QDRANT_PATH, force_disable_check_same_thread=True),
        index_name=INDEX_NAME,
        create_if_not_exist=True,
        dimension=1536,
    )

    # ---- Stage 1: HTML -> one Document (text) ----
    html_converter = HTMLConverter(
        id="html_converter",
        input_transformer=InputTransformer(selector={"files": "$.files", "metadata": "$.metadata"}),
    )

    # ---- Branch A: fine chunks -> embeddings -> Qdrant (precision-sized for retrieval) ----
    vector_splitter = DocumentSplitter(
        id="vector_splitter",
        split_by="sentence",
        split_length=12,  # ~a focused paragraph per chunk, ideal embedding size
        split_overlap=1,  # one-sentence overlap so facts straddling a boundary survive
        depends=[NodeDependency(html_converter)],
        input_transformer=InputTransformer(selector={"documents": f"$.{html_converter.id}.output.documents"}),
    )
    document_embedder = OpenAIDocumentEmbedder(
        id="document_embedder",
        connection=openai_connection,
        model=EMBEDDING_MODEL,
        depends=[NodeDependency(vector_splitter)],
        input_transformer=InputTransformer(selector={"documents": f"$.{vector_splitter.id}.output.documents"}),
    )
    vector_writer = QdrantDocumentWriter(
        id="vector_writer",
        vector_store=vector_store,
        depends=[NodeDependency(document_embedder)],
        input_transformer=InputTransformer(selector={"documents": f"$.{document_embedder.id}.output.documents"}),
    )

    # ---- Branch B: coarse chunks -> entity/relationship extraction -> Neo4j ----
    # Larger windows so relationships spanning paragraphs are visible to a single LLM call,
    # and the fixed prompt overhead (template + ontology guidance) is paid fewer times.
    kg_splitter = DocumentSplitter(
        id="kg_splitter",
        split_by="sentence",
        split_length=50,  # multi-paragraph context per extraction call
        split_overlap=3,  # a few sentences of overlap so boundary-spanning relations survive
        depends=[NodeDependency(html_converter)],
        input_transformer=InputTransformer(selector={"documents": f"$.{html_converter.id}.output.documents"}),
    )
    knowledge_graph = KnowledgeGraphWriter(
        id="knowledge_graph",
        llm=OpenAI(
            id=KG_LLM_ID, connection=openai_connection, model="gpt-4o-mini", temperature=0.0, max_tokens=4000
        ),
        connection=Neo4jConnection(),
        ontology=ONTOLOGY,
        depends=[NodeDependency(kg_splitter)],
        input_transformer=InputTransformer(selector={"documents": f"$.{kg_splitter.id}.output.documents"}),
    )

    return Workflow(
        flow=Flow(
            nodes=[html_converter, vector_splitter, document_embedder, vector_writer, kg_splitter, knowledge_graph]
        )
    )


def fetch_page(url: str) -> BytesIO:
    """Download the page and wrap it as a named file-like object for HTMLConverter."""
    response = requests.get(url, timeout=30, headers={"User-Agent": "dynamiq-kg-ingestion-example"})
    response.raise_for_status()
    page = BytesIO(response.content)
    page.name = "page.html"
    return page


if __name__ == "__main__":
    url = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_URL
    logger.info(f"Ingesting documentation from: {url}")

    workflow = build_workflow()
    try:
        result = workflow.run(
            input_data={
                "files": [fetch_page(url)],
                # Stamped onto the converted Document, copied onto every chunk by the splitter,
                # and finally onto every graph EDGE by the extractor (provenance + ACL slot).
                # allowed_principals is what GraphRetriever filters on at query time — anyone whose
                # principals intersect this list can see these facts. Vary it per source to scope access.
                "metadata": {"source_url": url, "allowed_principals": ["group:public"]},
            },
            config=RunnableConfig(
                request_timeout=600,
                callbacks=[IngestionProgress(llm_node_id=KG_LLM_ID, splitter_node_id="kg_splitter")],
            ),
        )

        if result.status != RunnableStatus.SUCCESS:
            raise RuntimeError(f"Ingestion failed: {result.status} / {result.output}")

        vector_chunks = result.output["vector_splitter"]["output"].get("documents", [])
        kg_chunks = result.output["kg_splitter"]["output"].get("documents", [])
        vector_out = result.output["vector_writer"]["output"]
        graph_out = result.output["knowledge_graph"]["output"]
        logger.info(f"Splitters: {len(vector_chunks)} vector chunks, {len(kg_chunks)} extraction chunks.")
        logger.info(f"Vector store: upserted {vector_out.get('upserted_count')} chunks to Qdrant ('{INDEX_NAME}').")
        logger.info(
            f"Knowledge graph: created {graph_out.get('nodes_created')} nodes and "
            f"{graph_out.get('relationships_created')} relationships in Neo4j."
        )
        logger.info("--- Ingestion complete. Run kg_question_answering.py next. ---")
    finally:
        # Close stores so Neo4j's driver is released and on-disk Qdrant flushes (see kg_ingestion.py).
        for node in workflow.flow.nodes:
            graph_store = getattr(node, "_graph_store", None)
            if graph_store is not None:
                graph_store.close()
            vector_store = getattr(node, "vector_store", None)
            if vector_store is not None and getattr(vector_store, "_client", None) is not None:
                vector_store._client.close()
