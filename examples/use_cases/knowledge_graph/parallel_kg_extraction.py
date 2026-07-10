"""Parallel KG extraction → single writer.

Extraction (the LLM calls) is the slow, embarrassingly-parallel part of building a knowledge graph;
write-time entity resolution is NOT parallelizable (two writers racing would both mint fresh ids for the
same name and create duplicates). Splitting ``KnowledgeGraphEntityExtractor`` from ``KnowledgeGraphWriter`` lets us fan
extraction out and funnel it into one serial writer:

    documents ──shard──►  Map(KnowledgeGraphEntityExtractor, max_workers=N)   ──►  merge  ──►  KnowledgeGraphWriter
                          (N chunks extracted concurrently)        (one payload)   (single, serial)

The ``Map`` operator runs the extractor over a list of inputs on a thread pool — ideal here because LLM
extraction is I/O-bound (the GIL is released during the network wait). The merge concatenates the per-shard
payloads into one ``{nodes, relationships, documents}`` for the writer, which resolves duplicates ACROSS all
shards in a single pass (its per-call cache converges same-name entities from different shards).

Requires NEO4J_URI / NEO4J_USERNAME / NEO4J_PASSWORD and OPENAI_API_KEY (skips otherwise).
"""

import os

from dynamiq.connections import Neo4j as Neo4jConnection
from dynamiq.connections import OpenAI as OpenAIConnection
from dynamiq.nodes.knowledge_graph import KnowledgeGraphEntityExtractor, KnowledgeGraphWriter, Ontology
from dynamiq.nodes.llms.openai import OpenAI
from dynamiq.nodes.operators import Map
from dynamiq.runnables import RunnableConfig, RunnableStatus
from dynamiq.types import Document
from dynamiq.utils.logger import logger

ONTOLOGY = Ontology(
    entity_types=["Person", "Organization", "System"],
    relationship_types=["WORKS_AT", "USES"],
)

# Enough small documents that sharding + parallel extraction is worthwhile.
DOCS = [
    Document(content="Jane Doe is the CTO of Acme Capital."),
    Document(content="Acme Capital uses the TradingX platform."),
    Document(content="John Roe works at Globex Corporation."),
    Document(content="Globex Corporation uses the Helios system."),
    Document(content="Maria Lee is an engineer at Acme Capital."),
    Document(content="Acme Capital uses the Borealis analytics system."),
]

SHARDS = 3  # split DOCS into this many groups, extracted concurrently


def _shard(documents: list[Document], n: int) -> list[list[Document]]:
    """Round-robin documents into ``n`` shards (keeps shard sizes balanced)."""
    buckets: list[list[Document]] = [[] for _ in range(n)]
    for i, document in enumerate(documents):
        buckets[i % n].append(document)
    return [bucket for bucket in buckets if bucket]


def main() -> None:
    if not (os.getenv("OPENAI_API_KEY") and os.getenv("NEO4J_URI")):
        logger.info("OPENAI_API_KEY and NEO4J_URI required; skipping parallel KG extraction example.")
        return

    openai_connection = OpenAIConnection()

    # 1) One extractor, run in parallel over shards via Map(max_workers=SHARDS).
    extractor = KnowledgeGraphEntityExtractor(
        llm=OpenAI(connection=openai_connection, model="gpt-4o-mini", temperature=0.0, max_tokens=4000),
        ontology=ONTOLOGY,
    )
    parallel_extract = Map(node=extractor, max_workers=SHARDS)

    shards = _shard(DOCS, SHARDS)
    logger.info(f"Extracting {len(DOCS)} documents across {len(shards)} shards concurrently...")
    map_result = parallel_extract.run(
        input_data={"input": [{"documents": shard} for shard in shards]},
        config=RunnableConfig(request_timeout=120),
    )
    if map_result.status != RunnableStatus.SUCCESS:
        raise RuntimeError(f"Parallel extraction failed: {map_result.output}")

    # 2) Merge the per-shard payloads into one. (In a Flow this is a small merge node between the
    #    Map and the writer; resolution still happens once, across the whole merged payload.)
    per_shard = map_result.output["output"]  # list of KnowledgeGraphEntityExtractor outputs, one per shard
    nodes = [n for shard_out in per_shard for n in shard_out["nodes"]]
    relationships = [r for shard_out in per_shard for r in shard_out["relationships"]]
    documents = [d for shard_out in per_shard for d in shard_out["documents"]]
    logger.info(f"Extracted {len(nodes)} nodes / {len(relationships)} relationships; writing to Neo4j...")

    # 3) Single writer: resolves duplicates across ALL shards and upserts. Never parallelize this.
    writer = KnowledgeGraphWriter(connection=Neo4jConnection())
    writer.init_components()
    try:
        result = writer.execute(
            KnowledgeGraphWriter.input_schema(nodes=nodes, relationships=relationships, documents=documents)
        )
        logger.info(
            f"Wrote {result['nodes_created']} new node(s) and {result['relationships_created']} new "
            f"relationship(s) to Neo4j."
        )
    finally:
        writer._graph_store.close()


if __name__ == "__main__":
    main()
