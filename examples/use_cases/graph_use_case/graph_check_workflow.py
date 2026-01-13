from dynamiq import Workflow
from dynamiq.connections import Neo4j
from dynamiq.flows import Flow
from dynamiq.nodes.tools import CypherExecutor
from dynamiq.utils.logger import logger

QUERIES = [
    "MATCH (n) RETURN count(n) AS node_count",
    "MATCH ()-[r]->() RETURN count(r) AS rel_count",
    "MATCH (d:Document) RETURN d.id AS id, d.source AS source LIMIT 5",
    "MATCH ()-[r]->() RETURN type(r) AS type, r.source_id AS source_id LIMIT 5",
]


def _get_first_value(results: list[dict], index: int, key: str):
    if index >= len(results):
        return None
    records = results[index].get("records") or []
    if not records:
        return None
    return records[0].get(key)


def run_check() -> None:
    executor = CypherExecutor(connection=Neo4j(), name="cypher_executor")
    workflow = Workflow(flow=Flow(nodes=[executor]))

    result = workflow.run(
        input_data={
            "mode": "execute",
            "query": QUERIES,
            "parameters": {},
            "routing": "r",
            "writes_allowed": False,
        }
    )
    output = result.output.get(executor.id, {}).get("output", {})
    results = output.get("results", [])

    node_count = _get_first_value(results, 0, "node_count")
    rel_count = _get_first_value(results, 1, "rel_count")

    logger.info("Graph summary: nodes=%s relationships=%s", node_count, rel_count)
    if not node_count:
        logger.warning("Graph appears empty. Run graph_ingest_workflow.py first.")
    else:
        doc_samples = results[2].get("records", []) if len(results) > 2 else []
        rel_samples = results[3].get("records", []) if len(results) > 3 else []
        if doc_samples:
            logger.info("Document samples: %s", doc_samples)
        if rel_samples:
            logger.info("Relationship samples: %s", rel_samples)

    if executor.client and hasattr(executor.client, "close"):
        executor.client.close()


if __name__ == "__main__":
    run_check()
