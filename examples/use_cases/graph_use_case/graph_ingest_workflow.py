import json
import re
from pathlib import Path

from dynamiq import Workflow
from dynamiq.connections import Neo4j
from dynamiq.flows import Flow
from dynamiq.nodes.tools import CypherExecutor
from dynamiq.utils.logger import logger

DATA_PATH = Path(__file__).parent / "data" / "graph_inputs.jsonl"

ENTITY_LABELS = ["Person", "Company", "Product", "Event", "Location"]
RELATIONSHIP_TYPES = [
    "WORKS_AT",
    "PARTNERED_WITH",
    "PARTNERS_WITH",
    "BUILDS",
    "USES",
    "ACQUIRED",
    "LAUNCHED",
    "LOCATED_IN",
    "SPOKE_AT",
    "INVESTED_IN",
    "REBRANDED_TO",
    "INTEGRATES_WITH",
    "HOSTED_BY",
]
RELATION_PATTERN = re.compile(rf"(?P<left>.+?)\s+(?P<rel>{'|'.join(RELATIONSHIP_TYPES)})\s+(?P<right>.+)")

PRODUCT_NAMES = {
    "Dynamiq Platform",
    "GraphStream",
    "GraphStream GA",
    "GraphGuard",
    "GraphShield",
    "GraphAssist",
    "AgentFlow",
    "AppPres",
    "VectorBridge",
    "Databricks",
    "Snowflake",
    "BigQuery",
    "Redshift",
    "Airflow",
    "dbt",
    "Kafka",
    "Neo4j",
    "Delta Lake",
    "OpenSearch",
}
EVENT_NAMES = {
    "Graph Summit 2025",
    "DynamiqCon 2025",
    "Retail Graph Forum 2025",
}
LOCATION_NAMES = {
    "Berlin",
    "Singapore",
    "New York",
    "Amsterdam",
    "Stockholm",
    "Toronto",
    "Chicago",
    "San Francisco",
    "London",
    "Warsaw",
}


def _build_label_block() -> str:
    return "\n".join(
        f"FOREACH (_ IN CASE WHEN ent.label = '{label}' THEN [1] ELSE [] END | SET e:{label})"
        for label in ENTITY_LABELS
    )


def _build_relationship_block() -> str:
    return "\n".join(
        "FOREACH (_ IN CASE WHEN rel.type = '{rel_type}' THEN [1] ELSE [] END | "
        "MERGE (from)-[:{rel_type} {{source_id: $id}}]->(to))".format(rel_type=rel_type)
        for rel_type in RELATIONSHIP_TYPES
    )


CYPHER_QUERIES = [
    "MERGE (d:Document {id: $id}) " "SET d.source = $source, d.text = $text, d.timestamp = $timestamp",
    f"""
    MATCH (d:Document {{id: $id}})
    UNWIND $entities AS ent
    MERGE (e:Entity {{id: ent.id}})
    SET e.name = ent.name,
        e.type = ent.label
    {_build_label_block()}
    MERGE (d)-[:MENTIONS]->(e)
    """,
    f"""
    UNWIND $relations AS rel
    MATCH (from:Entity {{id: rel.from}})
    MATCH (to:Entity {{id: rel.to}})
    {_build_relationship_block()}
    """,
]


def _slugify(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    return value.strip("-")


def _normalize_name(value: str) -> str:
    value = value.strip()
    value = value.strip(" ,.;:")
    value = re.sub(r"\s+", " ", value)
    if value.lower().startswith("the "):
        value = value[4:]
    return value


def _guess_label(name: str, rel: str | None, position: str | None) -> str:
    if name in LOCATION_NAMES:
        return "Location"
    if name in EVENT_NAMES:
        return "Event"
    if name in PRODUCT_NAMES:
        return "Product"
    if "platform" in name.lower():
        return "Product"
    if rel == "WORKS_AT" and position == "left":
        return "Person"
    if rel == "SPOKE_AT" and position == "left":
        return "Person"
    if rel == "SPOKE_AT" and position == "right":
        return "Event"
    return "Company"


def _entity_id(name: str, label: str) -> str:
    return f"{label.lower()}:{_slugify(name)}"


def _ensure_entity(entities: dict[str, dict], name: str, rel: str | None, position: str | None) -> dict:
    normalized = _normalize_name(name)
    label = _guess_label(normalized, rel, position)
    entity_id = _entity_id(normalized, label)
    entity = entities.get(entity_id)
    if not entity:
        entity = {"id": entity_id, "label": label, "name": normalized}
        entities[entity_id] = entity
    return entity


def _trim_after(value: str, tokens: list[str]) -> str:
    lowered = value.lower()
    for token in tokens:
        idx = lowered.find(token)
        if idx > 0:
            return value[:idx]
    return value


def _split_trailing_location(value: str) -> tuple[str, str | None]:
    match = re.search(r"\s+in\s+([A-Z][A-Za-z\s]+)$", value)
    if not match:
        return value, None
    location = _normalize_name(match.group(1))
    return value[: match.start()].strip(), location


def _extract_rebrand(sentence: str, entities: dict[str, dict], relations: list[dict]) -> dict | None:
    match = re.search(r"(.+?)\s+(?:was\s+)?rebranded\s+as\s+(.+?)(?:\s+by\s+(.+))?$", sentence)
    if match:
        left = _normalize_name(match.group(1))
        right = _normalize_name(match.group(2))
        actor = _normalize_name(match.group(3)) if match.group(3) else None
        left_entity = _ensure_entity(entities, left, "REBRANDED_TO", "left")
        right_entity = _ensure_entity(entities, right, "REBRANDED_TO", "right")
        relations.append({"from": left_entity["id"], "type": "REBRANDED_TO", "to": right_entity["id"]})
        if actor:
            _ensure_entity(entities, actor, None, None)
        return left_entity
    match = re.search(r"(.+?)\s+is\s+now\s+(.+)$", sentence)
    if match:
        left = _normalize_name(match.group(1))
        right = _normalize_name(match.group(2))
        left_entity = _ensure_entity(entities, left, "REBRANDED_TO", "left")
        right_entity = _ensure_entity(entities, right, "REBRANDED_TO", "right")
        relations.append({"from": left_entity["id"], "type": "REBRANDED_TO", "to": right_entity["id"]})
        return left_entity
    return None


def _extract_entities_relations(text: str) -> tuple[list[dict], list[dict]]:
    entities: dict[str, dict] = {}
    relations: list[dict] = []
    last_subject: dict | None = None

    sentences = re.split(r"[.!?;]\s+", text)
    for raw_sentence in sentences:
        sentence = raw_sentence.strip()
        if not sentence:
            continue

        rebrand_subject = _extract_rebrand(sentence, entities, relations)
        if rebrand_subject:
            last_subject = rebrand_subject

        for match in RELATION_PATTERN.finditer(sentence):
            left = _normalize_name(match.group("left"))
            rel = match.group("rel")
            if rel == "PARTNERED_WITH":
                rel = "PARTNERS_WITH"
            right = _normalize_name(match.group("right"))

            if left.lower().startswith("it"):
                if not last_subject:
                    continue
                left_entity = last_subject
            else:
                left_entity = _ensure_entity(entities, left, rel, "left")

            right, trailing_location = _split_trailing_location(right)
            right = _trim_after(
                right,
                [
                    " about ",
                    " for ",
                    " to ",
                    " with ",
                    " and ",
                    " in ",
                    " expanded ",
                    " renewed ",
                    " opened ",
                    " signed ",
                    " launched ",
                    " acquired ",
                    " joined ",
                    " is ",
                    " was ",
                    ",",
                    ";",
                ],
            )
            right_entity = _ensure_entity(entities, right, rel, "right")
            relations.append({"from": left_entity["id"], "type": rel, "to": right_entity["id"]})

            if trailing_location:
                location_entity = _ensure_entity(entities, trailing_location, "LOCATED_IN", "right")
                relations.append({"from": right_entity["id"], "type": "LOCATED_IN", "to": location_entity["id"]})
            last_subject = left_entity

    return list(entities.values()), relations


def load_records(path: Path = DATA_PATH) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Missing data file: {path}")

    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def build_ingest_executor() -> CypherExecutor:
    connection = Neo4j()
    return CypherExecutor(connection=connection, name="cypher_executor")


def run_ingest() -> list[dict]:
    executor = build_ingest_executor()
    workflow = Workflow(flow=Flow(nodes=[executor]))

    outputs = []
    for record in load_records():
        logger.info("=== Ingesting %s ===", record.get("id"))
        entities, relations = _extract_entities_relations(record.get("text", ""))
        parameters = {
            "id": record.get("id"),
            "source": record.get("source"),
            "text": record.get("text"),
            "timestamp": record.get("timestamp"),
            "entities": entities,
            "relations": relations,
        }
        result = workflow.run(
            input_data={
                "mode": "execute",
                "query": CYPHER_QUERIES,
                "parameters": parameters,
                "routing": "w",
                "writes_allowed": True,
            }
        )
        executor_output = result.output.get(executor.id, {}).get("output") or {}
        outputs.append(
            {
                "id": record.get("id"),
                "output": executor_output.get("content"),
            }
        )
    if executor.client and hasattr(executor.client, "close"):
        executor.client.close()
    return outputs


if __name__ == "__main__":
    results = run_ingest()
    for result in results:
        print(f"[{result['id']}] {result['output']}")
