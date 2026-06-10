import json
import re
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, Field

from dynamiq.connections.managers import ConnectionManager
from dynamiq.nodes.node import Node, NodeDependency, NodeGroup, ensure_config
from dynamiq.prompts import prompts
from dynamiq.runnables import RunnableConfig, RunnableStatus
from dynamiq.types import Document
from dynamiq.types.cancellation import check_cancellation
from dynamiq.utils.logger import logger

# Neo4j labels / relationship types / property keys must match this pattern,
# otherwise Neo4jGraphStore.write_graph raises (see storages/graph/neo4j/neo4j.py).
_IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class Triple(BaseModel):
    """A legal relationship pattern: (source entity type) -[relationship]-> (target entity type)."""

    source: str
    relationship: str
    target: str


class Ontology(BaseModel):
    """A schema the extracted knowledge graph must conform to.

    Attributes:
        entity_types: Allowed node labels.
        relationship_types: Allowed relationship types.
        triples: Legal ``(source)-[REL]->(target)`` patterns. If empty, any relationship
            (of an allowed type) between two allowed entity types is permitted; otherwise
            only the listed patterns are kept.
    """

    entity_types: list[str]
    relationship_types: list[str]
    triples: list[Triple] = Field(default_factory=list)

DEFAULT_EXTRACTION_PROMPT = """You are a knowledge-graph extraction engine. Read the text below and extract \
the entities and the relationships between them.

Return ONLY a single JSON object (no markdown, no prose) with exactly this shape:
{
  "entities": [
    {"id": "<stable unique id, e.g. slug of the name>", "type": "<EntityType>", "name": "<display name>", \
"properties": {"<optional extra key>": "<value>"}}
  ],
  "relationships": [
    {"source_id": "<id of source entity>", "target_id": "<id of target entity>", "type": "<RELATION_TYPE>", \
"properties": {"<optional extra key>": "<value>"}}
  ]
}

Rules:
- Every relationship's source_id and target_id MUST reference an id present in "entities".
- Keep ids stable and unique (reuse the same id when the same entity appears again).
- "type" should be a short identifier (letters, digits, underscores). Relationship types are usually UPPER_SNAKE_CASE.
- Use "properties" only for genuinely useful attributes; otherwise return an empty object.
{{type_guidance}}
Text:
{{document_text}}
"""


class EntityExtractorInputSchema(BaseModel):
    documents: list[Document] = Field(..., description="Documents to extract entities and relationships from.")


class EntityExtractor(Node):
    """Extracts a knowledge graph (entities + relationships) from documents using an LLM.

    The output is shaped for :meth:`Neo4jGraphStore.write_graph` so it can be passed
    directly to a :class:`~dynamiq.nodes.writers.graph.Neo4jGraphWriter` node:
    ``{"nodes": [...], "relationships": [...]}``.

    Attributes:
        group (Literal[NodeGroup.EXTRACTORS]): Node group. Defaults to NodeGroup.EXTRACTORS.
        name (str): Node name. Defaults to "entity-extractor".
        llm (Node): The LLM node used to perform the extraction.
        prompt_template (str): Jinja prompt template. Supports ``document_text`` and ``type_guidance``.
        entity_types (list[str] | None): Optional whitelist hint of entity types to extract.
        relationship_types (list[str] | None): Optional whitelist hint of relationship types to extract.
        ontology (Ontology | None): Optional schema. When set, it is the source of truth: the LLM is
            told the allowed types/triples AND the extracted graph is hard-filtered so only conforming
            nodes/relationships are written. When ``None`` the extractor is free-form (falling back to
            the soft ``entity_types``/``relationship_types`` hints if those are provided).
        response_format (dict | None): Optional litellm ``response_format`` forwarded to the LLM. Leave
            unset for providers that do not support structured-output schemas (extraction then relies on
            the JSON-instructed prompt + robust parsing).
    """

    group: Literal[NodeGroup.EXTRACTORS] = NodeGroup.EXTRACTORS
    name: str = "entity-extractor"
    llm: Node
    prompt_template: str = DEFAULT_EXTRACTION_PROMPT
    entity_types: list[str] | None = None
    relationship_types: list[str] | None = None
    ontology: Ontology | None = None
    response_format: dict[str, Any] | None = None
    input_schema: ClassVar[type[EntityExtractorInputSchema]] = EntityExtractorInputSchema

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._run_depends = []

    def reset_run_state(self):
        """Reset the intermediate steps (run_depends) of the node."""
        self._run_depends = []

    @property
    def to_dict_exclude_params(self):
        return super().to_dict_exclude_params | {"llm": True}

    def to_dict(self, **kwargs) -> dict:
        data = super().to_dict(**kwargs)
        data["llm"] = self.llm.to_dict(**kwargs)
        return data

    def init_components(self, connection_manager: ConnectionManager | None = None):
        """Initialize the embedded LLM component."""
        connection_manager = connection_manager or ConnectionManager()
        super().init_components(connection_manager)
        if self.llm.is_postponed_component_init:
            self.llm.init_components(connection_manager)

    def execute(
        self, input_data: EntityExtractorInputSchema, config: RunnableConfig = None, **kwargs
    ) -> dict[str, Any]:
        """Extract entities/relationships from the input documents.

        Returns:
            dict: ``{"nodes": [...], "relationships": [...], "entities": [...], "raw_relationships": [...]}``.
            ``nodes``/``relationships`` are ready for ``Neo4jGraphStore.write_graph``; ``entities``/
            ``raw_relationships`` are the un-transformed LLM output, kept for debugging.
        """
        config = ensure_config(config)
        self.reset_run_state()
        self.run_on_node_execute_run(config.callbacks, **kwargs)

        entities: list[dict] = []
        raw_relationships: list[dict] = []

        for document in input_data.documents:
            check_cancellation(config)
            extracted = self._extract_from_text(document.content, config, **kwargs)
            entities.extend(extracted.get("entities", []) or [])
            raw_relationships.extend(extracted.get("relationships", []) or [])

        entities = self._dedupe_entities(entities)
        nodes, relationships = self._to_write_graph_payload(entities, raw_relationships)

        if self.ontology is not None:
            nodes, relationships = self._enforce_ontology(nodes, relationships)

        logger.debug(
            f"Node {self.name} - {self.id}: extracted {len(nodes)} nodes and {len(relationships)} relationships"
        )

        return {
            "nodes": nodes,
            "relationships": relationships,
            "entities": entities,
            "raw_relationships": raw_relationships,
        }

    def _extract_from_text(self, text: str, config: RunnableConfig, **kwargs) -> dict[str, Any]:
        """Run the LLM on a single document's text and parse its JSON output."""
        prompt = prompts.Prompt(messages=[prompts.Message(role="user", content=self.prompt_template)])
        run_kwargs = kwargs | {"parent_run_id": kwargs.get("run_id")}
        run_kwargs.pop("run_depends", None)

        llm_result = self.llm.run(
            input_data={"document_text": text, "type_guidance": self._build_type_guidance()},
            prompt=prompt,
            response_format=self.response_format,
            config=config,
            run_depends=self._run_depends,
            **run_kwargs,
        )
        self._run_depends = [NodeDependency(node=self.llm).to_dict(for_tracing=True)]
        if llm_result.status != RunnableStatus.SUCCESS:
            logger.error(f"Node {self.name} - {self.id}: LLM execution failed: {llm_result.error.to_dict()}")
            raise ValueError("EntityExtractor LLM execution failed")

        return self._parse_llm_json(llm_result.output["content"])

    def _build_type_guidance(self) -> str:
        """Build optional prompt guidance constraining entity/relationship types.

        With an ``ontology`` set, the model is told the full schema (types + legal triples);
        otherwise it falls back to the soft ``entity_types``/``relationship_types`` hints.
        """
        if self.ontology is not None:
            o = self.ontology
            lines = [
                "You MUST conform to this ontology. Anything outside it will be discarded:",
                f"- Allowed entity types: {', '.join(o.entity_types)}.",
                f"- Allowed relationship types: {', '.join(o.relationship_types)}.",
            ]
            if o.triples:
                lines.append("- Only these relationship patterns are legal (source -[REL]-> target):")
                for t in o.triples:
                    lines.append(f"    ({t.source}) -[{t.relationship}]-> ({t.target})")
            return "\n".join(lines)

        lines = []
        if self.entity_types:
            lines.append(f"- Only extract entities of these types: {', '.join(self.entity_types)}.")
        if self.relationship_types:
            lines.append(f"- Only extract relationships of these types: {', '.join(self.relationship_types)}.")
        return "\n".join(lines)

    def _enforce_ontology(
        self, nodes: list[dict], relationships: list[dict]
    ) -> tuple[list[dict], list[dict]]:
        """Drop any node/relationship that is not allowed by ``self.ontology``.

        Membership is checked against the same normalized (UPPER_SNAKE) identifiers that
        ``_to_write_graph_payload`` produces, so comparisons match what gets written to Neo4j.
        """
        o = self.ontology
        allowed_entities = {self._sanitize_identifier(e) for e in o.entity_types}
        allowed_rels = {self._sanitize_identifier(r) for r in o.relationship_types}
        allowed_triples = {
            (
                self._sanitize_identifier(t.source),
                self._sanitize_identifier(t.relationship),
                self._sanitize_identifier(t.target),
            )
            for t in o.triples
        }

        # 1) Keep only nodes whose label is in the ontology.
        kept_nodes: list[dict] = []
        kept_ids: set[str] = set()
        for node in nodes:
            label = node["labels"][0]
            if label in allowed_entities:
                kept_nodes.append(node)
                kept_ids.add(node["properties"]["id"])
            else:
                logger.debug(f"EntityExtractor: dropping off-ontology entity label={label!r}")

        # 2) Keep only relationships with a legal type, surviving endpoints, and (when triples
        #    are defined) a legal (source_label, type, target_label) pattern.
        kept_rels: list[dict] = []
        for rel in relationships:
            rel_type, start_label, end_label = rel["type"], rel["start_label"], rel["end_label"]
            if rel["start_identity"] not in kept_ids or rel["end_identity"] not in kept_ids:
                reason = "endpoint dropped"
            elif rel_type not in allowed_rels:
                reason = "type not in ontology"
            elif allowed_triples and (start_label, rel_type, end_label) not in allowed_triples:
                reason = "illegal triple"
            else:
                kept_rels.append(rel)
                continue
            logger.debug(
                f"EntityExtractor: dropping relationship ({start_label})-[{rel_type}]->({end_label}): {reason}"
            )

        dropped_n, dropped_r = len(nodes) - len(kept_nodes), len(relationships) - len(kept_rels)
        if dropped_n or dropped_r:
            logger.info(
                f"EntityExtractor ontology enforcement: dropped {dropped_n} off-ontology node(s) "
                f"and {dropped_r} off-ontology relationship(s)."
            )
        return kept_nodes, kept_rels

    @staticmethod
    def _parse_llm_json(content: str | None) -> dict[str, Any]:
        """Parse the LLM response into a dict, tolerating markdown fences and surrounding prose."""
        if not content:
            return {"entities": [], "relationships": []}

        text = content.strip()
        # Strip ```json ... ``` / ``` ... ``` fences if present.
        fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, re.DOTALL)
        if fence:
            text = fence.group(1).strip()

        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            # Fall back to the first balanced-looking {...} block.
            start, end = text.find("{"), text.rfind("}")
            if start == -1 or end == -1 or end <= start:
                logger.warning("EntityExtractor: could not locate JSON object in LLM output.")
                return {"entities": [], "relationships": []}
            try:
                parsed = json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                logger.warning("EntityExtractor: failed to parse JSON from LLM output.")
                return {"entities": [], "relationships": []}

        if not isinstance(parsed, dict):
            return {"entities": [], "relationships": []}
        return parsed

    @staticmethod
    def _dedupe_entities(entities: list[dict]) -> list[dict]:
        """Deduplicate entities by id, keeping the first occurrence."""
        seen: set[str] = set()
        deduped: list[dict] = []
        for entity in entities:
            entity_id = entity.get("id")
            if entity_id is None or str(entity_id) in seen:
                continue
            seen.add(str(entity_id))
            deduped.append(entity)
        return deduped

    @classmethod
    def _to_write_graph_payload(cls, entities: list[dict], relationships: list[dict]) -> tuple[list[dict], list[dict]]:
        """Convert LLM-friendly entities/relationships into the write_graph node/relationship shape."""
        id_to_label: dict[str, str] = {}
        nodes: list[dict] = []
        for entity in entities:
            entity_id = entity.get("id")
            entity_type = entity.get("type")
            if entity_id is None or not entity_type:
                continue
            label = cls._sanitize_identifier(str(entity_type))
            entity_id = str(entity_id)
            id_to_label[entity_id] = label

            properties = dict(entity.get("properties") or {})
            properties["id"] = entity_id
            if entity.get("name") is not None:
                properties["name"] = entity["name"]
            nodes.append({"labels": [label], "identity_key": "id", "properties": properties})

        graph_relationships: list[dict] = []
        for rel in relationships:
            source_id = rel.get("source_id")
            target_id = rel.get("target_id")
            rel_type = rel.get("type")
            if source_id is None or target_id is None or not rel_type:
                continue
            source_id, target_id = str(source_id), str(target_id)
            # Skip dangling relationships whose endpoints were not extracted as entities.
            if source_id not in id_to_label or target_id not in id_to_label:
                logger.debug(f"EntityExtractor: skipping relationship with unresolved endpoint: {rel}")
                continue
            graph_relationships.append(
                {
                    "type": cls._sanitize_identifier(str(rel_type)),
                    "start_label": id_to_label[source_id],
                    "end_label": id_to_label[target_id],
                    "start_identity_key": "id",
                    "end_identity_key": "id",
                    "start_identity": source_id,
                    "end_identity": target_id,
                    "properties": dict(rel.get("properties") or {}),
                }
            )

        return nodes, graph_relationships

    @staticmethod
    def _sanitize_identifier(value: str) -> str:
        """Coerce an arbitrary string into a valid Neo4j label/relationship-type identifier.

        Replaces invalid characters with ``_``, collapses repeats, uppercases, and ensures the
        result starts with a letter or underscore. Guarantees the output matches Neo4j's identifier
        rules so ``write_graph`` will not reject it.
        """
        cleaned = re.sub(r"[^A-Za-z0-9_]+", "_", value.strip())
        cleaned = re.sub(r"_+", "_", cleaned).strip("_")
        cleaned = cleaned.upper()
        if not cleaned:
            return "ENTITY"
        if cleaned[0].isdigit():
            cleaned = f"_{cleaned}"
        # Defensive: should always hold given the steps above.
        if not _IDENTIFIER_PATTERN.match(cleaned):
            return "ENTITY"
        return cleaned
