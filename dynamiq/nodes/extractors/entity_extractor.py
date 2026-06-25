import json
import re
import uuid
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, Field

from dynamiq.connections.managers import ConnectionManager
from dynamiq.nodes.node import Node, NodeDependency, NodeGroup, ensure_config
from dynamiq.prompts import prompts
from dynamiq.runnables import RunnableConfig, RunnableStatus
from dynamiq.types import Document
from dynamiq.types.cancellation import check_cancellation
from dynamiq.utils.json_parser import parse_llm_json_output
from dynamiq.utils.logger import logger

# Labels / relationship types / property keys must match this pattern — the safe identifier
# subset shared by openCypher backends (Neo4j, Apache AGE, Neptune); stores reject anything else.
_IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

# An ontology-declared attribute is written as (entity)-[:HAS_ATTRIBUTE {key}]->(:AttributeValue {value})
# rather than as a node property, so its access can be scoped independently of the (shared) entity node.
ATTRIBUTE_VALUE_LABEL = "AttributeValue"
HAS_ATTRIBUTE_TYPE = "HAS_ATTRIBUTE"

# Shared label put on every entity node (in addition to its type label) so a single full-text index can
# cover entity names across all types — the index-backed entry point GraphRetriever uses instead of a scan.
ENTITY_LABEL = "Entity"
ENTITY_NAME_FULLTEXT_INDEX = "entity_name"
# Range index on the entity id (Neo4j) so GraphRetriever can seek when seeding traversal by resolved id.
ENTITY_ID_INDEX = "entity_id"

# Metadata key under which the resolved, unique entity ids a chunk mentions are attached to that chunk —
# done by KnowledgeGraphWriter (which alone has the post-resolution durable ids). Lets a hybrid retriever
# seed graph traversal by id (unique, variant-proof), not by ambiguous entity name. ``kg_`` prefixed to
# avoid colliding with a caller's own document metadata.
KG_ENTITY_IDS_KEY = "kg_entity_ids"


def build_attribute_value_id(owner_id: str, attr_key: str, document_id: str | None) -> str:
    """Stable AttributeValue node id: "{owner}::{key}::{doc}" (doc omitted when absent).

    Single source of truth for the format so KnowledgeGraphWriter can rebuild it from the resolved
    owner id without parsing the composite string back apart.
    """
    value_id = f"{owner_id}::{attr_key}"
    return f"{value_id}::{document_id}" if document_id is not None else value_id


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
        attributes: Attributes to extract per entity type, e.g. ``{"Person": ["title", "salary"]}``.
            Each extracted attribute becomes a separate ``(entity)-[:HAS_ATTRIBUTE {key}]->(:AttributeValue)``
            relationship rather than a node property — so an attribute that can be sensitive (salary, email)
            carries its own access scope on the edge and is filtered independently of the (shared) entity node.
        entity_descriptions: Optional ``{type: description}`` explaining what each entity type means, e.g.
            ``{"Person": "an individual human"}``. Injected into the extraction prompt so the model applies
            the type as intended; types without a description are listed bare.
        relationship_descriptions: Optional ``{type: description}`` explaining what each relationship type
            means, e.g. ``{"WORKS_AT": "employment of a person by an organization"}``. Prompt-only, same as
            ``entity_descriptions``.
    """

    entity_types: list[str]
    relationship_types: list[str]
    triples: list[Triple] = Field(default_factory=list)
    attributes: dict[str, list[str]] = Field(default_factory=dict)
    entity_descriptions: dict[str, str] = Field(default_factory=dict)
    relationship_descriptions: dict[str, str] = Field(default_factory=dict)


class GraphNode(BaseModel):
    """A provider-neutral ``BaseGraphStore.write_graph`` node payload.

    ``identity_key`` names the property used to MERGE the node (always ``id`` here); ``properties`` must
    contain that key. ``labels[0]`` is the entity type, ``labels[1]`` the shared ``Entity`` label.
    """

    labels: list[str]
    properties: dict[str, Any]
    identity_key: str = "id"


class GraphRelationship(BaseModel):
    """A provider-neutral ``BaseGraphStore.write_graph`` relationship payload.

    Both endpoints are matched on their ``id`` property (``start_identity_key`` / ``end_identity_key``);
    ``start_identity`` / ``end_identity`` are the id values to match. ``properties`` carries the edge's own
    data plus per-document ACL/provenance attached later by ``_apply_document_metadata``.
    """

    type: str
    start_label: str
    end_label: str
    start_identity: str
    end_identity: str
    properties: dict[str, Any] = Field(default_factory=dict)
    start_identity_key: str = "id"
    end_identity_key: str = "id"


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
"description": "<optional one-line description>", "properties": {"<optional extra key>": "<value>"}}
  ]
}

Rules:
- Every relationship's source_id and target_id MUST reference an id present in "entities".
- Keep ids stable and unique (reuse the same id when the same entity appears again).
- "type" should be a short identifier (letters, digits, underscores). Relationship types are usually UPPER_SNAKE_CASE.
- A relationship's "description" is optional: one line adding detail the type alone does not convey (role, \
date, magnitude). Omit it when it would only restate the type.
- Use "properties" only for genuinely useful attributes; otherwise return an empty object.
{{type_guidance}}
Text:
{{document_text}}
"""

JSON_RECOVERY_PROMPT = """Your previous response could not be parsed as JSON.

Previous response:
{{previous_response}}

Return ONLY the corrected response as a single valid JSON object with "entities" and "relationships" \
keys, exactly in the format originally requested. No markdown fences, no prose."""


def _build_extraction_response_format(
    entity_types: list[str] | None = None, relationship_types: list[str] | None = None
) -> dict[str, Any]:
    """Litellm ``response_format`` mirroring the JSON shape requested by DEFAULT_EXTRACTION_PROMPT.

    When ``entity_types`` / ``relationship_types`` are given, the corresponding ``type`` fields are
    emitted as enums so the model cannot produce off-ontology types in the first place. Deliberately
    non-strict: "properties" is an open-ended map, which strict structured-output modes
    (``additionalProperties: false`` everywhere) cannot express.
    """
    entity_type_schema: dict[str, Any] = {"type": "string"}
    if entity_types:
        entity_type_schema["enum"] = entity_types
    relationship_type_schema: dict[str, Any] = {"type": "string"}
    if relationship_types:
        relationship_type_schema["enum"] = relationship_types

    return {
        "type": "json_schema",
        "json_schema": {
            "name": "knowledge_graph_extraction",
            "schema": {
                "type": "object",
                "properties": {
                    "entities": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "id": {"type": "string"},
                                "type": entity_type_schema,
                                "name": {"type": "string"},
                                "properties": {"type": "object"},
                            },
                            "required": ["id", "type", "name"],
                        },
                    },
                    "relationships": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "source_id": {"type": "string"},
                                "target_id": {"type": "string"},
                                "type": relationship_type_schema,
                                "description": {"type": "string"},
                                "properties": {"type": "object"},
                            },
                            "required": ["source_id", "target_id", "type"],
                        },
                    },
                },
                "required": ["entities", "relationships"],
            },
        },
    }


class EntityExtractorInputSchema(BaseModel):
    documents: list[Document] = Field(..., description="Documents to extract entities and relationships from.")


class EntityExtractor(Node):
    """Extracts a knowledge graph (entities + relationships) from documents using an LLM.

    The output uses the provider-neutral graph payload format consumed by
    :meth:`BaseGraphStore.write_graph` (``{"nodes": [...], "relationships": [...]}``). Entity ids in
    the payload are per-extraction wiring, not durable identity — persist through
    :class:`~dynamiq.nodes.extractors.knowledge_graph.KnowledgeGraphWriter`, which assigns identity
    via name resolution before writing.

    Attributes:
        group (Literal[NodeGroup.EXTRACTORS]): Node group. Defaults to NodeGroup.EXTRACTORS.
        name (str): Node name. Defaults to "entity-extractor".
        llm (Node): The LLM node used to perform the extraction.
        prompt_template (str): Jinja prompt template. Supports ``document_text`` and ``type_guidance``.
        ontology (Ontology): Required schema, the source of truth for extraction. The LLM is told the
            allowed types/triples/attributes AND the extracted graph is hard-filtered so only conforming
            nodes/relationships are written. (There is no free-form mode: declare what you want extracted.)
        json_recovery_attempts (int): How many times to ask the LLM to repair unparseable output
            before skipping a document. Defaults to 1; set to 0 to disable recovery.
    """

    group: Literal[NodeGroup.EXTRACTORS] = NodeGroup.EXTRACTORS
    name: str = "entity-extractor"
    llm: Node
    prompt_template: str = DEFAULT_EXTRACTION_PROMPT
    ontology: Ontology
    json_recovery_attempts: int = Field(
        default=1,
        ge=0,
        description="How many times to ask the LLM to repair unparseable output before skipping a document.",
    )
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
            dict: ``{"nodes": [...], "relationships": [...], "documents": [...], "entities": [...],
            "raw_relationships": [...]}``. ``nodes``/``relationships`` are ready for
            ``BaseGraphStore.write_graph`` and ``documents`` passes the source chunks through, so the whole
            output feeds straight into ``KnowledgeGraphWriter``; ``entities``/``raw_relationships`` are the
            un-transformed LLM output, kept for debugging.
        """
        config = ensure_config(config)
        self.reset_run_state()
        self.run_on_node_execute_run(config.callbacks, **kwargs)

        nodes: list[dict] = []
        relationships: list[dict] = []
        entities_debug: list[dict] = []
        raw_relationships_debug: list[dict] = []

        # Process each document independently and stamp its metadata onto every element it produces. A
        # per-document LLM failure skips only that chunk (like an unparseable response) so one transient
        # error can't discard the whole batch -- but if EVERY document fails, that's systemic (e.g. bad
        # credentials), so we raise rather than silently report an empty graph as success.
        failed = 0
        documents: list[Document] = []
        for document in input_data.documents:
            check_cancellation(config)
            # A document id scopes EVERYTHING per-document: node wiring ids, attribute ids, and the
            # source_doc_id ACL merge discriminator. A missing id silently collapses all three -- id-less
            # docs alias each other's nodes and their edges merge, overwriting allowed_principals. Assign a
            # stable id up front so the whole pipeline (and the returned documents) keys off a real value.
            # Copy rather than mutate in place so callers holding the input list don't see ids change.
            if document.id is None:
                document = document.model_copy(update={"id": uuid.uuid4().hex})
            documents.append(document)
            try:
                extracted = self._extract_from_text(document.content, config, **kwargs)
            except ValueError as exc:
                logger.error(f"Node {self.name} - {self.id}: extraction failed for a document, skipping it: {exc}")
                failed += 1
                continue
            entities = extracted.get("entities", []) or []
            doc_relationships = extracted.get("relationships", []) or []
            entities_debug.extend(entities)
            raw_relationships_debug.extend(doc_relationships)

            doc_nodes, doc_rels = self._to_write_graph_payload(entities, doc_relationships, document=document)
            doc_nodes, doc_rels = self._enforce_ontology(doc_nodes, doc_rels)
            self._apply_document_metadata(doc_rels, document)
            nodes.extend(doc_nodes)
            relationships.extend(doc_rels)

        if failed and failed == len(input_data.documents):
            raise ValueError(
                f"EntityExtractor: all {failed} document(s) failed extraction; this is likely systemic "
                "(e.g. invalid credentials or model access), not a per-document issue."
            )

        logger.debug(
            f"Node {self.name} - {self.id}: extracted {len(nodes)} nodes and {len(relationships)} relationships"
        )

        return {
            "nodes": nodes,
            "relationships": relationships,
            "documents": documents,
            "entities": entities_debug,
            "raw_relationships": raw_relationships_debug,
        }

    def _apply_document_metadata(self, relationships: list[dict], document: Document) -> None:
        """Copy the document's metadata + provenance onto every RELATIONSHIP it produced.

        Entity nodes (and ``AttributeValue`` value-nodes) deliberately carry NO access metadata — they are
        pure identity. All ACL/provenance lives on the edges, so a node is visible exactly when the user can
        reach it through a visible edge. This keeps entity merge a no-op on the node (just attach edges) and
        removes the whole class of node-ACL union/leak bugs.

        Generic and ACL-agnostic — whatever keys are in ``document.metadata`` (``allowed_principals``,
        ``acl_workspace_id``, timestamps, custom fields) ride along, exactly like the splitter copies a
        document's metadata onto each chunk. A provenance pointer ``source_doc_ids=[document.id]`` is added
        too. Each edge's own keys (e.g. the attribute ``key``) take precedence over metadata keys.

        The document id is also stamped as a scalar ``source_doc_id`` and declared in ``identity_keys`` so
        the store includes it in the relationship MERGE: the SAME fact asserted by two documents stays two
        SEPARATE edges, each keeping its own ACL/provenance, instead of merging and overwriting (which would
        leak — a public document could overwrite a confidential document's ``allowed_principals``).
        """
        doc_props = self._flatten_metadata(dict(document.metadata or {}))
        identity_keys: list[str] = []
        if document.id is not None:
            doc_props["source_doc_ids"] = [str(document.id)]
            doc_props["source_doc_id"] = str(document.id)
            identity_keys = ["source_doc_id"]
        if not doc_props:
            return
        for relationship in relationships:
            relationship["properties"] = {**doc_props, **relationship["properties"]}
            if identity_keys:
                relationship["identity_keys"] = identity_keys

    @classmethod
    def _flatten_metadata(cls, metadata: dict[str, Any], prefix: str = "") -> dict[str, Any]:
        """Flatten a metadata dict into graph-storable top-level properties.

        Property-graph backends restrict properties to primitives or arrays of primitives (no nested
        maps), so nested dicts are flattened with ``_``-joined, sanitized keys and any non-primitive
        value is JSON-encoded to a string.
        """
        out: dict[str, Any] = {}
        for raw_key, value in metadata.items():
            key = cls._sanitize_property_key(f"{prefix}{raw_key}")
            if isinstance(value, dict):
                out.update(cls._flatten_metadata(value, prefix=f"{key}_"))
            elif cls._is_primitive_property_value(value):
                out[key] = value
            else:
                out[key] = json.dumps(value, default=str)
        return out

    @staticmethod
    def _is_primitive_property_value(value: Any) -> bool:
        """True if value is a graph-storable scalar or a (homogeneous) list of scalars."""
        if isinstance(value, (str, int, float, bool)) or value is None:
            return True
        if isinstance(value, (list, tuple)):
            return all(isinstance(v, (str, int, float, bool)) or v is None for v in value)
        return False

    @staticmethod
    def _sanitize_property_key(key: str) -> str:
        """Coerce an arbitrary metadata key into a valid graph property identifier (case preserved)."""
        cleaned = re.sub(r"[^A-Za-z0-9_]+", "_", str(key).strip())
        cleaned = re.sub(r"_+", "_", cleaned).strip("_")
        if not cleaned:
            return "field"
        if cleaned[0].isdigit():
            cleaned = f"_{cleaned}"
        return cleaned

    def _extract_from_text(self, text: str, config: RunnableConfig, **kwargs) -> dict[str, Any]:
        """Run the LLM on a single document's text and parse its JSON output.

        If the output cannot be parsed, the raw response is sent back to the LLM with a repair
        instruction (up to ``json_recovery_attempts`` times). Only when recovery also fails is the
        document skipped (empty extraction), so one bad response does not abort a whole batch.
        """
        content = self._run_llm(
            self.prompt_template,
            {"document_text": text, "type_guidance": self._build_type_guidance()},
            config,
            **kwargs,
        )
        parsed = self._parse_llm_json(content)

        for attempt in range(self.json_recovery_attempts):
            if parsed is not None:
                break
            logger.warning(
                f"Node {self.name} - {self.id}: LLM output is not valid JSON; asking the LLM to "
                f"repair it (attempt {attempt + 1}/{self.json_recovery_attempts})."
            )
            content = self._run_llm(JSON_RECOVERY_PROMPT, {"previous_response": content or ""}, config, **kwargs)
            parsed = self._parse_llm_json(content)

        if parsed is None:
            logger.warning(
                f"Node {self.name} - {self.id}: could not parse LLM output after recovery; "
                "skipping this document (empty extraction)."
            )
            return {"entities": [], "relationships": []}
        return parsed

    def _run_llm(
        self, prompt_template: str, input_data: dict[str, Any], config: RunnableConfig, **kwargs
    ) -> str | None:
        """Run the embedded LLM with a prompt template and return its raw text content."""
        prompt = prompts.Prompt(messages=[prompts.Message(role="user", content=prompt_template)])
        run_kwargs = kwargs | {"parent_run_id": kwargs.get("run_id")}
        run_kwargs.pop("run_depends", None)

        llm_result = self.llm.run(
            input_data=input_data,
            prompt=prompt,
            response_format=self._resolve_response_format(),
            config=config,
            run_depends=self._run_depends,
            **run_kwargs,
        )
        self._run_depends = [NodeDependency(node=self.llm).to_dict(for_tracing=True)]
        if llm_result.status != RunnableStatus.SUCCESS:
            logger.error(f"Node {self.name} - {self.id}: LLM execution failed: {llm_result.error.to_dict()}")
            raise ValueError("EntityExtractor LLM execution failed")
        return llm_result.output["content"]

    def _resolve_response_format(self) -> dict[str, Any]:
        """The litellm ``response_format`` sent to the LLM: the extraction-JSON schema with
        entity/relationship ``type`` constrained to an enum of the ontology's types."""
        return _build_extraction_response_format(self.ontology.entity_types, self.ontology.relationship_types)

    def _build_type_guidance(self) -> str:
        """Build prompt guidance: the model is told the full ontology (types + legal triples + attributes)."""
        o = self.ontology
        lines = [
            "You MUST conform to this ontology. Anything outside it will be discarded:",
            f"- Allowed entity types: {self._format_typed(o.entity_types, o.entity_descriptions)}.",
            f"- Allowed relationship types: {self._format_typed(o.relationship_types, o.relationship_descriptions)}.",
        ]
        if o.triples:
            lines.append("- Only these relationship patterns are legal (source -[REL]-> target):")
            for t in o.triples:
                lines.append(f"    ({t.source}) -[{t.relationship}]-> ({t.target})")
        if o.attributes:
            lines.append('- For each entity, also extract these attributes when stated, inside its "properties":')
            for etype, attrs in o.attributes.items():
                lines.append(f"    {etype}: {', '.join(attrs)}")
        return "\n".join(lines)

    @staticmethod
    def _format_typed(types: list[str], descriptions: dict[str, str]) -> str:
        """Render ``types`` as a comma list, annotating any that have a description: ``Person (a human)``."""
        return ", ".join(f"{t} ({descriptions[t]})" if descriptions.get(t) else t for t in types)

    def _enforce_ontology(self, nodes: list[dict], relationships: list[dict]) -> tuple[list[dict], list[dict]]:
        """Drop any node/relationship that is not allowed by ``self.ontology``.

        Membership is checked against the same normalized (UPPER_SNAKE) identifiers that
        ``_to_write_graph_payload`` produces, so comparisons match what gets written to the graph store.
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

        # 1) Keep entity nodes whose label is in the ontology. AttributeValue nodes are held aside, NOT
        #    committed yet: an attribute value carries its access scope on its HAS_ATTRIBUTE edge, so it
        #    may only survive if that edge survives (step 3). Committing it here would orphan it -- a
        #    sensitive value left as a disconnected node with no edge ACL -- when its owner is dropped.
        kept_nodes: list[dict] = []
        kept_ids: set[str] = set()
        attribute_nodes: dict[str, dict] = {}
        for node in nodes:
            label = node["labels"][0]
            if label == ATTRIBUTE_VALUE_LABEL:
                attribute_nodes[node["properties"]["id"]] = node
            elif label in allowed_entities:
                kept_nodes.append(node)
                kept_ids.add(node["properties"]["id"])
            else:
                logger.debug(f"EntityExtractor: dropping off-ontology entity label={label!r}")

        # An AttributeValue endpoint is a valid target only while its node exists; pair it with the surviving
        # entity ids so the endpoint check below can vet HAS_ATTRIBUTE edges without committing the values.
        valid_endpoint_ids = kept_ids | attribute_nodes.keys()

        # 2) Keep only relationships with a legal type, surviving endpoints, and (when triples
        #    are defined) a legal (source_label, type, target_label) pattern. HAS_ATTRIBUTE edges are
        #    system-generated (declared attributes) — kept as long as their endpoints survive.
        kept_rels: list[dict] = []
        for rel in relationships:
            rel_type, start_label, end_label = rel["type"], rel["start_label"], rel["end_label"]
            if rel["start_identity"] not in valid_endpoint_ids or rel["end_identity"] not in valid_endpoint_ids:
                reason = "endpoint dropped"
            elif rel_type == HAS_ATTRIBUTE_TYPE:
                kept_rels.append(rel)
                continue
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

        # 3) Commit only the AttributeValue nodes still reached by a surviving HAS_ATTRIBUTE edge. Any value
        #    whose owning entity was dropped lost its edge in step 2, so it is dropped here too -- never left
        #    as an orphan node holding a value with no ACL.
        referenced_attr_ids = {r["end_identity"] for r in kept_rels if r["type"] == HAS_ATTRIBUTE_TYPE}
        for attr_id in attribute_nodes.keys() - referenced_attr_ids:
            logger.debug(f"EntityExtractor: dropping orphaned AttributeValue id={attr_id!r} (owner removed)")
        kept_nodes.extend(attribute_nodes[attr_id] for attr_id in referenced_attr_ids)

        dropped_n, dropped_r = len(nodes) - len(kept_nodes), len(relationships) - len(kept_rels)
        if dropped_n or dropped_r:
            logger.info(
                f"EntityExtractor ontology enforcement: dropped {dropped_n} off-ontology node(s) "
                f"and {dropped_r} off-ontology relationship(s)."
            )
        return kept_nodes, kept_rels

    @staticmethod
    def _parse_llm_json(content: str | None) -> dict[str, Any] | None:
        """Parse the LLM response into a dict, tolerating markdown fences and surrounding prose.

        Returns ``None`` when no JSON object can be recovered, so the caller can trigger
        an LLM repair round-trip.
        """
        if not content:
            return None

        text = content.strip()
        # Strip ```json ... ``` / ``` ... ``` fences if present.
        fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, re.DOTALL)
        if fence:
            text = fence.group(1).strip()

        try:
            # Direct parse -> balanced {...}/[...] extraction -> comment/quote cleanup.
            parsed = parse_llm_json_output(text)
        except ValueError:
            return None
        return parsed if isinstance(parsed, dict) else None

    def _attributes_for_type(self, entity_type: str) -> list[str]:
        """Attribute names declared for ``entity_type`` in the ontology (empty if none declared)."""
        return self.ontology.attributes.get(entity_type, [])

    def _to_write_graph_payload(
        self, entities: list[dict], relationships: list[dict], document: Document | None = None
    ) -> tuple[list[dict], list[dict]]:
        """Convert LLM-friendly entities/relationships into the write_graph node/relationship shape.

        Entity nodes carry ONLY identity (``id`` + ``name``) — never the LLM's free-form ``properties``.
        Nodes are merged by name and hold no ACL, so any inline property would be readable by anyone who can
        reach the shared node; keeping nodes identity-only removes that leak. Ontology-declared attributes
        (``Ontology.attributes``) are instead promoted to a separate
        ``(entity)-[:HAS_ATTRIBUTE {key}]->(:AttributeValue {value})`` relationship so their visibility is
        scoped independently of the (shared) entity node; any other emitted property is dropped. Value-node
        ids are scoped per source document (``{wiring_id}::{key}::{document_id}``): the same attribute asserted
        by two documents yields two value nodes, each on its own edge carrying its own document's
        ACL/provenance metadata, while re-ingesting the same document updates the same value node.

        Entity ids are emitted doc-scoped too (``{llm_id}@{document_id}``): LLM ids are only unique within
        one response, so two documents using the same id for DIFFERENT concepts must not alias once the
        per-document payloads are pooled. The wiring id carries no identity — entity identity is assigned
        by name resolution in ``KnowledgeGraphWriter``.
        """
        doc_suffix = f"@{document.id}" if document is not None and document.id is not None else ""
        id_to_label: dict[str, str] = {}
        nodes: list[dict] = []
        attribute_relationships: list[dict] = []
        for entity in entities:
            entity_id = entity.get("id")
            entity_type = entity.get("type")
            if entity_id is None or not entity_type:
                continue
            label = self._sanitize_identifier(str(entity_type))
            entity_id = str(entity_id)
            id_to_label[entity_id] = label
            wiring_id = f"{entity_id}{doc_suffix}"

            emitted = dict(entity.get("properties") or {})
            # Declared ontology attributes are reified to ACL-scoped HAS_ATTRIBUTE edges below.
            declared = set(self._attributes_for_type(str(entity_type)))
            attribute_values = {key: emitted[key] for key in emitted if key in declared}

            # Node = identity only (id + name). Other emitted props are dropped: nodes are merged by name and
            # carry no ACL, so an inline prop would leak to anyone who can reach the shared node.
            properties: dict[str, Any] = {"id": wiring_id}
            if entity.get("name") is not None:
                properties["name"] = entity["name"]

            # Type label first (ontology enforcement + resolution key on labels[0]); shared ENTITY_LABEL second
            # so one full-text index over names spans every type.
            nodes.append(GraphNode(labels=[label, ENTITY_LABEL], properties=properties).model_dump())

            for attr_key, attr_value in attribute_values.items():
                if attr_value is None:
                    continue
                doc_id = document.id if document is not None and document.id is not None else None
                value_id = build_attribute_value_id(wiring_id, attr_key, doc_id)
                value_node = GraphNode(
                    labels=[ATTRIBUTE_VALUE_LABEL], properties={"id": value_id, "value": attr_value}
                ).model_dump()
                # Structured parts for the writer to rebuild the resolved id without parsing the
                # composite string. Sibling key (not in `properties`) so the store never persists it.
                value_node["attr_ref"] = {"owner": wiring_id, "key": attr_key, "doc": doc_id}
                nodes.append(value_node)
                attribute_relationships.append(
                    GraphRelationship(
                        type=HAS_ATTRIBUTE_TYPE,
                        start_label=label,
                        end_label=ATTRIBUTE_VALUE_LABEL,
                        start_identity=wiring_id,
                        end_identity=value_id,
                        properties={"key": attr_key},
                    ).model_dump()
                )

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
            # Edges carry per-document ACL, so the optional description rides safely on the edge.
            rel_props = dict(rel.get("properties") or {})
            if rel.get("description") is not None:
                rel_props["description"] = rel["description"]
            graph_relationships.append(
                GraphRelationship(
                    type=self._sanitize_identifier(str(rel_type)),
                    start_label=id_to_label[source_id],
                    end_label=id_to_label[target_id],
                    start_identity=f"{source_id}{doc_suffix}",
                    end_identity=f"{target_id}{doc_suffix}",
                    properties=rel_props,
                ).model_dump()
            )

        return nodes, graph_relationships + attribute_relationships

    @staticmethod
    def _sanitize_identifier(value: str) -> str:
        """Coerce an arbitrary string into a valid label/relationship-type identifier.

        Replaces invalid characters with ``_``, collapses repeats, uppercases, and ensures the
        result starts with a letter or underscore. Guarantees the output matches the identifier
        rules shared by openCypher backends so ``write_graph`` will not reject it.
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
