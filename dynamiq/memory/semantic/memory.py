from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from dynamiq.memory.semantic.context import ContextBlockBuilder
from dynamiq.memory.semantic.extraction import LLMOntologyExtractor, OntologyExtractor
from dynamiq.memory.semantic.invalidation import facts_conflict, predicate_supports_invalidation
from dynamiq.memory.semantic.retrieval import ContextRetrievalMode, lexical_score
from dynamiq.nodes.llms.base import BaseLLM
from dynamiq.ontology import Entity, Episode, FactStatus, OntologyEntityType, Provenance, TemporalFact
from dynamiq.ontology.resolver import EntityResolver, normalize_entity_label
from dynamiq.ontology.validators import validate_temporal_fact
from dynamiq.storages.graph.base import BaseGraphStore
from dynamiq.utils.logger import logger

EPISODE_LABEL = "OntologyEpisode"
ENTITY_LABEL = "OntologyEntity"
FACT_LABEL = "OntologyFact"
VALUE_LABEL = "OntologyValue"
PROVENANCE_LABEL = "OntologyProvenance"


class OntologyMemory(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    graph_store: BaseGraphStore
    query_limit: int = Field(default=100, gt=0)
    llm: BaseLLM | None = None
    extractor: OntologyExtractor | None = None
    resolver: EntityResolver = Field(default_factory=EntityResolver)
    context_builder: ContextBlockBuilder = Field(default_factory=ContextBlockBuilder)

    @model_validator(mode="after")
    def configure_extractor(self) -> OntologyMemory:
        if self.extractor is None and self.llm is not None:
            self.extractor = LLMOntologyExtractor(llm=self.llm)
        return self

    def ensure_schema(self) -> None:
        if type(self.graph_store).__name__ != "Neo4jGraphStore":
            return

        statements = [
            f"CREATE CONSTRAINT {EPISODE_LABEL.lower()}_id IF NOT EXISTS FOR (n:{EPISODE_LABEL}) REQUIRE n.id IS UNIQUE",  # noqa: E501
            f"CREATE CONSTRAINT {ENTITY_LABEL.lower()}_id IF NOT EXISTS FOR (n:{ENTITY_LABEL}) REQUIRE n.id IS UNIQUE",
            f"CREATE CONSTRAINT {FACT_LABEL.lower()}_id IF NOT EXISTS FOR (n:{FACT_LABEL}) REQUIRE n.id IS UNIQUE",
            (
                f"CREATE CONSTRAINT {PROVENANCE_LABEL.lower()}_id IF NOT EXISTS "
                f"FOR (n:{PROVENANCE_LABEL}) REQUIRE n.id IS UNIQUE"
            ),
            f"CREATE INDEX {FACT_LABEL.lower()}_predicate IF NOT EXISTS FOR (n:{FACT_LABEL}) ON (n.predicate)",
            f"CREATE INDEX {FACT_LABEL.lower()}_status IF NOT EXISTS FOR (n:{FACT_LABEL}) ON (n.status)",
        ]

        for statement in statements:
            self.graph_store.run_cypher(statement)

    def add_episode(self, episode: Episode | None = None, **kwargs: Any) -> Episode:
        episode = episode or Episode(**kwargs)
        properties = self._serialize_model(episode)
        self._run_named_query(
            "upsert_episode",
            f"MERGE (episode:{EPISODE_LABEL} {{id: $id}}) SET episode += $properties RETURN episode",
            {"id": episode.id, "properties": properties},
        )
        return episode

    def get_episodes(
        self,
        *,
        episode_id: str | None = None,
        user_id: str | None = None,
        session_id: str | None = None,
        limit: int | None = None,
    ) -> list[Episode]:
        records, _, _ = self._run_named_query(
            "list_episodes",
            (
                f"MATCH (episode:{EPISODE_LABEL}) "
                "WHERE ($episode_id IS NULL OR episode.id = $episode_id) "
                "AND ($user_id IS NULL OR episode.user_id = $user_id) "
                "AND ($session_id IS NULL OR episode.session_id = $session_id) "
                "RETURN episode ORDER BY episode.observed_at DESC LIMIT $limit"
            ),
            {
                "episode_id": episode_id,
                "user_id": user_id,
                "session_id": session_id,
                "limit": limit or self.query_limit,
            },
        )
        episodes: list[Episode] = []
        for record in records:
            payload = self._normalize_value(self._normalize_record(record).get("episode"))
            if payload:
                episodes.append(Episode(**payload))
        return episodes

    def upsert_entity(self, entity: Entity | None = None, **kwargs: Any) -> Entity:
        entity = entity or Entity(**kwargs)
        properties = self._serialize_model(entity)
        self._run_named_query(
            "upsert_entity",
            f"MERGE (entity:{ENTITY_LABEL} {{id: $id}}) SET entity += $properties RETURN entity",
            {"id": entity.id, "properties": properties},
        )
        return entity

    def link_episode_entity(self, *, episode_id: str, entity_id: str) -> None:
        self._run_named_query(
            "link_episode_entity",
            (
                f"MATCH (episode:{EPISODE_LABEL} {{id: $episode_id}}) "
                f"MATCH (entity:{ENTITY_LABEL} {{id: $entity_id}}) "
                "MERGE (episode)-[:MENTIONED_ENTITY]->(entity) "
                "RETURN entity"
            ),
            {"episode_id": episode_id, "entity_id": entity_id},
        )

    def get_episode_entities(
        self,
        *,
        episode_id: str | None = None,
        user_id: str | None = None,
        session_id: str | None = None,
        limit: int | None = None,
    ) -> list[Entity]:
        records, _, _ = self._run_named_query(
            "list_episode_entities",
            (
                f"MATCH (episode:{EPISODE_LABEL})-[:MENTIONED_ENTITY]->(entity:{ENTITY_LABEL}) "
                "WHERE ($episode_id IS NULL OR episode.id = $episode_id) "
                "AND ($user_id IS NULL OR episode.user_id = $user_id) "
                "AND ($session_id IS NULL OR episode.session_id = $session_id) "
                "RETURN entity, episode ORDER BY episode.observed_at DESC LIMIT $limit"
            ),
            {
                "episode_id": episode_id,
                "user_id": user_id,
                "session_id": session_id,
                "limit": limit or self.query_limit,
            },
        )
        entities: list[Entity] = []
        seen_entity_ids: set[str] = set()
        for record in records:
            payload = self._normalize_value(self._normalize_record(record).get("entity"))
            if not payload or payload["id"] in seen_entity_ids:
                continue
            entities.append(Entity(**payload))
            seen_entity_ids.add(payload["id"])
        return entities

    def get_entities(
        self,
        *,
        entity_id: str | None = None,
        entity_type: OntologyEntityType | str | None = None,
        limit: int | None = None,
    ) -> list[Entity]:
        type_value = entity_type.value if isinstance(entity_type, Enum) else entity_type
        records, _, _ = self._run_named_query(
            "list_entities",
            (
                f"MATCH (entity:{ENTITY_LABEL}) "
                "WHERE ($entity_id IS NULL OR entity.id = $entity_id) "
                "AND ($entity_type IS NULL OR entity.entity_type = $entity_type) "
                "RETURN entity ORDER BY entity.updated_at DESC LIMIT $limit"
            ),
            {
                "entity_id": entity_id,
                "entity_type": type_value,
                "limit": limit or self.query_limit,
            },
        )
        entities: list[Entity] = []
        for record in records:
            payload = self._normalize_value(self._normalize_record(record).get("entity"))
            if payload:
                entities.append(Entity(**payload))
        return entities

    def resolve_or_create_entity(
        self,
        *,
        label: str,
        entity_type: OntologyEntityType,
        aliases: list[str] | None = None,
        summary: str | None = None,
        confidence: float = 0.9,
    ) -> Entity:
        candidates = self.get_entities(entity_type=entity_type, limit=self.query_limit)
        resolved = self.resolver.resolve(label=label, candidates=candidates)
        if resolved:
            entity = resolved.entity
            updated_aliases = sorted({*entity.aliases, *(aliases or [])})
            if updated_aliases != entity.aliases or summary and summary != entity.summary:
                entity = entity.model_copy(
                    update={
                        "aliases": updated_aliases,
                        "summary": summary or entity.summary,
                        "confidence": max(entity.confidence, confidence),
                        "updated_at": datetime.now(entity.updated_at.tzinfo),
                    }
                )
                self.upsert_entity(entity)
            return entity

        entity = Entity(
            label=label,
            entity_type=entity_type,
            aliases=aliases or [],
            summary=summary,
            confidence=confidence,
        )
        return self.upsert_entity(entity)

    def add_fact(self, fact: TemporalFact | None = None, **kwargs: Any) -> TemporalFact:
        fact = validate_temporal_fact(fact or TemporalFact(**kwargs))
        properties = self._serialize_model(fact)
        self._run_named_query(
            "upsert_fact",
            (
                f"MATCH (subject:{ENTITY_LABEL} {{id: $subject_id}}) "
                f"MERGE (fact:{FACT_LABEL} {{id: $fact_id}}) "
                "SET fact += $properties "
                "MERGE (subject)-[:HAS_FACT]->(fact) "
                "RETURN fact"
            ),
            {"subject_id": fact.subject_id, "fact_id": fact.id, "properties": properties},
        )

        if fact.object_id:
            self._run_named_query(
                "link_fact_object_entity",
                (
                    f"MATCH (fact:{FACT_LABEL} {{id: $fact_id}}) "
                    f"MATCH (object:{ENTITY_LABEL} {{id: $object_id}}) "
                    "MERGE (fact)-[:FACT_OBJECT]->(object) "
                    "RETURN fact"
                ),
                {"fact_id": fact.id, "object_id": fact.object_id},
            )
        else:
            value_node_id = f"value:{fact.id}"
            self._run_named_query(
                "link_fact_object_value",
                (
                    f"MATCH (fact:{FACT_LABEL} {{id: $fact_id}}) "
                    f"MERGE (value:{VALUE_LABEL} {{id: $value_id}}) "
                    "SET value += $value_properties "
                    "MERGE (fact)-[:FACT_OBJECT_VALUE]->(value) "
                    "RETURN fact"
                ),
                {
                    "fact_id": fact.id,
                    "value_id": value_node_id,
                    "value_properties": {"id": value_node_id, "value": self._serialize_value(fact.object_value)},
                },
            )

        for episode_id in fact.episode_ids:
            self._run_named_query(
                "link_episode_fact",
                (
                    f"MATCH (episode:{EPISODE_LABEL} {{id: $episode_id}}) "
                    f"MATCH (fact:{FACT_LABEL} {{id: $fact_id}}) "
                    "MERGE (episode)-[:SUPPORTS_FACT]->(fact) "
                    "RETURN fact"
                ),
                {"episode_id": episode_id, "fact_id": fact.id},
            )

        self._invalidate_conflicting_facts(fact)
        return fact

    def add_provenance(self, provenance: Provenance | None = None, **kwargs: Any) -> Provenance:
        provenance = provenance or Provenance(**kwargs)
        properties = self._serialize_model(provenance)
        self._run_named_query(
            "upsert_provenance",
            f"MERGE (provenance:{PROVENANCE_LABEL} {{id: $id}}) SET provenance += $properties RETURN provenance",
            {"id": provenance.id, "properties": properties},
        )
        self._run_named_query(
            "link_provenance_episode",
            (
                f"MATCH (provenance:{PROVENANCE_LABEL} {{id: $id}}) "
                f"MATCH (episode:{EPISODE_LABEL} {{id: $episode_id}}) "
                "MERGE (provenance)-[:PROVENANCE_EPISODE]->(episode) "
                "RETURN provenance"
            ),
            {"id": provenance.id, "episode_id": provenance.episode_id},
        )
        self._run_named_query(
            "link_provenance_fact",
            (
                f"MATCH (provenance:{PROVENANCE_LABEL} {{id: $id}}) "
                f"MATCH (fact:{FACT_LABEL} {{id: $fact_id}}) "
                "MERGE (provenance)-[:PROVENANCE_FACT]->(fact) "
                "RETURN provenance"
            ),
            {"id": provenance.id, "fact_id": provenance.fact_id},
        )
        return provenance

    def get_provenance(self, *, fact_id: str, limit: int | None = None) -> list[Provenance]:
        records, _, _ = self._run_named_query(
            "list_provenance",
            (
                f"MATCH (provenance:{PROVENANCE_LABEL})-[:PROVENANCE_FACT]->(fact:{FACT_LABEL} {{id: $fact_id}}) "
                "RETURN provenance ORDER BY provenance.created_at DESC LIMIT $limit"
            ),
            {"fact_id": fact_id, "limit": limit or self.query_limit},
        )
        items: list[Provenance] = []
        for record in records:
            payload = self._normalize_value(self._normalize_record(record).get("provenance"))
            if payload:
                items.append(Provenance(**payload))
        return items

    def get_facts(
        self,
        *,
        fact_id: str | None = None,
        subject_id: str | None = None,
        predicate: str | None = None,
        user_id: str | None = None,
        session_id: str | None = None,
        valid_at: datetime | None = None,
        include_inactive: bool = False,
        limit: int | None = None,
    ) -> list[TemporalFact]:
        records, _, _ = self._run_named_query(
            "list_facts",
            (
                f"MATCH (fact:{FACT_LABEL}) "
                f"OPTIONAL MATCH (subject:{ENTITY_LABEL})-[:HAS_FACT]->(fact) "
                f"OPTIONAL MATCH (fact)-[:FACT_OBJECT]->(object:{ENTITY_LABEL}) "
                f"OPTIONAL MATCH (fact)-[:FACT_OBJECT_VALUE]->(value:{VALUE_LABEL}) "
                "WHERE ($fact_id IS NULL OR fact.id = $fact_id) "
                "AND ($subject_id IS NULL OR subject.id = $subject_id) "
                "AND ($predicate IS NULL OR fact.predicate = $predicate) "
                "AND ($user_id IS NULL OR fact.user_id = $user_id) "
                "AND ($session_id IS NULL OR fact.session_id = $session_id) "
                "AND ($include_inactive = true OR fact.status = $active_status) "
                "AND ("
                "$valid_at IS NULL OR (fact.valid_at <= $valid_at AND (fact.invalid_at IS NULL OR fact.invalid_at > $valid_at))"  # noqa: E501
                ") "
                "RETURN fact, subject, object, value "
                "ORDER BY fact.valid_at DESC, fact.created_at DESC "
                "LIMIT $limit"
            ),
            {
                "fact_id": fact_id,
                "subject_id": subject_id,
                "predicate": predicate,
                "user_id": user_id,
                "session_id": session_id,
                "valid_at": valid_at.isoformat() if valid_at else None,
                "include_inactive": include_inactive,
                "active_status": FactStatus.ACTIVE.value,
                "limit": limit or self.query_limit,
            },
        )

        facts: list[TemporalFact] = []
        for record in records:
            row = self._normalize_record(record)
            payload = self._normalize_value(row.get("fact"))
            if not payload:
                continue

            subject = self._normalize_value(row.get("subject"))
            object_entity = self._normalize_value(row.get("object"))
            value_node = self._normalize_value(row.get("value"))

            payload.setdefault("subject_id", subject.get("id") if subject else None)
            payload.setdefault("object_id", object_entity.get("id") if object_entity else None)
            payload.setdefault("object_value", value_node.get("value") if value_node else None)

            facts.append(TemporalFact(**payload))

        return facts

    def search_facts(
        self,
        *,
        query: str | None = None,
        user_id: str | None = None,
        session_id: str | None = None,
        predicate: str | None = None,
        include_inactive: bool = False,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        facts = self.get_facts(
            predicate=predicate,
            user_id=user_id,
            session_id=session_id,
            include_inactive=include_inactive,
            limit=limit or self.query_limit,
        )
        entities = {entity.id: entity for entity in self.get_entities(limit=self.query_limit)}

        for fact in facts:
            subject = entities.get(fact.subject_id)
            object_entity = entities.get(fact.object_id) if fact.object_id else None
            row = {
                **fact.model_dump(),
                "subject_label": subject.label if subject else fact.subject_id,
                "object_label": object_entity.label if object_entity else None,
            }
            row["score"] = lexical_score(
                query=query,
                haystacks=[
                    row.get("predicate", ""),
                    row.get("subject_label", "") or "",
                    row.get("object_label", "") or "",
                    str(row.get("object_value", "") or ""),
                ],
            )
            rows.append(row)

        if query:
            rows.sort(key=lambda item: (item["score"], item["valid_at"]), reverse=True)

        return rows[: (limit or self.query_limit)]

    def extract_and_commit(self, *, episode_id: str | None = None, episode: Episode | None = None) -> dict[str, Any]:
        if self.extractor is None:
            raise ValueError("Ontology extractor is not configured. Pass an LLM node or explicit extractor.")
        if episode is None:
            if not episode_id:
                raise ValueError("episode_id or episode is required.")
            episodes = self.get_episodes(episode_id=episode_id, limit=1)
            if not episodes:
                raise ValueError(f"Episode '{episode_id}' not found.")
            episode = episodes[0]

        extraction = self.extractor.extract(episode)
        entity_map: dict[str, Entity] = {}
        for candidate in extraction.entities:
            entity = self.resolve_or_create_entity(
                label=candidate.label,
                entity_type=candidate.entity_type,
                aliases=candidate.aliases,
                summary=candidate.summary,
                confidence=candidate.confidence,
            )
            entity_map[normalize_entity_label(candidate.label)] = entity
            self.link_episode_entity(episode_id=episode.id, entity_id=entity.id)

        created_facts: list[TemporalFact] = []
        created_provenance: list[Provenance] = []
        for candidate in extraction.facts:
            subject_entity = entity_map.get(
                normalize_entity_label(candidate.subject_label)
            ) or self.resolve_or_create_entity(
                label=candidate.subject_label,
                entity_type=candidate.subject_type,
                confidence=candidate.confidence,
            )
            object_entity = None
            object_value = candidate.object_value
            if candidate.object_label and candidate.object_type:
                object_entity = entity_map.get(
                    normalize_entity_label(candidate.object_label)
                ) or self.resolve_or_create_entity(
                    label=candidate.object_label,
                    entity_type=candidate.object_type,
                    confidence=candidate.confidence,
                )
                object_value = None

            fact = self.add_fact(
                TemporalFact(
                    subject_id=subject_entity.id,
                    predicate=candidate.predicate,
                    object_id=object_entity.id if object_entity else None,
                    object_value=object_value,
                    confidence=candidate.confidence,
                    episode_ids=[episode.id],
                    created_by_agent_id=episode.actor_id,
                    user_id=episode.user_id,
                    session_id=episode.session_id,
                    workflow_id=episode.workflow_id,
                    valid_at=episode.observed_at,
                )
            )
            created_facts.append(fact)
            created_provenance.append(
                self.add_provenance(
                    Provenance(
                        episode_id=episode.id,
                        fact_id=fact.id,
                        extracting_agent_id=episode.actor_id,
                        extraction_model=type(self.extractor).__name__,
                        validation_result="accepted",
                    )
                )
            )

        return {
            "episode": episode,
            "entities": list(entity_map.values()),
            "facts": created_facts,
            "provenance": created_provenance,
            "notes": extraction.notes,
        }

    def get_entity_context(self, *, entity_id: str | None = None, entity_label: str | None = None) -> dict[str, Any]:
        entity = None
        if entity_id:
            entities = self.get_entities(entity_id=entity_id, limit=1)
            entity = entities[0] if entities else None
        elif entity_label:
            normalized = normalize_entity_label(entity_label)
            for candidate in self.get_entities(limit=self.query_limit):
                if normalize_entity_label(candidate.label) == normalized:
                    entity = candidate
                    break
                if any(normalize_entity_label(alias) == normalized for alias in candidate.aliases):
                    entity = candidate
                    break

        if entity is None:
            raise ValueError("Entity not found.")

        facts = self.search_facts(limit=self.query_limit)
        related_facts = [
            fact for fact in facts if fact.get("subject_id") == entity.id or fact.get("object_id") == entity.id
        ]
        episode_ids = sorted({episode_id for fact in related_facts for episode_id in fact.get("episode_ids", [])})
        episodes: list[Episode] = []
        for episode_id in episode_ids:
            matched = self.get_episodes(episode_id=episode_id, limit=1)
            if matched:
                episodes.append(matched[0])
        return {"entity": entity, "facts": related_facts, "episodes": episodes}

    def audit_fact(self, *, fact_id: str) -> dict[str, Any]:
        facts = self.get_facts(fact_id=fact_id, include_inactive=True, limit=1)
        if not facts:
            raise ValueError(f"Fact '{fact_id}' not found.")
        fact = facts[0]
        subject = self.get_entities(entity_id=fact.subject_id, limit=1)
        object_entity = self.get_entities(entity_id=fact.object_id, limit=1) if fact.object_id else []
        episodes = [
            episode for episode_id in fact.episode_ids for episode in self.get_episodes(episode_id=episode_id, limit=1)
        ]
        provenance = self.get_provenance(fact_id=fact_id)
        return {
            "fact": fact,
            "subject": subject[0] if subject else None,
            "object": object_entity[0] if object_entity else None,
            "episodes": episodes,
            "provenance": provenance,
        }

    def get_context_block(
        self,
        *,
        query: str | None = None,
        user_id: str | None = None,
        session_id: str | None = None,
        mode: ContextRetrievalMode | str = ContextRetrievalMode.CURRENT,
        valid_at: datetime | None = None,
        limit: int | None = None,
    ) -> str:
        mode_value = mode.value if isinstance(mode, Enum) else mode
        include_inactive = mode_value == ContextRetrievalMode.AUDIT.value
        fact_rows = self.search_facts(
            query=query,
            user_id=user_id,
            session_id=session_id,
            include_inactive=include_inactive,
            limit=limit or self.query_limit,
        )
        if mode_value == ContextRetrievalMode.HISTORICAL.value and valid_at:
            fact_rows = [
                fact
                for fact in fact_rows
                if fact["valid_at"] <= valid_at and (fact["invalid_at"] is None or fact["invalid_at"] > valid_at)
            ]

        entities_by_id = {entity.id: entity for entity in self.get_entities(limit=self.query_limit)}
        relevant_entities = {
            entity_id: entity
            for fact in fact_rows
            for entity_id, entity in (
                (fact.get("subject_id"), entities_by_id.get(fact.get("subject_id"))),
                (fact.get("object_id"), entities_by_id.get(fact.get("object_id"))),
            )
            if entity_id and entity is not None
        }
        episodes = self.get_episodes(user_id=user_id, session_id=session_id, limit=limit or 10)
        if mode_value == ContextRetrievalMode.HISTORICAL.value and valid_at:
            episodes = [episode for episode in episodes if episode.observed_at <= valid_at]
        for episode in episodes:
            for entity in self.get_episode_entities(episode_id=episode.id, limit=limit or 10):
                relevant_entities.setdefault(entity.id, entity)
        return self.context_builder.build(
            query=query,
            entities=list(relevant_entities.values()),
            facts=fact_rows,
            episodes=episodes,
        )

    def _invalidate_conflicting_facts(self, fact: TemporalFact) -> None:
        if not predicate_supports_invalidation(fact.predicate):
            return
        current_facts = self.get_facts(
            subject_id=fact.subject_id,
            predicate=fact.predicate,
            include_inactive=False,
            limit=self.query_limit,
        )
        for existing in current_facts:
            if facts_conflict(existing, fact):
                self._run_named_query(
                    "update_fact_status",
                    (
                        f"MATCH (fact:{FACT_LABEL} {{id: $fact_id}}) "
                        "SET fact.invalid_at = $invalid_at, fact.status = $status RETURN fact"
                    ),
                    {
                        "fact_id": existing.id,
                        "invalid_at": fact.valid_at.isoformat(),
                        "status": FactStatus.INVALIDATED.value,
                    },
                )

    def _run_named_query(
        self,
        name: str,
        query: str,
        parameters: dict[str, Any] | None = None,
    ) -> tuple[Any, Any, list[str]]:
        return self.graph_store.run_cypher(f"// ontology: {name}\n{query}", parameters or {})

    @staticmethod
    def _serialize_model(model: BaseModel) -> dict[str, Any]:
        return {key: OntologyMemory._serialize_value(value) for key, value in model.model_dump().items()}

    @staticmethod
    def _serialize_value(value: Any) -> Any:
        if isinstance(value, datetime):
            return value.isoformat()
        if isinstance(value, Enum):
            return value.value
        if isinstance(value, list):
            return [OntologyMemory._serialize_value(item) for item in value]
        if isinstance(value, dict):
            return {key: OntologyMemory._serialize_value(item) for key, item in value.items()}
        return value

    @staticmethod
    def _normalize_record(record: Any) -> dict[str, Any]:
        if isinstance(record, dict):
            return record
        if hasattr(record, "data") and callable(record.data):
            return record.data()
        logger.warning("Unexpected graph record type: %s", type(record).__name__)
        return {}

    @staticmethod
    def _normalize_value(value: Any) -> dict[str, Any] | None:
        if value is None:
            return None
        if isinstance(value, dict):
            return value
        if hasattr(value, "items"):
            return dict(value.items())
        try:
            return dict(value)
        except Exception:  # noqa: BLE001
            logger.warning("Unable to normalize graph value type: %s", type(value).__name__)
            return None
