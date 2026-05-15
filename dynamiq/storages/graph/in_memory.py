from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from dynamiq.storages.graph.base import BaseGraphStore


class InMemoryGraphStore(BaseGraphStore):
    """Small in-memory graph store for ontology-memory tests and examples.

    It supports the named ontology queries emitted by ``OntologyMemory``.
    It is not a general Cypher interpreter.
    """

    def __init__(self, state_file: str | None = None) -> None:
        self.state_file = state_file
        self.episodes: dict[str, dict[str, Any]] = {}
        self.entities: dict[str, dict[str, Any]] = {}
        self.episode_entities: dict[str, list[str]] = {}
        self.facts: dict[str, dict[str, Any]] = {}
        self.fact_subjects: dict[str, str] = {}
        self.fact_objects: dict[str, str] = {}
        self.fact_values: dict[str, dict[str, Any]] = {}
        self.fact_episodes: dict[str, list[str]] = {}
        self.provenance: dict[str, dict[str, Any]] = {}
        self.provenance_episode_links: dict[str, str] = {}
        self.provenance_fact_links: dict[str, str] = {}
        self._load_state()

    def run_cypher(
        self,
        query: str,
        parameters: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> tuple[list[Any], dict[str, Any], list[str]]:
        params = parameters or {}
        name = self._extract_name(query)

        if query.startswith("CREATE CONSTRAINT") or query.startswith("CREATE INDEX"):
            return [], {"query": query, "counters": {}}, []

        if name == "upsert_episode":
            self.episodes[params["id"]] = dict(params["properties"])
            self._save_state()
            return [{"episode": dict(self.episodes[params["id"]])}], {"query": query, "counters": {}}, ["episode"]

        if name == "upsert_entity":
            self.entities[params["id"]] = dict(params["properties"])
            self._save_state()
            return [{"entity": dict(self.entities[params["id"]])}], {"query": query, "counters": {}}, ["entity"]

        if name == "link_episode_entity":
            if params["episode_id"] not in self.episodes:
                raise ValueError(f"Unknown episode_id: {params['episode_id']}")
            if params["entity_id"] not in self.entities:
                raise ValueError(f"Unknown entity_id: {params['entity_id']}")
            entity_ids = self.episode_entities.setdefault(params["episode_id"], [])
            if params["entity_id"] not in entity_ids:
                entity_ids.append(params["entity_id"])
                self._save_state()
            return [{"entity": dict(self.entities[params["entity_id"]])}], {"query": query, "counters": {}}, ["entity"]

        if name == "upsert_fact":
            if params["subject_id"] not in self.entities:
                raise ValueError(f"Unknown subject_id: {params['subject_id']}")
            self.facts[params["fact_id"]] = dict(params["properties"])
            self.fact_subjects[params["fact_id"]] = params["subject_id"]
            self._save_state()
            return [{"fact": dict(self.facts[params["fact_id"]])}], {"query": query, "counters": {}}, ["fact"]

        if name == "link_fact_object_entity":
            if params["object_id"] not in self.entities:
                raise ValueError(f"Unknown object_id: {params['object_id']}")
            self.fact_objects[params["fact_id"]] = params["object_id"]
            self._save_state()
            return [{"fact": dict(self.facts[params["fact_id"]])}], {"query": query, "counters": {}}, ["fact"]

        if name == "link_fact_object_value":
            self.fact_values[params["fact_id"]] = dict(params["value_properties"])
            self._save_state()
            return [{"fact": dict(self.facts[params["fact_id"]])}], {"query": query, "counters": {}}, ["fact"]

        if name == "link_episode_fact":
            episode_ids = self.fact_episodes.setdefault(params["fact_id"], [])
            if params["episode_id"] not in episode_ids:
                episode_ids.append(params["episode_id"])
                self._save_state()
            return [{"fact": dict(self.facts[params["fact_id"]])}], {"query": query, "counters": {}}, ["fact"]

        if name == "upsert_provenance":
            self.provenance[params["id"]] = dict(params["properties"])
            self._save_state()
            return (
                [{"provenance": dict(self.provenance[params["id"]])}],
                {"query": query, "counters": {}},
                ["provenance"],
            )

        if name == "link_provenance_episode":
            self.provenance_episode_links[params["id"]] = params["episode_id"]
            self._save_state()
            return (
                [{"provenance": dict(self.provenance[params["id"]])}],
                {"query": query, "counters": {}},
                ["provenance"],
            )

        if name == "link_provenance_fact":
            self.provenance_fact_links[params["id"]] = params["fact_id"]
            self._save_state()
            return (
                [{"provenance": dict(self.provenance[params["id"]])}],
                {"query": query, "counters": {}},
                ["provenance"],
            )

        if name == "list_facts":
            return self._list_facts(params), {"query": query, "counters": {}}, ["fact", "subject", "object", "value"]

        if name == "list_entities":
            return self._list_entities(params), {"query": query, "counters": {}}, ["entity"]

        if name == "list_episodes":
            return self._list_episodes(params), {"query": query, "counters": {}}, ["episode"]

        if name == "list_episode_entities":
            return self._list_episode_entities(params), {"query": query, "counters": {}}, ["entity", "episode"]

        if name == "list_provenance":
            return self._list_provenance(params), {"query": query, "counters": {}}, ["provenance"]

        if name == "update_fact_status":
            fact = self.facts[params["fact_id"]]
            fact["invalid_at"] = params["invalid_at"]
            fact["status"] = params["status"]
            self._save_state()
            return [{"fact": dict(fact)}], {"query": query, "counters": {}}, ["fact"]

        raise NotImplementedError(f"Unsupported in-memory graph query: {name or query}")

    def clear(self) -> None:
        self.episodes.clear()
        self.entities.clear()
        self.episode_entities.clear()
        self.facts.clear()
        self.fact_subjects.clear()
        self.fact_objects.clear()
        self.fact_values.clear()
        self.fact_episodes.clear()
        self.provenance.clear()
        self.provenance_episode_links.clear()
        self.provenance_fact_links.clear()
        self._save_state()

    def introspect_schema(self, *, include_properties: bool, **kwargs: Any) -> dict[str, Any]:
        return {
            "labels": ["OntologyEpisode", "OntologyEntity", "OntologyFact", "OntologyValue", "OntologyProvenance"],
            "relationship_types": [
                "HAS_FACT",
                "FACT_OBJECT",
                "FACT_OBJECT_VALUE",
                "SUPPORTS_FACT",
                "MENTIONED_ENTITY",
                "PROVENANCE_EPISODE",
                "PROVENANCE_FACT",
            ],
            "node_properties": [],
            "relationship_properties": [],
        }

    @staticmethod
    def _extract_name(query: str) -> str | None:
        first_line = query.strip().splitlines()[0] if query.strip() else ""
        prefix = "// ontology: "
        return first_line[len(prefix) :].strip() if first_line.startswith(prefix) else None

    def _list_facts(self, params: dict[str, Any]) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for fact_id, fact in self.facts.items():
            subject_id = self.fact_subjects.get(fact_id)
            if params.get("fact_id") and fact_id != params["fact_id"]:
                continue
            if params.get("subject_id") and subject_id != params["subject_id"]:
                continue
            if params.get("predicate") and fact.get("predicate") != params["predicate"]:
                continue
            if params.get("user_id") and fact.get("user_id") != params["user_id"]:
                continue
            if params.get("session_id") and fact.get("session_id") != params["session_id"]:
                continue
            if not params.get("include_inactive") and fact.get("status") != params["active_status"]:
                continue
            valid_at = params.get("valid_at")
            if valid_at:
                if fact.get("valid_at") > valid_at:
                    continue
                if fact.get("invalid_at") and fact.get("invalid_at") <= valid_at:
                    continue
            results.append(
                {
                    "fact": dict(fact),
                    "subject": dict(self.entities[subject_id]) if subject_id else None,
                    "object": dict(self.entities[self.fact_objects[fact_id]]) if fact_id in self.fact_objects else None,
                    "value": dict(self.fact_values[fact_id]) if fact_id in self.fact_values else None,
                }
            )

        results.sort(key=lambda row: (row["fact"].get("valid_at", ""), row["fact"].get("created_at", "")), reverse=True)
        return results[: params["limit"]]

    def _list_entities(self, params: dict[str, Any]) -> list[dict[str, Any]]:
        rows = []
        for entity in self.entities.values():
            if params.get("entity_id") and entity.get("id") != params["entity_id"]:
                continue
            if params.get("entity_type") and entity.get("entity_type") != params["entity_type"]:
                continue
            rows.append({"entity": dict(entity)})
        rows.sort(key=lambda row: row["entity"].get("updated_at", ""), reverse=True)
        return rows[: params["limit"]]

    def _list_episodes(self, params: dict[str, Any]) -> list[dict[str, Any]]:
        rows = []
        for episode in self.episodes.values():
            if params.get("episode_id") and episode.get("id") != params["episode_id"]:
                continue
            if params.get("user_id") and episode.get("user_id") != params["user_id"]:
                continue
            if params.get("session_id") and episode.get("session_id") != params["session_id"]:
                continue
            rows.append({"episode": dict(episode)})
        rows.sort(key=lambda row: row["episode"].get("observed_at", ""), reverse=True)
        return rows[: params["limit"]]

    def _list_episode_entities(self, params: dict[str, Any]) -> list[dict[str, Any]]:
        rows = []
        for episode_id, entity_ids in self.episode_entities.items():
            episode = self.episodes.get(episode_id)
            if not episode:
                continue
            if params.get("episode_id") and episode_id != params["episode_id"]:
                continue
            if params.get("user_id") and episode.get("user_id") != params["user_id"]:
                continue
            if params.get("session_id") and episode.get("session_id") != params["session_id"]:
                continue
            for entity_id in entity_ids:
                rows.append(
                    {
                        "entity": dict(self.entities[entity_id]),
                        "episode": dict(episode),
                    }
                )
        rows.sort(key=lambda row: row["episode"].get("observed_at", ""), reverse=True)
        return rows[: params["limit"]]

    def _list_provenance(self, params: dict[str, Any]) -> list[dict[str, Any]]:
        rows = []
        for provenance_id, provenance in self.provenance.items():
            if params.get("fact_id") and self.provenance_fact_links.get(provenance_id) != params["fact_id"]:
                continue
            rows.append({"provenance": dict(provenance)})
        rows.sort(key=lambda row: row["provenance"].get("created_at", ""), reverse=True)
        return rows[: params["limit"]]

    def _load_state(self) -> None:
        if not self.state_file:
            return
        path = Path(self.state_file)
        if not path.exists():
            return
        payload = json.loads(path.read_text())
        self.episodes = payload.get("episodes", {})
        self.entities = payload.get("entities", {})
        self.episode_entities = payload.get("episode_entities", {})
        self.facts = payload.get("facts", {})
        self.fact_subjects = payload.get("fact_subjects", {})
        self.fact_objects = payload.get("fact_objects", {})
        self.fact_values = payload.get("fact_values", {})
        self.fact_episodes = payload.get("fact_episodes", {})
        self.provenance = payload.get("provenance", {})
        self.provenance_episode_links = payload.get("provenance_episode_links", {})
        self.provenance_fact_links = payload.get("provenance_fact_links", {})

    def _save_state(self) -> None:
        if not self.state_file:
            return
        path = Path(self.state_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "episodes": self.episodes,
            "entities": self.entities,
            "episode_entities": self.episode_entities,
            "facts": self.facts,
            "fact_subjects": self.fact_subjects,
            "fact_objects": self.fact_objects,
            "fact_values": self.fact_values,
            "fact_episodes": self.fact_episodes,
            "provenance": self.provenance,
            "provenance_episode_links": self.provenance_episode_links,
            "provenance_fact_links": self.provenance_fact_links,
        }
        path.write_text(json.dumps(payload, indent=2, sort_keys=True))
