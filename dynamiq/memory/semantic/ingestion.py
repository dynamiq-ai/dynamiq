from __future__ import annotations

from typing import Any

from dynamiq.memory.semantic.memory import OntologyMemory
from dynamiq.ontology import Episode


class OntologyIngestionService:
    """Thin orchestration layer around OntologyMemory for episode ingestion."""

    def __init__(self, memory: OntologyMemory):
        self.memory = memory

    def add_episode(self, episode: Episode | None = None, **kwargs: Any) -> Episode:
        return self.memory.add_episode(episode=episode, **kwargs)

    def extract_and_commit(self, episode_id: str | None = None, episode: Episode | None = None) -> dict[str, Any]:
        return self.memory.extract_and_commit(episode_id=episode_id, episode=episode)
