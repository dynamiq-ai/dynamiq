from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from dynamiq.utils import generate_uuid


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class EpisodeSourceType(str, Enum):
    MESSAGE = "message"
    TOOL_CALL = "tool_call"
    DOCUMENT = "document"
    API_EVENT = "api_event"
    MANUAL_NOTE = "manual_note"
    WORKFLOW_EVENT = "workflow_event"


class OntologyEntityType(str, Enum):
    USER = "User"
    AGENT = "Agent"
    WORKFLOW = "Workflow"
    NODE = "Node"
    TOOL = "Tool"
    SESSION = "Session"
    DOCUMENT = "Document"
    TASK = "Task"
    ORGANIZATION = "Organization"
    CONCEPT = "Concept"
    PREFERENCE = "Preference"
    POLICY = "Policy"
    DATASET = "Dataset"


class FactStatus(str, Enum):
    ACTIVE = "active"
    INVALIDATED = "invalidated"
    REJECTED = "rejected"
    NEEDS_REVIEW = "needs_review"


class Episode(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    id: str = Field(default_factory=generate_uuid)
    source_type: EpisodeSourceType
    source_id: str = Field(default_factory=generate_uuid)
    actor_id: str | None = None
    user_id: str | None = None
    session_id: str | None = None
    workflow_id: str | None = None
    content: str = Field(min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)
    observed_at: datetime = Field(default_factory=utc_now)
    created_at: datetime = Field(default_factory=utc_now)


class Entity(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    id: str = Field(default_factory=generate_uuid)
    ontology_uri: str | None = None
    label: str = Field(min_length=1)
    aliases: list[str] = Field(default_factory=list)
    entity_type: OntologyEntityType
    summary: str | None = None
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def populate_ontology_uri(self) -> Entity:
        if not self.ontology_uri:
            self.ontology_uri = f"dynamiq://{self.entity_type.value}/{self.id}"
        return self


class TemporalFact(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    id: str = Field(default_factory=generate_uuid)
    subject_id: str = Field(min_length=1)
    predicate: str = Field(min_length=1)
    object_id: str | None = None
    object_value: Any | None = None
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    valid_at: datetime = Field(default_factory=utc_now)
    invalid_at: datetime | None = None
    created_at: datetime = Field(default_factory=utc_now)
    expired_at: datetime | None = None
    episode_ids: list[str] = Field(default_factory=list)
    created_by_agent_id: str | None = None
    extraction_model: str | None = None
    user_id: str | None = None
    session_id: str | None = None
    workflow_id: str | None = None
    status: FactStatus = FactStatus.ACTIVE

    @model_validator(mode="after")
    def validate_object_target(self) -> TemporalFact:
        if self.object_id is None and self.object_value is None:
            raise ValueError("TemporalFact requires either object_id or object_value.")
        return self


class Provenance(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    id: str = Field(default_factory=generate_uuid)
    episode_id: str = Field(min_length=1)
    fact_id: str = Field(min_length=1)
    extracting_agent_id: str | None = None
    extraction_model: str | None = None
    prompt_template_version: str | None = None
    ingestion_run_id: str | None = None
    validation_result: str | None = None
    reviewer_id: str | None = None
    created_at: datetime = Field(default_factory=utc_now)
