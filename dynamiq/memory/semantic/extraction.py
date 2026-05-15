from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from dynamiq.nodes.llms.base import BaseLLM
from dynamiq.ontology import Episode, OntologyEntityType
from dynamiq.prompts import Message, MessageRole, Prompt
from dynamiq.runnables import RunnableResult
from dynamiq.utils.json_parser import parse_llm_json_output

EXTRACTION_SYSTEM_PROMPT = """You extract ontology memory records from source episodes.

Return only valid JSON matching the requested schema.

Rules:
- Extract durable facts only. Ignore stylistic filler and one-off chit-chat.
- Reuse the user label from metadata when the speaker refers to themselves as "I".
- Prefer canonical entity labels without trailing punctuation.
- Use only these entity types when appropriate:
  User, Agent, Workflow, Node, Tool, Session, Document, Task, Organization, Concept, Preference, Policy, Dataset
- A fact must include subject_label, predicate, and either object_label or object_value.
- Confidence must be between 0 and 1.
- If there are no durable facts, return empty lists.
"""

ExtractionLiteralValue = str | int | float | bool


class ExtractedEntity(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    label: str = Field(min_length=1, description="Canonical display label for the entity.")
    entity_type: OntologyEntityType = Field(description="Ontology entity type.")
    aliases: list[str] = Field(default_factory=list, description="Known aliases or alternate spellings.")
    summary: str | None = Field(default=None, description="Optional short summary of the entity.")
    confidence: float = Field(default=0.9, ge=0.0, le=1.0, description="Confidence in the extracted entity.")


class ExtractedFact(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    subject_label: str = Field(min_length=1, description="Label of the subject entity.")
    predicate: str = Field(min_length=1, description="Canonical predicate name, e.g. has_preference.")
    object_label: str | None = Field(default=None, description="Label of the object entity when relational.")
    object_value: ExtractionLiteralValue | None = Field(
        default=None,
        description="Literal value when the fact object is not an entity.",
    )
    subject_type: OntologyEntityType = Field(default=OntologyEntityType.USER, description="Subject entity type.")
    object_type: OntologyEntityType | None = Field(default=None, description="Object entity type when present.")
    confidence: float = Field(default=0.85, ge=0.0, le=1.0, description="Confidence in the extracted fact.")


class ExtractionResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    entities: list[ExtractedEntity] = Field(default_factory=list)
    facts: list[ExtractedFact] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list, description="Optional extraction notes for audit/debugging.")


class OntologyExtractor(ABC, BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    @abstractmethod
    def extract(self, episode: Episode) -> ExtractionResult:
        raise NotImplementedError


class LLMOntologyExtractor(OntologyExtractor):
    """Structured ontology extractor backed by a Dynamiq LLM node."""

    llm: BaseLLM
    system_prompt: str = EXTRACTION_SYSTEM_PROMPT

    def extract(self, episode: Episode) -> ExtractionResult:
        prompt = Prompt(
            messages=[
                Message(role=MessageRole.SYSTEM, content=self.system_prompt, static=True),
                Message(role=MessageRole.USER, content=self._build_user_prompt(episode), static=True),
            ]
        )
        response = self.llm.run(
            input_data={},
            prompt=prompt,
            response_format=self.response_format_schema,
        )
        content = self._extract_content(response)
        payload = parse_llm_json_output(content) if isinstance(content, str) else content
        return ExtractionResult.model_validate(payload)

    @property
    def response_format_schema(self) -> dict[str, Any]:
        return _build_response_format(ExtractionResult)

    @staticmethod
    def _extract_content(response: RunnableResult | Any) -> str | dict[str, Any]:
        output = getattr(response, "output", response)
        if isinstance(output, dict):
            return output.get("content", output)
        return output

    @staticmethod
    def _build_user_prompt(episode: Episode) -> str:
        metadata = episode.metadata or {}
        return (
            "Extract ontology entities and durable facts from this episode.\n\n"
            f"episode_id: {episode.id}\n"
            f"source_type: {episode.source_type.value}\n"
            f"actor_id: {episode.actor_id}\n"
            f"user_id: {episode.user_id}\n"
            f"session_id: {episode.session_id}\n"
            f"workflow_id: {episode.workflow_id}\n"
            f"observed_at: {episode.observed_at.isoformat()}\n"
            f"metadata: {metadata}\n"
            f"content:\n{episode.content}\n"
        )


def _build_response_format(response_format: dict | type[BaseModel]) -> dict[str, Any]:
    if isinstance(response_format, type) and issubclass(response_format, BaseModel):
        schema = _to_openai_strict_schema(response_format.model_json_schema())
        name = response_format.__name__
    elif isinstance(response_format, dict):
        if response_format.get("type") == "json_schema" and "json_schema" in response_format:
            return response_format
        schema = _to_openai_strict_schema(response_format)
        name = "structured_output"
    else:
        raise TypeError(f"Unsupported response_format type: {type(response_format).__name__}")

    return {
        "type": "json_schema",
        "json_schema": {
            "name": name,
            "strict": True,
            "schema": schema,
        },
    }


def _to_openai_strict_schema(schema: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(schema, dict):
        raise TypeError(f"Schema must be a dict, got {type(schema).__name__}")
    return _normalize_schema_node(schema)


def _normalize_schema_node(node: Any) -> Any:
    if isinstance(node, list):
        return [_normalize_schema_node(item) for item in node]
    if not isinstance(node, dict):
        return node

    normalized = {key: _normalize_schema_node(value) for key, value in node.items() if key not in {"default", "title"}}

    if "$ref" in normalized:
        return {"$ref": normalized["$ref"]}

    properties = normalized.get("properties")
    schema_type = normalized.get("type")
    is_object = schema_type == "object" or isinstance(properties, dict)
    if is_object:
        properties = properties or {}
        normalized["properties"] = properties
        normalized["required"] = list(properties.keys())
        normalized["additionalProperties"] = False

    return normalized
