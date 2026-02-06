"""Skills configuration: enabled + source registry (Dynamiq or FileSystem)."""

from __future__ import annotations

import importlib
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from dynamiq.skills.registries import BaseSkillRegistry
from dynamiq.skills.types import SkillInstructions, SkillMetadata, SkillRegistryError


class SkillsConfig(BaseModel):
    """Configuration for agent skills."""

    enabled: bool = Field(default=False, description="Enable skill support for the agent.")
    source: BaseSkillRegistry | None = Field(
        default=None,
        description="Registry providing skills (Dynamiq or FileSystem).",
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @field_validator("source", mode="before")
    @classmethod
    def resolve_source(cls, v: Any) -> Any:
        if v is None or isinstance(v, BaseSkillRegistry):
            return v
        if isinstance(v, dict):
            registry_type = v.get("type")
            if not registry_type:
                return v
            if "." not in registry_type:
                raise SkillRegistryError(
                    "Registry type must be a fully qualified class name"
                    " (e.g. dynamiq.skills.registries.filesystem.FileSystem).",
                    details={"type": registry_type},
                )
            module_name, class_name = registry_type.rsplit(".", 1)
            module = importlib.import_module(module_name)
            registry_cls = getattr(module, class_name)
            init_data = {k: val for k, val in v.items() if k != "type"}
            return registry_cls(**init_data)
        return v

    @model_validator(mode="after")
    def validate_enabled_source(self) -> SkillsConfig:
        if self.enabled and self.source is None:
            raise SkillRegistryError("Skills are enabled but no source registry is configured.")
        return self

    def get_skills_metadata(self) -> list[SkillMetadata]:
        if not self.enabled or self.source is None:
            return []
        return self.source.get_skills_metadata()

    def get_skill_instructions(self, name: str) -> SkillInstructions:
        if not self.enabled or self.source is None:
            raise SkillRegistryError("Skills are disabled for this agent.")
        return self.source.get_skill_instructions(name)

    def to_dict(self, **kwargs: Any) -> dict[str, Any]:
        """Serialize config to dict for YAML/JSON. Source connection is serialized via connection.to_dict."""
        data = self.model_dump()
        if self.source is not None and getattr(self.source, "connection", None) is not None:
            if "source" in data:
                data["source"]["connection"] = self.source.connection.to_dict(**kwargs)
        return data
