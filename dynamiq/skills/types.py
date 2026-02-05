from typing import Any

from pydantic import BaseModel, Field


class SkillRegistryError(Exception):
    """Base exception for skill registry operations."""

    def __init__(self, message: str, *, details: dict[str, Any] | None = None):
        super().__init__(message)
        self.details = details or {}


class SkillMetadata(BaseModel):
    """Unified metadata shape for skills (registry API)."""

    name: str = Field(..., description="Skill name.")
    description: str | None = Field(default=None, description="Optional skill description.")


class SkillInstructions(SkillMetadata):
    """Unified instructions shape for skills (registry API).

    Supports both plain-text instructions and API responses that include
    an instructions field plus optional metadata (e.g. version, format).
    """

    instructions: str = Field(default="", description="Full skill instructions (e.g. SKILL.md content).")
    metadata: dict[str, Any] | None = Field(
        default=None,
        description="Optional metadata from the registry API (e.g. version, content_type).",
    )
