from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class SkillWhitelistEntry(BaseModel):
    """Whitelist entry describing which skills are available to the agent."""

    id: str = Field(..., min_length=1, description="Skill identifier.")
    version_id: str | None = Field(default=None, description="Skill version identifier.")
    name: str | None = Field(default=None, description="Optional cached skill name.")
    description: str | None = Field(default=None, description="Optional cached skill description.")


class LocalSkillWhitelistEntry(BaseModel):
    """Whitelist entry for local skill registry."""

    name: str = Field(..., min_length=1, description="Skill name.")
    description: str | None = Field(default=None, description="Optional cached skill description.")


class SkillMetadata(BaseModel):
    """Unified metadata shape for skills."""

    name: str
    description: str | None = None


class SkillInstructions(SkillMetadata):
    """Unified instructions shape for skills."""

    instructions: str = Field(default="", description="Full skill instructions.")


class SkillRegistryError(Exception):
    """Base exception for skill registry operations."""

    def __init__(self, message: str, *, details: dict[str, Any] | None = None):
        super().__init__(message)
        self.details = details or {}
