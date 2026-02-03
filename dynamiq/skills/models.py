from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


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
