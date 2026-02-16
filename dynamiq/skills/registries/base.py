"""Base skill registry interface."""

from abc import ABC, abstractmethod
from functools import cached_property
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, computed_field

from dynamiq.skills.types import SkillInstructions, SkillMetadata


class BaseSkillRegistry(ABC, BaseModel):
    """Abstract base class for skill registries."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    skills: list[Any] = Field(
        default_factory=list,
        description="Skills available in this registry.",
    )

    @computed_field
    @cached_property
    def type(self) -> str:
        return f"{self.__module__}.{self.__class__.__name__}"

    @abstractmethod
    def get_skills_metadata(self) -> list[SkillMetadata]:
        raise NotImplementedError

    @abstractmethod
    def get_skill_instructions(self, name: str) -> SkillInstructions:
        raise NotImplementedError

    def get_skill_scripts_path(self, name: str) -> str | None:
        """Return absolute scripts path for a skill when available.

        Registries that can expose runnable scripts should override this.
        Default behavior returns None.
        """
        return None
