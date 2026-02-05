"""Base skill registry interface."""

from abc import ABC, abstractmethod
from functools import cached_property

from pydantic import BaseModel, ConfigDict, computed_field

from dynamiq.skills.models import SkillInstructions, SkillMetadata


class BaseSkillRegistry(ABC, BaseModel):
    """Abstract base class for skill registries."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

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
