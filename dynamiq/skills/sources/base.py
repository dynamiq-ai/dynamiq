"""Skill source abstraction: where skills are loaded from (FileStore, E2B, etc.)."""

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from dynamiq.skills.models import Skill, SkillReference
from dynamiq.skills.utils import extract_skill_content_slice
from dynamiq.utils import generate_uuid


class SkillSource(ABC, BaseModel):
    """Abstract source for skills (discovery and content)."""

    name: str = "SkillSource"
    id: str = Field(default_factory=generate_uuid)
    model_config = ConfigDict(arbitrary_types_allowed=True)

    @abstractmethod
    def discover_skills(self) -> list[SkillReference]:
        """Return lightweight skill references (metadata only)."""
        raise NotImplementedError

    @abstractmethod
    def load_skill(self, name: str) -> Skill | None:
        """Load full skill by name."""
        raise NotImplementedError

    def load_skill_content(
        self,
        name: str,
        section: str | None = None,
        line_start: int | None = None,
        line_end: int | None = None,
    ) -> dict[str, Any] | None:
        """Load skill content, optionally a section or line range. Uses shared slice utility."""
        skill = self.load_skill(name)
        if not skill:
            return None
        instructions, section_used = extract_skill_content_slice(
            skill.instructions,
            section=section,
            line_start=line_start,
            line_end=line_end,
        )
        return {
            "skill_name": skill.name,
            "description": skill.metadata.description,
            "instructions": instructions,
            "section_used": section_used,
            "supporting_files": [str(p) for p in skill.supporting_files_paths],
            "dependencies": skill.metadata.dependencies,
        }
