"""Skill source abstraction: where skills are loaded from (FileStore, E2B, etc.)."""

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from dynamiq.skills.models import Skill, SkillReference
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
        """Load skill content, optionally a section or line range. Default uses load_skill + slice."""
        skill = self.load_skill(name)
        if not skill:
            return None
        instructions = skill.instructions
        lines = instructions.splitlines()
        section_used: str | None = None

        if section:
            section_lower = section.strip().lower()
            start_i = None
            end_i = len(lines)
            for i, line in enumerate(lines):
                s = line.strip()
                if s.startswith("#"):
                    header_text = s.lstrip("#").strip().lower()
                    if header_text == section_lower:
                        start_i = i
                        for j in range(i + 1, len(lines)):
                            if lines[j].strip().startswith("##"):
                                end_i = j
                                break
                        section_used = section
                        break
            if start_i is not None:
                instructions = "\n".join(lines[start_i:end_i])
            else:
                section_used = None
        elif line_start is not None or line_end is not None:
            start = max(0, (line_start or 1) - 1)
            end = line_end if line_end is not None else len(lines)
            end = min(end, len(lines))
            instructions = "\n".join(lines[start:end])

        return {
            "skill_name": skill.name,
            "description": skill.metadata.description,
            "instructions": instructions,
            "section_used": section_used,
            "supporting_files": [str(p) for p in skill.supporting_files_paths],
            "dependencies": skill.metadata.dependencies,
        }
