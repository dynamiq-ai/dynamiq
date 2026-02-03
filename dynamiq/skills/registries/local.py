"""Local filesystem-backed skill registry."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

from dynamiq.skills.models import SkillInstructions, SkillMetadata, SkillRegistryError
from dynamiq.skills.registries.base import BaseSkillRegistry


class LocalSkillWhitelistEntry(BaseModel):
    """Whitelist entry for local skill registry."""

    name: str = Field(..., min_length=1, description="Skill name.")
    description: str | None = Field(default=None, description="Optional cached skill description.")


class Local(BaseSkillRegistry):
    """Local filesystem-backed skill registry."""

    base_path: str = Field(
        default="~/.dynamiq/skills",
        description="Base path for local skills.",
    )
    whitelist: list[LocalSkillWhitelistEntry] = Field(default_factory=list)

    def get_skills_metadata(self) -> list[SkillMetadata]:
        metadata: list[SkillMetadata] = []
        for entry in self.whitelist:
            metadata.append(SkillMetadata(name=entry.name, description=entry.description))
        return metadata

    def get_skill_instructions(self, name: str) -> SkillInstructions:
        skill_path = self._resolve_skill_path(name)
        if not skill_path.exists():
            raise SkillRegistryError(
                "Local skill instructions not found.",
                details={"name": name, "path": str(skill_path)},
            )
        instructions = skill_path.read_text(encoding="utf-8")
        return SkillInstructions(name=name, description=None, instructions=instructions)

    def _resolve_skill_path(self, skill_name: str) -> Path:
        base = Path(self.base_path).expanduser()
        return base / skill_name / "SKILL.md"
