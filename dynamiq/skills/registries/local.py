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
        entry = self._get_whitelist_entry_by_name(name)
        skill_path = self._resolve_skill_path(name)
        if not skill_path.exists():
            raise SkillRegistryError(
                "Local skill instructions not found.",
                details={"name": name, "path": str(skill_path)},
            )
        instructions = skill_path.read_text(encoding="utf-8")
        return SkillInstructions(
            name=entry.name,
            description=entry.description,
            instructions=instructions,
        )

    def _get_whitelist_entry_by_name(self, name: str) -> LocalSkillWhitelistEntry:
        for entry in self.whitelist:
            if entry.name == name:
                return entry
        raise SkillRegistryError("Skill not found in whitelist.", details={"name": name})

    def _resolve_skill_path(self, skill_name: str) -> Path:
        if "/" in skill_name or "\\" in skill_name or ".." in skill_name:
            raise SkillRegistryError(
                "Invalid skill name: path components are not allowed.",
                details={"name": skill_name},
            )
        base = Path(self.base_path).expanduser().resolve()
        full_path = (base / skill_name / "SKILL.md").resolve()
        try:
            full_path.relative_to(base)
        except ValueError:
            raise SkillRegistryError(
                "Invalid skill path: resolved path is outside base path.",
                details={"name": skill_name, "path": str(full_path)},
            )
        return full_path
