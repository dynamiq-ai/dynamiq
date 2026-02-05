"""Filesystem-backed skill registry."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

from dynamiq.skills.registries.base import BaseSkillRegistry
from dynamiq.skills.types import SkillInstructions, SkillMetadata, SkillRegistryError


class FileSystemSkillEntry(BaseModel):
    """Allowed skill entry for filesystem skill registry."""

    name: str = Field(..., min_length=1, description="Skill name.")
    description: str | None = Field(default=None, description="Optional cached skill description.")


class FileSystem(BaseSkillRegistry):
    """Filesystem-backed skill registry."""

    base_path: str = Field(
        default="~/.dynamiq/skills",
        description="Base path for skills on the filesystem.",
    )
    skill_filename: str = Field(
        default="SKILL.md",
        description="Filename for skill instructions within each skill directory.",
    )
    allowed_skills: list[FileSystemSkillEntry] = Field(default_factory=list)

    def get_skills_metadata(self) -> list[SkillMetadata]:
        metadata: list[SkillMetadata] = []
        for entry in self.allowed_skills:
            metadata.append(SkillMetadata(name=entry.name, description=entry.description))
        return metadata

    def get_skill_instructions(self, name: str) -> SkillInstructions:
        entry = self._get_entry_by_name(name)
        skill_path = self._resolve_skill_path(name)
        if not skill_path.exists():
            raise SkillRegistryError(
                "Skill instructions file not found.",
                details={"name": name, "path": str(skill_path)},
            )
        instructions = skill_path.read_text(encoding="utf-8")
        return SkillInstructions(
            name=entry.name,
            description=entry.description,
            instructions=instructions,
        )

    def _get_entry_by_name(self, name: str) -> FileSystemSkillEntry:
        for entry in self.allowed_skills:
            if entry.name == name:
                return entry
        raise SkillRegistryError("Skill not in allowed skills.", details={"name": name})

    def _resolve_skill_path(self, skill_name: str) -> Path:
        if "/" in skill_name or "\\" in skill_name or ".." in skill_name:
            raise SkillRegistryError(
                "Invalid skill name: path components are not allowed.",
                details={"name": skill_name},
            )
        base = Path(self.base_path).expanduser().resolve()
        full_path = (base / skill_name / self.skill_filename).resolve()
        try:
            full_path.relative_to(base)
        except ValueError:
            raise SkillRegistryError(
                "Invalid skill path: resolved path is outside base path.",
                details={"name": skill_name, "path": str(full_path)},
            )
        return full_path
