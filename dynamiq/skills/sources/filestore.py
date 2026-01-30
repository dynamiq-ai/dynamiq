"""FileStore-backed skill source (default)."""

from typing import Any

from pydantic import ConfigDict, Field, model_validator

from dynamiq.skills.loader import SkillLoader
from dynamiq.skills.models import Skill, SkillReference
from dynamiq.skills.sources.base import SkillSource
from dynamiq.storages.file.base import FileStore


class FileStoreSkillSource(SkillSource):
    """Skill source that reads skills from a FileStore (e.g. .skills/ prefix)."""

    name: str = "FileStoreSkillSource"
    file_store: FileStore = Field(..., description="FileStore containing skill files")
    skills_prefix: str = Field(default=".skills/", description="Prefix path for skills")

    model_config = ConfigDict(arbitrary_types_allowed=True)

    _loader: SkillLoader | None = None

    @model_validator(mode="after")
    def _init_loader(self) -> "FileStoreSkillSource":
        self._loader = SkillLoader(self.file_store, self.skills_prefix)
        return self

    def discover_skills(self) -> list[SkillReference]:
        return self._loader.discover_skills()

    def load_skill(self, name: str) -> Skill | None:
        return self._loader.load_skill(name)

    def load_skill_content(
        self,
        name: str,
        section: str | None = None,
        line_start: int | None = None,
        line_end: int | None = None,
    ) -> dict[str, Any] | None:
        """Use loader's implementation for section/line range."""
        return self._loader.load_skill_content(name, section=section, line_start=line_start, line_end=line_end)
