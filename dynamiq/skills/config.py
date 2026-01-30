"""Skills configuration: source and executor (like memory backend)."""

from pydantic import BaseModel, ConfigDict, Field

from dynamiq.skills.executor import SkillExecutor
from dynamiq.skills.sources.base import SkillSource


class SkillsConfig(BaseModel):
    """Configuration for agent skills: where they load from and where scripts run."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    source: SkillSource = Field(..., description="Where skills are discovered and loaded from")
    executor: SkillExecutor | None = Field(
        default=None,
        description="Where skill scripts run (e.g. subprocess sandbox, E2B). None disables run_script.",
    )
