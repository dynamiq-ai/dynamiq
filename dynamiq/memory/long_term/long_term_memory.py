from functools import cached_property
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, computed_field

from dynamiq.memory.long_term.base import LongTermMemoryBackend


class LongTermMemoryConfig(BaseModel):
    """Agent-level configuration for long-term memory.

    Mirrors `SandboxConfig` / `SkillsConfig`: an on/off switch plus the backend
    that does the work. All operations (remember/recall/forget) live on
    `backend`. The agent always exposes both the `remember_fact` and
    `recall_facts` tools when LTM is on; per-tool subsetting was intentionally
    removed — see plan v2 "Reversible cuts" appendix if it ever needs to come
    back.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    enabled: bool = True
    backend: LongTermMemoryBackend = Field(
        ...,
        description="Backend engine that stores facts, embeds text, and serves remember/recall/forget.",
    )

    @computed_field
    @cached_property
    def type(self) -> str:
        """Fully-qualified class id used by the YAML loader for reconstruction."""
        return f"{self.__module__.rsplit('.', 1)[0]}.{self.__class__.__name__}"

    @property
    def to_dict_exclude_params(self) -> dict[str, bool]:
        """Fields excluded from default model_dump; re-added by `to_dict`."""
        return {"backend": True}

    def to_dict(self, include_secure_params: bool = False, **kwargs) -> dict[str, Any]:
        """Serialize so the backend round-trips via its own `to_dict`."""
        for_tracing = kwargs.pop("for_tracing", False)
        data = self.model_dump(exclude=kwargs.pop("exclude", self.to_dict_exclude_params), **kwargs)
        data["backend"] = self.backend.to_dict(
            include_secure_params=include_secure_params, for_tracing=for_tracing, **kwargs
        )
        return data
