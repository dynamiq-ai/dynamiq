from functools import cached_property
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, computed_field, field_serializer

from dynamiq.memory.long_term.base import LongTermMemoryBackend
from dynamiq.memory.long_term.types import MemoryToolKind


class LongTermMemoryConfig(BaseModel):
    """Agent-level configuration for long-term memory.

    Mirrors `SandboxConfig` / `SkillsConfig`: an on/off switch plus the backend
    that does the work, plus which memory tools to expose to the LLM. All
    operations (remember/recall/forget) live on `backend`.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    enabled: bool = True
    backend: LongTermMemoryBackend = Field(
        ...,
        description="Backend engine that stores facts, embeds text, and serves remember/recall/forget.",
    )
    tools: tuple[MemoryToolKind, ...] = Field(
        default=(MemoryToolKind.REMEMBER, MemoryToolKind.RECALL),
        description="Which long-term-memory tools to expose to the agent's LLM.",
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

    @field_serializer("tools")
    def _serialize_tools(self, tools: tuple[MemoryToolKind, ...]) -> tuple[str, ...]:
        # Emit plain string values so YAML round-trip and tracing work; pydantic
        # default-mode dump returns enum members which yaml.safe_dump cannot render.
        return tuple(t.value for t in tools)
