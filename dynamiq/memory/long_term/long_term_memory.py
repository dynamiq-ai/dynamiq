from functools import cached_property
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, computed_field

from dynamiq.memory.long_term.base import LongTermMemoryBackend


class LongTermMemoryConfig(BaseModel):
    """Agent-level on/off switch + backend for long-term memory."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    enabled: bool = True
    backend: LongTermMemoryBackend = Field(
        ...,
        description="Backend engine that stores facts, embeds text, and serves remember/recall/forget.",
    )

    @computed_field
    @cached_property
    def type(self) -> str:
        return f"{self.__module__.rsplit('.', 1)[0]}.{self.__class__.__name__}"

    @property
    def to_dict_exclude_params(self) -> dict[str, bool]:
        return {"backend": True}

    def to_dict(self, include_secure_params: bool = False, **kwargs) -> dict[str, Any]:
        for_tracing = kwargs.pop("for_tracing", False)
        data = self.model_dump(exclude=kwargs.pop("exclude", self.to_dict_exclude_params), **kwargs)
        data["backend"] = self.backend.to_dict(
            include_secure_params=include_secure_params, for_tracing=for_tracing, **kwargs
        )
        return data
