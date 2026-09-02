from functools import cached_property
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, computed_field

from dynamiq.memory.notes.base import NotesBackend


class NotesConfig(BaseModel):
    """Agent-level on/off switch + backend for the agent's titled notes.

    When enabled, the agent gets `write_note` / `read_note` / `delete_note` tools and an
    index of `title — description` lines injected into its system prompt on every run.

    Unrelated to the sandbox `agent_notes.md` compaction scratchpad (see
    `HistoryManagerMixin.get_notes_file_path`), which is a single markdown file the agent
    writes through its sandbox tools.

    Only the ReAct `Agent` renders the index block; the plain agent never builds a ReAct
    prompt and so never shows it.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    enabled: bool = True
    backend: NotesBackend = Field(..., description="Storage engine for user-scoped notes.")
    max_index_entries: int = Field(
        default=200,
        ge=1,
        description="Maximum `title — description` lines in the always-loaded index.",
    )
    max_index_chars: int = Field(
        default=6000,
        ge=0,
        description="Hard cap on the rendered index size; overflow is truncated visibly.",
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
