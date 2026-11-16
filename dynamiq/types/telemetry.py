"""Per-run tracing settings carried by ``RunnableConfig.tracing``."""

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ContentCaptureMode(str, Enum):
    """Whether prompts, outputs and other content are recorded on spans."""

    NO_CONTENT = "NO_CONTENT"
    SPAN_ONLY = "SPAN_ONLY"


class TracingConfig(BaseModel):
    """Settings of the OpenTelemetry spans emitted for one run."""

    enabled: bool = Field(default=True, description="Whether spans are emitted for the run.")
    tracer_provider: Any | None = Field(
        default=None,
        exclude=True,
        description="OpenTelemetry TracerProvider for the run instead of the global one. Never serialized.",
    )
    session_id: str | None = Field(default=None, description="Conversation or session the run belongs to.")
    user_id: str | None = Field(default=None, description="End user the run is made for.")
    tags: list[str] = Field(default_factory=list, description="Tags recorded on every span of the run.")
    metadata: dict[str, str | int | float | bool] = Field(
        default_factory=dict, description="Key-value pairs recorded on every span of the run."
    )
    source_id: str | None = Field(default=None, description="Platform entity the trace belongs to.")
    content_capture: ContentCaptureMode | None = Field(
        default=None, description="Content capture mode for the run. None reads the environment."
    )
    binary_content_enabled: bool | None = Field(
        default=None, description="Whether inline binary message content is recorded. None reads the environment."
    )
    node_config_enabled: bool | None = Field(
        default=None, description="Whether node configuration is recorded. None reads the environment."
    )
    content_max_length: int | None = Field(
        default=None, gt=0, description="Longest string kept inside recorded content. None reads the environment."
    )
    prompt_compaction_enabled: bool | None = Field(
        default=None,
        description=(
            "Whether a chat span that continues the prompt of the previous chat span of the same LLM under the same "
            "parent run records only the messages after the shared start, with a pointer to that span. Only a "
            "backend that follows the pointer shows the whole prompt. None reads the environment."
        ),
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __getstate__(self) -> dict[Any, Any]:
        # A provider holds locks and exporter threads and cannot be pickled; ProcessExecutor pickles the run config.
        state = super().__getstate__()
        return {**state, "__dict__": {**state["__dict__"], "tracer_provider": None}}
