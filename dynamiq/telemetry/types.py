"""Types shared by the span lifecycle (emitter) and the semantic mapping (semconv)."""

from enum import Enum
from typing import Any

from opentelemetry.trace import SpanKind
from pydantic import BaseModel, ConfigDict, Field


class SpanType(str, Enum):
    WORKFLOW = "workflow"
    AGENT = "agent"
    CHAT = "chat"
    EMBEDDINGS = "embeddings"
    RETRIEVAL = "retrieval"
    TOOL = "tool"
    NODE = "node"


class RunKind(str, Enum):
    WORKFLOW = "workflow"
    FLOW = "flow"
    NODE = "node"


class RunOutcome(str, Enum):
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    SKIPPED = "skipped"
    CANCELED = "canceled"
    PAUSED = "paused"


class CaptureSettings(BaseModel):
    """What one run tree records, resolved once at its root from TracingConfig and the environment."""

    content_enabled: bool = True
    binary_content_enabled: bool = False
    node_config_enabled: bool = False
    content_max_length: int | None = None
    prompt_compaction_enabled: bool = False
    session_id: str | None = None
    user_id: str | None = None
    source_id: str | None = None
    tags: tuple[str, ...] = ()
    metadata: dict[str, str | int | float | bool] = Field(default_factory=dict)

    model_config = ConfigDict(frozen=True)


class PromptRecord(BaseModel):
    """Prompt of a ``chat`` span, kept for the next run of the same LLM node under the same parent run to be
    compared with (prompt compaction)."""

    span_id: str
    messages: list[Any]

    model_config = ConfigDict(frozen=True)


class SpanSpec(BaseModel):
    """Everything needed to start a span, built before the sampling decision."""

    span_type: SpanType
    name: str
    kind: SpanKind
    attributes: dict[str, Any] = Field(default_factory=dict)


class ErrorSpec(BaseModel):
    """How a failure is recorded on a span."""

    error_type: str
    description: str | None = None
    event_attributes: dict[str, Any] | None = None
