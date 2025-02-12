from enum import Enum
from functools import cached_property
from queue import Queue
from threading import Event
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from dynamiq.utils import generate_uuid


class StreamingMode(str, Enum):
    """Enumeration for streaming modes."""

    FINAL = "final"  # Streams only final output in agents nodes.
    ALL = "all"  # Streams all intermediate steps and final output in agents and llms nodes.


STREAMING_EVENT = "streaming"


class StreamingEventMessage(BaseModel):
    """Message for streaming events.

    Attributes:
        run_id (str | None): Run ID.
        wf_run_id (str | None): Workflow run ID. Defaults to a generated UUID.
        entity_id (str): Entity ID.
        data (Any): Data associated with the event.
        event (str | None): Event name. Defaults to "streaming".
    """

    run_id: str | None = None
    wf_run_id: str | None = Field(default_factory=generate_uuid)
    entity_id: str
    data: Any
    event: str | None = None

    @field_validator("event")
    @classmethod
    def set_event(cls, value: str | None) -> str:
        """Set the event name.

        Args:
            value (str | None): Event name.

        Returns:
            str: Event name or default.
        """
        return value or STREAMING_EVENT

    def to_dict(self, **kwargs) -> dict:
        """Convert to dictionary.

        Returns:
            dict: Dictionary representation.
        """
        return self.model_dump(**kwargs)

    def to_json(self, **kwargs) -> str:
        """Convert to JSON string.

        Returns:
            str: JSON string representation.
        """
        return self.model_dump_json(**kwargs)


class StreamingConfig(BaseModel):
    """Configuration for streaming.

    Attributes:
        enabled (bool): Whether streaming is enabled. Defaults to False.
        event (str): Event name. Defaults to "streaming".
        timeout (float | None): Timeout for streaming. Defaults to None.
        input_queue (Queue | None): Input queue for streaming. Defaults to None.
        input_queue_done_event (Event | None): Event to signal input queue completion. Defaults to None.
        mode (StreamingMode): Streaming mode. Defaults to StreamingMode.ANSWER.
        by_tokens (bool): Whether to stream  by tokens. Defaults to False.
    """
    enabled: bool = False
    event: str = STREAMING_EVENT
    timeout: float | None = None
    input_queue: Queue | None = None
    input_queue_done_event: Event | None = None
    mode: StreamingMode = StreamingMode.FINAL
    by_tokens: bool = True

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @cached_property
    def input_streaming_enabled(self) -> bool:
        """Check if input streaming is enabled.

        Returns:
            bool: True if input streaming is enabled, False otherwise.
        """
        return self.enabled and self.input_queue
