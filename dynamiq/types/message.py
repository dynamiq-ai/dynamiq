from typing import Any

from pydantic import BaseModel, Field, field_validator

from dynamiq.utils.utils import generate_uuid


class EventMessage(BaseModel):
    """Message for events.

    Attributes:
        run_id (str | None): Run ID.
        wf_run_id (str | None): Workflow run ID. Defaults to a generated UUID.
        entity_id (str): Entity ID.
        data (Any): Data associated with the event. Defaults to None.
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
        return value

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
