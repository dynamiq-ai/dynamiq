from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, model_validator

from dynamiq.types.streaming import StreamingEventMessage


class FeedbackMethod(Enum):
    CONSOLE = "console"
    STREAM = "stream"


APPROVAL_EVENT = "approval"
PLAN_APPROVAL_EVENT = "plan_approval"


class ApprovalOutputEventData(BaseModel):
    template: str = Field(..., description="Message template that will be sent.")
    data: dict[str, Any] = Field(..., description="Data that will be sent.")
    mutable_data_params: list[str] = Field(
        ...,
        description=(
            "List of parameters from 'data'"
            "field that will be possible to update."
            "This field is used to secure data that shouldn't be modified."
        ),
    )
    request_id: str | None = Field(
        default=None,
        description=(
            "Set for the approvals of an agent hook: the answer must echo it in `request_id`, and an answer with "
            "another id is ignored, so a late answer never approves the next call."
        ),
    )


class ApprovalStreamingOutputEventMessage(StreamingEventMessage):
    data: ApprovalOutputEventData


class ApprovalInputData(BaseModel):
    feedback: str = None
    data: dict[str, Any] = {}
    is_approved: bool | None = None
    request_id: str | None = None

    @model_validator(mode="after")
    def validate_feedback(self):
        if self.feedback is None and self.is_approved is None:
            raise ValueError("Error: No feedback or approval has been provided.")
        return self


class ApprovalStreamingInputEventMessage(StreamingEventMessage):
    data: ApprovalInputData


class ApprovalConfig(BaseModel):
    enabled: bool = False
    feedback_method: FeedbackMethod = FeedbackMethod.CONSOLE

    mutable_data_params: list[str] = []
    msg_template: str = """
            Node {{name}}: Approve or cancel execution. Send nothing for approval; provide feedback to cancel.
    """

    event: str = APPROVAL_EVENT
    accept_pattern: str = ""

    def to_dict(self, for_tracing: bool = False, **kwargs) -> dict:
        if for_tracing and not self.enabled:
            return {"enabled": False}
        return self.model_dump(**kwargs)


class PlanApprovalConfig(ApprovalConfig):
    msg_template: str = (
        """
            Approve or cancel plan. Send nothing for approval; provide feedback to cancel.
            {% for task in input_data.tasks %}
                Task name: {{ task.name }}
                Task description: {{ task.description }}
                Task dependencies: {{ task.dependencies }}
            {% endfor %}
            """
    )

    event: str = PLAN_APPROVAL_EVENT
