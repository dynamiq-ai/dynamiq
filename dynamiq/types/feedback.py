from enum import Enum

from pydantic import BaseModel

from dynamiq.types.streaming import StreamingEventMessage


class FeedbackMethod(Enum):
    CONSOLE = "console"
    STREAM = "stream"


APPROVAL_EVENT = "approval"


class ApprovalConfig(BaseModel):
    enabled: bool = False
    feedback_method: FeedbackMethod = FeedbackMethod.CONSOLE
    msg_template: str = (
        "Node {{name}}: Approve or cancel node execution. Send nothing for approval; provide feedback to cancel. "
    )
    event: str = APPROVAL_EVENT
    accept_pattern: str = ""
