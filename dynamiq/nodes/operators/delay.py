import time
from datetime import datetime, timedelta, timezone
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from dynamiq.checkpoints.checkpoint import BaseCheckpointState
from dynamiq.checkpoints.types import RunPausedException
from dynamiq.nodes import Node, NodeGroup
from dynamiq.nodes.node import ensure_config
from dynamiq.runnables import RunnableConfig
from dynamiq.types.cancellation import check_cancellation

# A wait up to this long happens in place; a longer one pauses the run. Like n8n's Wait node, a short wait
# is not worth writing the run away for, and pausing a long one frees the worker and survives a restart.
INLINE_WAIT_LIMIT_SECONDS = 60

# How often an in-place wait checks for cancellation.
INLINE_WAIT_POLL_SECONDS = 0.5


class DelayInputSchema(BaseModel):
    """What a Delay waits for. Fields other than these pass through to its output."""

    model_config = ConfigDict(extra="allow")

    until: datetime | None = Field(
        default=None, description="Time to continue at, as ISO-8601. A time without a zone is taken as UTC."
    )
    duration_seconds: float | None = Field(
        default=None, ge=0, description="How long to wait. Overrides the node's own duration_seconds."
    )


class DelayCheckpointState(BaseCheckpointState):
    """The wait a Delay fixed on its first run, kept so a resumed run waits for the same time."""

    wait_started_at: datetime | None = None
    resume_at: datetime | None = None


class Delay(Node):
    """Waits before the flow continues: for a duration, or until a time taken from the input.

    The time to continue at is fixed on the first run and kept in the checkpoint, so neither a pause, an
    early resume nor a repeated one moves it. A wait of up to a minute happens in place. A longer one pauses
    the run: nodes that depend on the Delay wait with it, independent branches finish, and the run resumes
    when the time comes. Pausing needs checkpointing and a Delay at the top level of the flow; without them a
    longer wait fails with a message saying so, rather than holding a worker for hours.

    Everything the node receives passes through to its output, with `waited_until` (the time it waited for)
    and `waited_seconds` (how long it waited) added.
    """

    name: str | None = "delay"
    group: Literal[NodeGroup.OPERATORS] = NodeGroup.OPERATORS
    duration_seconds: float | None = Field(
        default=None, ge=0, description="How long to wait when the input gives neither `until` nor `duration_seconds`."
    )
    input_schema: ClassVar[type[DelayInputSchema]] = DelayInputSchema

    _wait_started_at: datetime | None = PrivateAttr(default=None)
    _resume_at: datetime | None = PrivateAttr(default=None)

    def to_checkpoint_state(self) -> DelayCheckpointState:
        state = super().to_checkpoint_state()
        return DelayCheckpointState(
            **state.model_dump(), wait_started_at=self._wait_started_at, resume_at=self._resume_at
        )

    def from_checkpoint_state(self, state: BaseCheckpointState | dict[str, Any]) -> None:
        super().from_checkpoint_state(state)
        restored = DelayCheckpointState.model_validate(state if isinstance(state, dict) else state.model_dump())
        self._wait_started_at = restored.wait_started_at
        self._resume_at = restored.resume_at

    def execute(self, input_data: DelayInputSchema, config: RunnableConfig = None, **kwargs) -> dict[str, Any]:
        """Waits until the time the first run fixed, in place or by pausing the run, then passes the input on."""
        config = ensure_config(config)
        self.run_on_node_execute_run(config.callbacks, **kwargs)

        now = datetime.now(timezone.utc)
        if not self.is_resumed or self._resume_at is None:
            self._wait_started_at = now
            self._resume_at = self._resolve_resume_at(input_data, now)

        remaining = (self._resume_at - now).total_seconds()
        if remaining > INLINE_WAIT_LIMIT_SECONDS:
            self._pause_run(config)
        if remaining > 0:
            self._wait_in_place(remaining, config)

        # The extra fields as received, like Pass: dumping them would turn files into iterators.
        passed_on = dict(input_data.model_extra or {})
        return passed_on | {
            "waited_until": self._resume_at.isoformat(),
            "waited_seconds": round((datetime.now(timezone.utc) - self._wait_started_at).total_seconds(), 3),
        }

    def _resolve_resume_at(self, input_data: DelayInputSchema, now: datetime) -> datetime:
        if input_data.until is not None:
            until = input_data.until
            return until if until.tzinfo else until.replace(tzinfo=timezone.utc)

        duration = input_data.duration_seconds if input_data.duration_seconds is not None else self.duration_seconds
        if duration is None:
            raise ValueError(
                f"Delay '{self.name or self.id}' has nothing to wait for: set `duration_seconds` on the node, "
                "or pass `until` or `duration_seconds` as input."
            )
        try:
            return now + timedelta(seconds=duration)
        except OverflowError as e:
            raise ValueError(f"Delay '{self.name or self.id}': {duration} seconds is too far in the future.") from e

    def _pause_run(self, config: RunnableConfig) -> None:
        """Pauses the run until the wait is over, or explains why this run cannot pause."""
        context = config.checkpoint.context if config.checkpoint else None
        if context is not None and context.pause_run(self.id, resume_at=self._resume_at):
            raise RunPausedException(
                f"Delay '{self.name or self.id}' waits until {self._resume_at.isoformat()}", resume_at=self._resume_at
            )
        raise ValueError(
            f"Delay '{self.name or self.id}' would wait until {self._resume_at.isoformat()}, longer than the "
            f"{INLINE_WAIT_LIMIT_SECONDS} seconds that can be waited in place. A longer wait pauses the run, which "
            "needs checkpointing and the Delay at the top level of the flow, not inside a Map, an agent or a "
            "sub-workflow. To test the flow without waiting, mock this node."
        )

    @staticmethod
    def _wait_in_place(seconds: float, config: RunnableConfig) -> None:
        deadline = time.monotonic() + seconds
        while (remaining := deadline - time.monotonic()) > 0:
            check_cancellation(config)
            time.sleep(min(INLINE_WAIT_POLL_SECONDS, remaining))
