"""An approval gate nobody can answer must not open itself.

An empty answer means "approved" (``accept_pattern`` defaults to ``""``), so a prompt
that could not be read at all must be reported as a refusal rather than as an empty
answer. Without stdin — any container, service, or CI job — ``input()`` raises, and the
two cases used to be indistinguishable.
"""

import builtins
from typing import ClassVar

import pytest
from pydantic import BaseModel

from dynamiq.nodes.node import Node, NodeGroup
from dynamiq.runnables import RunnableConfig, RunnableStatus
from dynamiq.types.feedback import ApprovalConfig, FeedbackMethod


class _Input(BaseModel):
    value: str = "x"


class GuardedTool(Node):
    group: NodeGroup = NodeGroup.TOOLS
    name: str = "guarded"
    description: str = "records whether it ran"
    input_schema: ClassVar[type[BaseModel]] = _Input

    def execute(self, input_data, config: RunnableConfig = None, **kwargs):
        _RAN.append(input_data.value)
        return {"content": "executed"}


_RAN: list[str] = []


@pytest.fixture(autouse=True)
def clear_ran():
    _RAN.clear()
    yield
    _RAN.clear()


@pytest.fixture
def no_stdin(monkeypatch):
    """Simulate a process with no readable stdin."""

    def _raise(*args, **kwargs):
        raise EOFError("EOF when reading a line")

    monkeypatch.setattr(builtins, "input", _raise)


class TestUnanswerableApproval:
    def test_denies_when_prompt_cannot_be_read(self, no_stdin):
        tool = GuardedTool(approval=ApprovalConfig(enabled=True, feedback_method=FeedbackMethod.CONSOLE))

        result = tool.run(input_data={"value": "x"})

        assert result.status == RunnableStatus.SKIP
        assert _RAN == [], "the guarded body must not run"

    def test_denial_is_recoverable(self, no_stdin):
        """The agent should be able to react, not have the whole run torn down."""
        tool = GuardedTool(approval=ApprovalConfig(enabled=True, feedback_method=FeedbackMethod.CONSOLE))

        result = tool.run(input_data={"value": "x"})

        assert result.error.recoverable is True

    def test_denies_regardless_of_accept_pattern(self, no_stdin):
        tool = GuardedTool(
            approval=ApprovalConfig(enabled=True, feedback_method=FeedbackMethod.CONSOLE, accept_pattern="yes")
        )

        result = tool.run(input_data={"value": "x"})

        assert result.status == RunnableStatus.SKIP
        assert _RAN == []

    def test_opt_out_restores_previous_behavior(self, no_stdin):
        """Deployments that depended on fail-open have an explicit escape hatch."""
        tool = GuardedTool(
            approval=ApprovalConfig(
                enabled=True, feedback_method=FeedbackMethod.CONSOLE, approve_when_unanswerable=True
            )
        )

        result = tool.run(input_data={"value": "x"})

        assert result.status == RunnableStatus.SUCCESS
        assert _RAN == ["x"]

    def test_closed_stdin_is_also_a_denial(self, monkeypatch):
        """OSError, not just EOFError — a closed rather than absent stdin."""

        def _raise(*args, **kwargs):
            raise OSError("Bad file descriptor")

        monkeypatch.setattr(builtins, "input", _raise)
        tool = GuardedTool(approval=ApprovalConfig(enabled=True, feedback_method=FeedbackMethod.CONSOLE))

        assert tool.run(input_data={"value": "x"}).status == RunnableStatus.SKIP
        assert _RAN == []


class TestAnsweredApprovalIsUnchanged:
    def test_empty_answer_still_approves(self, monkeypatch):
        """A real user pressing enter is an approval and must stay one."""
        monkeypatch.setattr(builtins, "input", lambda *a, **k: "")
        tool = GuardedTool(approval=ApprovalConfig(enabled=True, feedback_method=FeedbackMethod.CONSOLE))

        result = tool.run(input_data={"value": "x"})

        assert result.status == RunnableStatus.SUCCESS
        assert _RAN == ["x"]

    def test_feedback_still_cancels(self, monkeypatch):
        monkeypatch.setattr(builtins, "input", lambda *a, **k: "no, stop")
        tool = GuardedTool(approval=ApprovalConfig(enabled=True, feedback_method=FeedbackMethod.CONSOLE))

        result = tool.run(input_data={"value": "x"})

        assert result.status == RunnableStatus.SKIP
        assert _RAN == []

    def test_matching_accept_pattern_still_approves(self, monkeypatch):
        monkeypatch.setattr(builtins, "input", lambda *a, **k: "yes")
        tool = GuardedTool(
            approval=ApprovalConfig(enabled=True, feedback_method=FeedbackMethod.CONSOLE, accept_pattern="yes")
        )

        assert tool.run(input_data={"value": "x"}).status == RunnableStatus.SUCCESS
        assert _RAN == ["x"]

    def test_approval_disabled_is_untouched(self, no_stdin):
        """With approval off, nothing prompts and nothing is denied."""
        tool = GuardedTool()

        assert tool.run(input_data={"value": "x"}).status == RunnableStatus.SUCCESS
        assert _RAN == ["x"]


class TestApprovalConfigDefaults:
    def test_fails_closed_by_default(self):
        assert ApprovalConfig().approve_when_unanswerable is False
