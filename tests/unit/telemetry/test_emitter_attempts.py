"""Data an execution attempt reports reaches the span only while that attempt is the one in progress.

A node with a timeout runs each attempt in a thread, and an attempt that times out is abandoned, not stopped: it can
report long after the retry took over, or after the span has ended.
"""

import asyncio
import threading
from typing import Any

import pytest
from pydantic import PrivateAttr

from dynamiq.callbacks import BaseCallbackHandler
from dynamiq.nodes import Node, NodeGroup
from dynamiq.nodes.node import ErrorHandling
from dynamiq.runnables import RunnableConfig, RunnableStatus
from dynamiq.telemetry import semconv
from dynamiq.types.telemetry import TracingConfig
from tests.unit.telemetry.conftest import span_named, status_of

sdk_trace = pytest.importorskip("opentelemetry.sdk.trace")

pytestmark = pytest.mark.usefixtures("no_span_leaks")

ATTEMPT = "dynamiq.execution.attempt"
WAIT_SECONDS = 5
FIRST, RETRY = 111, 222


def report(node: Node, hook: str, value: int, config: RunnableConfig, kwargs: dict[str, Any]) -> None:
    """Report ``value`` through one of the per-attempt hooks, as node code does from inside ``execute``."""
    if hook == "run":
        node.run_on_node_execute_run(config.callbacks, usage_data={"total_tokens": value}, **kwargs)
    elif hook == "end":
        node.run_on_node_execute_end(config.callbacks, {"value": value}, **kwargs)
    else:
        node.run_on_node_execute_error(config.callbacks, ValueError(f"late {value}"), **kwargs)


class OutlivedByItsRetry(Node):
    """Its first attempt outlives the timeout and reports through ``late_hook`` once the retry has reported."""

    group: NodeGroup = NodeGroup.UTILS
    name: str = "outlived"
    late_hook: str = "run"
    _calls: int = PrivateAttr(default=0)
    _retry_reported: threading.Event = PrivateAttr(default_factory=threading.Event)
    _late_reported: threading.Event = PrivateAttr(default_factory=threading.Event)

    def execute(self, input_data: dict[str, Any], config: RunnableConfig = None, **kwargs):
        self._calls += 1
        if self._calls == 1:
            return self._first_attempt(config, kwargs)
        report(self, "run", RETRY, config, kwargs)
        self._retry_reported.set()
        # The retry finishes only once the abandoned attempt has reported, so that its report lands on an open span.
        assert self._late_reported.wait(WAIT_SECONDS)
        return {"value": RETRY}

    def _first_attempt(self, config: RunnableConfig, kwargs: dict[str, Any]) -> dict[str, Any]:
        try:
            assert self._retry_reported.wait(WAIT_SECONDS)
            report(self, self.late_hook, FIRST, config, kwargs)
            return {"value": FIRST}
        finally:
            self._late_reported.set()


class AsyncOutlivedByItsRetry(OutlivedByItsRetry):
    """The same, with a native async execute whose first attempt works in a thread that its timeout leaves running."""

    name: str = "async-outlived"

    async def execute_async(self, input_data: dict[str, Any], config: RunnableConfig = None, **kwargs):
        self._calls += 1
        if self._calls == 1:
            return await asyncio.to_thread(self._first_attempt, config, kwargs)
        report(self, "run", RETRY, config, kwargs)
        self._retry_reported.set()
        await asyncio.to_thread(self._late_reported.wait, WAIT_SECONDS)
        return {"value": RETRY}


class ReportingBetweenAttempts(Node):
    """Its first attempt reports after its timeout was reported and before the retry reports."""

    group: NodeGroup = NodeGroup.UTILS
    name: str = "between"
    _calls: int = PrivateAttr(default=0)
    _timed_out: threading.Event = PrivateAttr(default_factory=threading.Event)
    _late_reported: threading.Event = PrivateAttr(default_factory=threading.Event)

    def execute(self, input_data: dict[str, Any], config: RunnableConfig = None, **kwargs):
        self._calls += 1
        if self._calls == 1:
            try:
                assert self._timed_out.wait(WAIT_SECONDS)
                report(self, "run", FIRST, config, kwargs)
                return {"value": FIRST}
            finally:
                self._late_reported.set()
        assert self._late_reported.wait(WAIT_SECONDS)
        report(self, "run", RETRY, config, kwargs)
        return {"value": RETRY}


class OutlivedByItsRun(Node):
    """Its only attempt outlives the timeout and reports through ``late_hook`` after the run has returned."""

    group: NodeGroup = NodeGroup.UTILS
    name: str = "outlived-run"
    late_hook: str = "run"
    _run_returned: threading.Event = PrivateAttr(default_factory=threading.Event)
    _late_reported: threading.Event = PrivateAttr(default_factory=threading.Event)

    def execute(self, input_data: dict[str, Any], config: RunnableConfig = None, **kwargs):
        try:
            assert self._run_returned.wait(WAIT_SECONDS)
            report(self, self.late_hook, FIRST, config, kwargs)
            return {"value": FIRST}
        finally:
            self._late_reported.set()


class TimeoutSignal(BaseCallbackHandler):
    """Tells a node that the timeout of its attempt has been reported."""

    def __init__(self, node: ReportingBetweenAttempts):
        self.node = node

    def on_node_execute_error(self, serialized, error, **kwargs):
        self.node._timed_out.set()


@pytest.fixture
def reported(monkeypatch):
    """Makes every report show its value, under a key of its own so that an overwritten one still shows."""

    def execute_run_attributes(span_type, entity, *, run_kwargs, settings, state):
        tokens = run_kwargs["usage_data"]["total_tokens"]
        return {"test.total_tokens": tokens, f"test.reported.{tokens}": True}

    def execute_output_attributes(span_type, entity, output_data, *, settings, state):
        return {"test.output": output_data["value"]}

    monkeypatch.setattr(semconv, "execute_run_attributes", execute_run_attributes)
    monkeypatch.setattr(semconv, "execute_output_attributes", execute_output_attributes)


def retried_after_timeout(seconds: float = 0.5) -> ErrorHandling:
    return ErrorHandling(timeout_seconds=seconds, max_retries=1, retry_interval_seconds=0.01, backoff_rate=1)


def execution_attempts(span) -> list[int]:
    return [event.attributes[ATTEMPT] for event in span.events if event.name == semconv.EXECUTION_EVENT_NAME]


def assert_only_the_retry_reported(span) -> None:
    assert status_of(span) == "succeeded"
    assert span.attributes["test.total_tokens"] == RETRY
    assert f"test.reported.{FIRST}" not in span.attributes
    assert span.attributes["test.output"] == RETRY
    # The timeout of the first attempt and the success of the retry, nothing the abandoned attempt reported.
    assert execution_attempts(span) == [1, 2]


@pytest.mark.parametrize("late_hook", ["run", "end", "error"])
def test_abandoned_attempt_reporting_during_the_retry_leaves_the_span_to_the_retry(
    exporter, traced, reported, late_hook
):
    node = OutlivedByItsRetry(late_hook=late_hook, error_handling=retried_after_timeout())

    result = node.run(input_data={}, config=traced)

    assert result.status == RunnableStatus.SUCCESS
    assert result.output == {"value": RETRY}
    assert node._late_reported.is_set()
    assert_only_the_retry_reported(span_named(exporter, "outlived"))


@pytest.mark.asyncio
async def test_abandoned_async_attempt_reporting_during_the_retry_leaves_the_span_to_the_retry(
    exporter, traced, reported
):
    node = AsyncOutlivedByItsRetry(error_handling=retried_after_timeout())

    result = await node.run_async(input_data={}, config=traced)

    assert result.status == RunnableStatus.SUCCESS
    assert node._late_reported.is_set()
    assert_only_the_retry_reported(span_named(exporter, "async-outlived"))


def test_abandoned_attempt_reporting_before_the_retry_starts_reaches_no_span(exporter, provider, reported):
    node = ReportingBetweenAttempts(error_handling=retried_after_timeout(0.3))
    config = RunnableConfig(callbacks=[TimeoutSignal(node)], tracing=TracingConfig(tracer_provider=provider))

    result = node.run(input_data={}, config=config)

    assert result.status == RunnableStatus.SUCCESS
    assert_only_the_retry_reported(span_named(exporter, "between"))


@pytest.mark.parametrize("late_hook", ["run", "end", "error"])
def test_abandoned_attempt_reporting_after_its_span_ended_is_ignored(exporter, traced, reported, caplog, late_hook):
    node = OutlivedByItsRun(late_hook=late_hook, error_handling=ErrorHandling(timeout_seconds=0.05))

    result = node.run(input_data={}, config=traced)
    caplog.clear()
    node._run_returned.set()
    assert node._late_reported.wait(WAIT_SECONDS)

    assert result.status == RunnableStatus.FAILURE
    (span,) = exporter.get_finished_spans()
    assert status_of(span) == "failed"
    assert "test.total_tokens" not in span.attributes
    assert "test.output" not in span.attributes
    # The timeout of the attempt, reported by the run before it ended the span.
    assert execution_attempts(span) == [1]
    assert [record for record in caplog.records if "Telemetry" in record.getMessage()] == []
