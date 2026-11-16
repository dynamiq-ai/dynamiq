"""Fixtures and test nodes of the span lifecycle tests.

Nothing here imports the OpenTelemetry SDK at module level: test_api_only.py runs in the default install, which
has the API only.
"""

import asyncio
import time
from typing import Any, Callable

import pytest
from opentelemetry import context as context_api
from pydantic import PrivateAttr

from dynamiq.checkpoints.types import RunPausedException
from dynamiq.nodes import Node, NodeGroup
from dynamiq.runnables import RunnableConfig
from dynamiq.telemetry import registry
from dynamiq.types.cancellation import check_cancellation
from dynamiq.types.telemetry import TracingConfig

RUN_ID = "dynamiq.run.id"
RUN_TYPE = "dynamiq.run.type"
RUN_STATUS = "dynamiq.run.status"


class Echo(Node):
    """Returns the value it is given."""

    group: NodeGroup = NodeGroup.UTILS
    name: str = "echo"

    def execute(self, input_data: dict[str, Any], config: RunnableConfig = None, **kwargs):
        return {"value": input_data.get("value")}


class AsyncEcho(Echo):
    """Echo with a native async execute."""

    name: str = "async-echo"

    async def execute_async(self, input_data: dict[str, Any], config: RunnableConfig = None, **kwargs):
        await asyncio.sleep(0)
        return {"value": input_data.get("value")}


class Failing(Node):
    group: NodeGroup = NodeGroup.UTILS
    name: str = "failing"

    def execute(self, input_data: dict[str, Any], config: RunnableConfig = None, **kwargs):
        raise ValueError("boom")


class Flaky(Node):
    """Fails ``fail_times`` times, then succeeds."""

    group: NodeGroup = NodeGroup.UTILS
    name: str = "flaky"
    fail_times: int = 1
    _attempts: int = PrivateAttr(default=0)

    def execute(self, input_data: dict[str, Any], config: RunnableConfig = None, **kwargs):
        self._attempts += 1
        if self._attempts <= self.fail_times:
            raise ValueError(f"attempt {self._attempts} failed")
        return {"attempts": self._attempts}

    async def execute_async(self, input_data: dict[str, Any], config: RunnableConfig = None, **kwargs):
        return self.execute(input_data, config, **kwargs)


class Waiting(Node):
    """Waits until it is canceled or ``seconds`` have passed."""

    group: NodeGroup = NodeGroup.UTILS
    name: str = "waiting"
    seconds: float = 10

    def execute(self, input_data: dict[str, Any], config: RunnableConfig = None, **kwargs):
        deadline = time.monotonic() + self.seconds
        while time.monotonic() < deadline:
            check_cancellation(config)
            time.sleep(0.005)
        return {"waited": True}


class AsyncWaiting(Waiting):
    name: str = "async-waiting"

    async def execute_async(self, input_data: dict[str, Any], config: RunnableConfig = None, **kwargs):
        await asyncio.sleep(self.seconds)
        return {"waited": True}


class Pausing(Node):
    """Pauses the run through the flow's checkpoint context until the test sets ``_pause`` to False."""

    group: NodeGroup = NodeGroup.UTILS
    name: str = "pausing"
    _pause: bool = PrivateAttr(default=True)

    def execute(self, input_data: dict[str, Any], config: RunnableConfig = None, **kwargs):
        context = config.checkpoint.context if config.checkpoint else None
        if self._pause and context is not None and context.pause_run(self.id):
            raise RunPausedException("waiting for the test")
        return {"resumed": True}


class Instrumented(Node):
    """Starts a span of its own while it executes, as an instrumented HTTP client or database driver would."""

    group: NodeGroup = NodeGroup.UTILS
    name: str = "instrumented"
    _tracer: Any = PrivateAttr(default=None)

    def execute(self, input_data: dict[str, Any], config: RunnableConfig = None, **kwargs):
        with self._tracer.start_as_current_span("inner"):
            return {"done": True}


class AsyncInstrumented(Instrumented):
    name: str = "async-instrumented"

    async def execute_async(self, input_data: dict[str, Any], config: RunnableConfig = None, **kwargs):
        await asyncio.sleep(0)
        return self.execute(input_data, config, **kwargs)


class Calling(Node):
    """Runs another node while it executes, naming itself as the parent run or leaving the parent out."""

    group: NodeGroup = NodeGroup.UTILS
    name: str = "calling"
    inner: Node
    inner_input: dict[str, Any] = {"value": 1}
    parent_run_id_passed: bool = True

    def execute(self, input_data: dict[str, Any], config: RunnableConfig = None, **kwargs):
        run_kwargs = {"parent_run_id": kwargs["run_id"]} if self.parent_run_id_passed else {}
        return self.inner.run(input_data=self.inner_input, config=config, **run_kwargs).output


class Prompting(Node):
    """Reports the prompt in its input while it executes, as LLM nodes do.

    Its first ``fail_times`` attempts fail after they reported, so that a retry reports the prompt again;
    ``_before_failing`` runs between the report and the failure.
    """

    group: NodeGroup = NodeGroup.LLMS
    name: str = "llm"
    fail_times: int = 0
    _attempts: int = PrivateAttr(default=0)
    _before_failing: Callable[[], Any] | None = PrivateAttr(default=None)

    def execute(self, input_data: dict[str, Any], config: RunnableConfig = None, **kwargs):
        self._attempts += 1
        self.run_on_node_execute_run(config.callbacks, prompt_messages=input_data["messages"], **kwargs)
        if self._attempts <= self.fail_times:
            if self._before_failing is not None:
                self._before_failing()
            raise ValueError(f"attempt {self._attempts} failed")
        return {"content": "ok"}


class Parent(Node):
    """Calls ``_script(config, run_id)`` while it executes, for a test to run nodes as the children of its run,
    as an agent runs its LLM once per loop."""

    group: NodeGroup = NodeGroup.UTILS
    name: str = "parent"
    _script: Callable[[RunnableConfig, Any], Any] = PrivateAttr(default=None)

    def execute(self, input_data: dict[str, Any], config: RunnableConfig = None, **kwargs):
        self._script(config, kwargs["run_id"])
        return {}


@pytest.fixture
def exporter():
    # Imported here, as in `provider`: the SDK is an extra, missing where test_api_only.py loads this file.
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    return InMemorySpanExporter()


@pytest.fixture
def provider(exporter):
    """An SDK provider for one test. The global provider can be set once per process, so tests inject theirs."""
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor

    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    yield provider
    provider.shutdown()


@pytest.fixture
def traced(provider) -> RunnableConfig:
    return RunnableConfig(tracing=TracingConfig(tracer_provider=provider))


@pytest.fixture
def no_span_leaks():
    """Fails a test that leaves a span registered or a context attached."""
    context_before = context_api.get_current()
    assert registry.size() == 0
    yield
    assert registry.size() == 0
    assert context_api.get_current() is context_before


def span_named(exporter, name: str):
    """The one finished span with this name."""
    (span,) = (span for span in exporter.get_finished_spans() if span.name == name)
    return span


def spans_named(exporter, name: str) -> list:
    return [span for span in exporter.get_finished_spans() if span.name == name]


def parent_span_id(span) -> int | None:
    return span.parent.span_id if span.parent is not None else None


def status_of(span) -> str | None:
    return span.attributes.get(RUN_STATUS)
