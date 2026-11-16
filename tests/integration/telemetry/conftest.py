"""Fixtures and helpers of the end-to-end span tests.

Every test injects its own provider through ``RunnableConfig.tracing``; the global provider can be set once per
process. Spans are read from an in-memory exporter and compared with the contract in
docs/reference/opentelemetry-tracing.md, never with what the implementation happens to emit.
"""

import json
from dataclasses import dataclass
from typing import Any

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from dynamiq.runnables import RunnableConfig
from dynamiq.telemetry import registry
from dynamiq.types.telemetry import TracingConfig

RUN_ID = "dynamiq.run.id"
RUN_TYPE = "dynamiq.run.type"
RUN_STATUS = "dynamiq.run.status"
NODE_NAME = "dynamiq.node.name"
OPERATION = "gen_ai.operation.name"


@pytest.fixture
def exporter():
    return InMemorySpanExporter()


@pytest.fixture
def provider(exporter):
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    yield provider
    provider.shutdown()


@pytest.fixture
def make_config(provider):
    """Builds a RunnableConfig that sends spans to the test provider; extra tracing settings go in ``tracing``."""

    def make(callbacks=None, **tracing) -> RunnableConfig:
        return RunnableConfig(
            callbacks=callbacks if callbacks is not None else [],
            tracing=TracingConfig(tracer_provider=provider, **tracing),
        )

    return make


@pytest.fixture
def config(make_config) -> RunnableConfig:
    return make_config()


@pytest.fixture(autouse=True)
def registry_is_empty_after_the_test():
    yield
    assert registry.size() == 0


@dataclass
class Trace:
    """The finished spans of one test, with tree helpers."""

    spans: list

    def named(self, name: str) -> list:
        return [span for span in self.spans if span.name == name]

    def one(self, name: str):
        found = self.named(name)
        assert len(found) == 1, f"expected one span named {name!r}, got {[s.name for s in self.spans]}"
        return found[0]

    def by_op(self, operation: str) -> list:
        return [span for span in self.spans if span.attributes.get(OPERATION) == operation]

    def by_node(self, name: str) -> list:
        return [span for span in self.spans if span.attributes.get(NODE_NAME) == name]

    def children(self, parent) -> list:
        children = [s for s in self.spans if s.parent is not None and s.parent.span_id == parent.context.span_id]
        return sorted(children, key=lambda s: s.start_time)

    def parent_of(self, span):
        if span.parent is None:
            return None
        (parent,) = (s for s in self.spans if s.context.span_id == span.parent.span_id)
        return parent

    def roots(self) -> list:
        ids = {s.context.span_id for s in self.spans}
        return [s for s in self.spans if s.parent is None or s.parent.span_id not in ids]

    def shape(self, span=None) -> list:
        """Nested ``[name, [children...]]`` of the tree below ``span`` (the root when omitted); sorted for stability
        among siblings that may start in any order."""
        if span is None:
            (span,) = self.roots()
        return [span.name, sorted((self.shape(c) for c in self.children(span)), key=repr)]


@pytest.fixture
def trace(exporter):
    def collect() -> Trace:
        return Trace(list(exporter.get_finished_spans()))

    return collect


def attrs_json(span, key: str) -> Any:
    return json.loads(span.attributes[key])


def all_strings(span) -> list[str]:
    """Every string a span holds: attribute values (also inside sequences), event names and attributes, status."""
    found: list[str] = []

    def add(value):
        if isinstance(value, str):
            found.append(value)
        elif isinstance(value, (list, tuple)):
            for item in value:
                add(item)

    for key, value in span.attributes.items():
        found.append(key)
        add(value)
    for event in span.events:
        found.append(event.name)
        for key, value in (event.attributes or {}).items():
            found.append(key)
            add(value)
    found.append(span.name)
    if span.status and span.status.description:
        found.append(span.status.description)
    return found


@pytest.fixture
def script_llm(mocker):
    """Makes every LLM node follow ``steps`` in ``mode`` through the real node code (only the provider call is
    replaced), so ``chat`` spans, usage and messages are produced by the framework. Returns the list of requests."""
    import threading

    from litellm import ModelResponse
    from litellm.types.utils import Usage
    from litellm.utils import Delta

    from tests.helpers.agent_hooks import encode

    def stream_of(text: str):
        for start in range(0, len(text), 7):
            chunk = ModelResponse(stream=True)
            chunk.choices[0].delta = Delta(role="assistant", content=text[start : start + 7])
            yield chunk

    def install(mode, steps, usage=(11, 7)):
        lock = threading.Lock()
        state = {"n": 0}
        requests: list[dict] = []

        def completion(stream=False, *args, **kwargs):
            with lock:
                index = state["n"]
                state["n"] += 1
                requests.append(kwargs)
            output = encode(mode, steps[min(index, len(steps) - 1)], index)
            if stream:
                return stream_of(output["content"])
            message = {"content": output["content"]}
            finish = "stop"
            if output.get("tool_calls"):
                message["tool_calls"] = output["tool_calls"]
                finish = "tool_calls"
            return ModelResponse(
                choices=[{"message": message, "finish_reason": finish}],
                usage=Usage(prompt_tokens=usage[0], completion_tokens=usage[1], total_tokens=sum(usage)),
            )

        mocker.patch("dynamiq.nodes.llms.base.BaseLLM._completion", side_effect=completion)
        return requests

    return install
