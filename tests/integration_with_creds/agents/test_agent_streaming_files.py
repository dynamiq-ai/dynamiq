"""Integration tests for streaming file serialization across all three callback handlers.

Verifies that BytesIO objects produced by tools are correctly serialized (base64)
in streaming events and never leak as raw binary or SerializationIterator strings.
Uses a Python tool that generates dummy files.
"""

import asyncio
import json
import threading
from io import BytesIO
from queue import Empty, Queue

import pytest

from dynamiq import Workflow
from dynamiq.callbacks import TracingCallbackHandler
from dynamiq.callbacks.streaming import (
    AsyncStreamingIteratorCallbackHandler,
    StreamingIteratorCallbackHandler,
    StreamingQueueCallbackHandler,
)
from dynamiq.connections import OpenAI as OpenAIConnection
from dynamiq.flows import Flow
from dynamiq.nodes.agents import Agent
from dynamiq.nodes.llms import OpenAI
from dynamiq.nodes.tools.python import Python
from dynamiq.nodes.types import InferenceMode
from dynamiq.runnables import RunnableConfig
from dynamiq.types.streaming import StreamingConfig, StreamingMode
from dynamiq.utils.logger import logger

PYTHON_TOOL_CODE = """
import io

def run(input_data):
    filename = input_data.get("filename", "test.bin")
    content_str = input_data.get("content", "hello world")
    raw_bytes = content_str.encode("utf-8")

    file_obj = io.BytesIO(raw_bytes)
    file_obj.name = filename
    file_obj.content_type = "application/octet-stream"

    return {
        "content": f"Created file {filename} ({len(raw_bytes)} bytes)",
        "files": [file_obj],
        "output_data": {
            "embedded_file": file_obj,
            "nested": {"deep_file": io.BytesIO(b"nested-bytes")},
        },
    }
"""

AGENT_ROLE = (
    "You create files when asked. You have a tool that generates binary files. "
    "When the user asks to create a file, call the tool with a filename and content. "
    "Always call the tool exactly once."
)


@pytest.fixture(scope="module")
def llm():
    return OpenAI(
        connection=OpenAIConnection(),
        model="gpt-4o-mini",
        max_tokens=1000,
        temperature=0,
    )


@pytest.fixture(scope="module")
def file_tool():
    return Python(
        name="FileCreator",
        description=(
            "Creates a binary file from text content. "
            "Input: 'filename' (string, the name of the file) and 'content' (string, the text content). "
            "Returns the created file."
        ),
        code=PYTHON_TOOL_CODE,
    )


def _build_agent(llm, file_tool):
    return Agent(
        name="FileAgent",
        id="FileAgent",
        llm=llm,
        tools=[file_tool],
        role=AGENT_ROLE,
        inference_mode=InferenceMode.XML,
        streaming=StreamingConfig(enabled=True, mode=StreamingMode.ALL),
    )


def _has_raw_binary(obj) -> str | None:
    """Recursively check for BytesIO, bytes, or SerializationIterator strings. Returns path or None."""

    def _check(val, path):
        if isinstance(val, (BytesIO, bytes)):
            return f"Raw binary at {path}: {type(val).__name__}"
        if isinstance(val, str) and "SerializationIterator" in val:
            return f"SerializationIterator string at {path}: {val[:120]}"
        if isinstance(val, dict):
            for k, v in val.items():
                result = _check(v, f"{path}.{k}")
                if result:
                    return result
        if isinstance(val, (list, tuple)):
            for i, v in enumerate(val):
                result = _check(v, f"{path}[{i}]")
                if result:
                    return result
        return None

    return _check(obj, "root")


def _extract_tool_events_with_files(events: list[dict]) -> list[dict]:
    """Return tool-result dicts that contain files from StreamChunk-wrapped events."""
    results = []
    for ev in events:
        data = ev.get("data")
        if not isinstance(data, dict):
            continue
        for choice in data.get("choices") or []:
            delta = choice.get("delta") or {}
            content = delta.get("content")
            step = delta.get("step")
            if isinstance(content, dict) and step == "tool" and content.get("files"):
                results.append(content)
    return results


def _event_to_dict(event) -> dict:
    if hasattr(event, "model_dump"):
        return event.model_dump()
    if hasattr(event, "to_dict"):
        return event.to_dict()
    return {"raw": str(event)}


# ---------------------------------------------------------------------------
# Test 1: StreamingIteratorCallbackHandler (sync)
# ---------------------------------------------------------------------------
@pytest.mark.integration
def test_streaming_files_sync_iterator(llm, file_tool):
    """Sync StreamingIteratorCallbackHandler correctly serializes BytesIO in tool results."""
    agent = _build_agent(llm, file_tool)
    wf = Workflow(flow=Flow(nodes=[agent]))

    handler = StreamingIteratorCallbackHandler()
    tracer = TracingCallbackHandler()
    config = RunnableConfig(callbacks=[handler, tracer])

    events: list[dict] = []
    response_holder: list = []

    def run_workflow():
        result = wf.run(
            input_data={"input": "Create a file called hello.txt with content 'Hello World'"},
            config=config,
            is_async=False,
        )
        response_holder.append(result)

    wf_thread = threading.Thread(target=run_workflow, daemon=True)
    wf_thread.start()

    for event in handler:
        events.append(_event_to_dict(event))

    wf_thread.join(timeout=120)

    assert response_holder, "Workflow did not return a result"

    assert len(events) > 0, "Expected at least one streaming event"
    for i, ev in enumerate(events):
        binary_issue = _has_raw_binary(ev)
        assert binary_issue is None, f"event[{i}]: {binary_issue}"
        json.dumps(ev, default=str)

    tool_events = _extract_tool_events_with_files(events)
    logger.info(f"[sync] Found {len(tool_events)} tool-result events with files")
    assert len(tool_events) > 0, "Expected at least one tool-result streaming event containing files"
    for tc in tool_events:
        for f in tc["files"]:
            assert isinstance(f, dict), f"File should be a dict, got {type(f)}"
            assert "content" in f, f"Serialized file missing 'content' key: {f}"
            assert "size" in f, f"Serialized file missing 'size' key: {f}"

    assert len(tracer.runs) > 0, "Tracer should capture at least one run"
    for run_id, run in tracer.runs.items():
        json.dumps(run.to_dict(), default=str)


# ---------------------------------------------------------------------------
# Test 2: AsyncStreamingIteratorCallbackHandler (async)
# ---------------------------------------------------------------------------
@pytest.mark.integration
def test_streaming_files_async_iterator(llm, file_tool):
    """Async StreamingIteratorCallbackHandler correctly serializes BytesIO in tool results."""

    async def _run():
        agent = _build_agent(llm, file_tool)
        wf = Workflow(flow=Flow(nodes=[agent]))

        handler = AsyncStreamingIteratorCallbackHandler()
        tracer = TracingCallbackHandler()
        config = RunnableConfig(callbacks=[handler, tracer])

        events: list[dict] = []

        async def consume():
            async for event in handler:
                events.append(_event_to_dict(event))

        loop = asyncio.get_running_loop()
        consumer_task = loop.create_task(consume())
        await asyncio.sleep(0.01)

        response = await loop.run_in_executor(
            None,
            lambda: wf.run(
                input_data={"input": "Create a file called data.bin with content 'Binary Data Test'"},
                config=config,
            ),
        )

        await consumer_task
        return events, tracer, response

    events, tracer, response = asyncio.run(_run())

    assert response is not None

    assert len(events) > 0, "Expected at least one streaming event"
    for i, ev in enumerate(events):
        binary_issue = _has_raw_binary(ev)
        assert binary_issue is None, f"event[{i}]: {binary_issue}"
        json.dumps(ev, default=str)

    tool_events = _extract_tool_events_with_files(events)
    logger.info(f"[async] Found {len(tool_events)} tool-result events with files")
    assert len(tool_events) > 0, "Expected at least one tool-result streaming event containing files"
    for tc in tool_events:
        for f in tc["files"]:
            assert isinstance(f, dict), f"File should be a dict, got {type(f)}"
            assert "content" in f, f"Serialized file missing 'content' key: {f}"
            assert "size" in f, f"Serialized file missing 'size' key: {f}"

    assert len(tracer.runs) > 0, "Tracer should capture at least one run"
    for run_id, run in tracer.runs.items():
        json.dumps(run.to_dict(), default=str)


# ---------------------------------------------------------------------------
# Test 3: StreamingQueueCallbackHandler (raw queue)
# ---------------------------------------------------------------------------
@pytest.mark.integration
def test_streaming_files_queue_handler(llm, file_tool):
    """Raw StreamingQueueCallbackHandler correctly serializes BytesIO in tool results."""
    agent = _build_agent(llm, file_tool)
    wf = Workflow(flow=Flow(nodes=[agent]))

    queue = Queue()
    done_event = threading.Event()
    handler = StreamingQueueCallbackHandler(queue=queue, done_event=done_event)
    tracer = TracingCallbackHandler()
    config = RunnableConfig(callbacks=[handler, tracer])

    events: list[dict] = []
    response_holder: list = []

    def run_workflow():
        result = wf.run(
            input_data={"input": "Create a file called report.csv with content 'col1,col2\\n1,2'"},
            config=config,
            is_async=False,
        )
        response_holder.append(result)

    wf_thread = threading.Thread(target=run_workflow, daemon=True)
    wf_thread.start()

    while not done_event.is_set() or not queue.empty():
        try:
            event = queue.get(timeout=0.5)
        except Empty:
            continue
        events.append(_event_to_dict(event))

    wf_thread.join(timeout=120)

    assert response_holder, "Workflow did not return a result"

    assert len(events) > 0, "Expected at least one streaming event"
    for i, ev in enumerate(events):
        binary_issue = _has_raw_binary(ev)
        assert binary_issue is None, f"event[{i}]: {binary_issue}"
        json.dumps(ev, default=str)

    tool_events = _extract_tool_events_with_files(events)
    logger.info(f"[queue] Found {len(tool_events)} tool-result events with files")
    assert len(tool_events) > 0, "Expected at least one tool-result streaming event containing files"
    for tc in tool_events:
        for f in tc["files"]:
            assert isinstance(f, dict), f"File should be a dict, got {type(f)}"
            assert "content" in f, f"Serialized file missing 'content' key: {f}"
            assert "size" in f, f"Serialized file missing 'size' key: {f}"

    assert len(tracer.runs) > 0, "Tracer should capture at least one run"
    for run_id, run in tracer.runs.items():
        json.dumps(run.to_dict(), default=str)
