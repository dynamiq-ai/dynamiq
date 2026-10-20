import asyncio
from collections import Counter
from unittest.mock import patch

import pytest

from dynamiq.callbacks import BaseCallbackHandler, TracingCallbackHandler
from dynamiq.connections import OpenAI as OpenAIConnection
from dynamiq.nodes.llms.openai import OpenAI
from dynamiq.nodes.node import Node, stream_serialization_scope
from dynamiq.runnables import RunnableConfig
from dynamiq.types.streaming import StreamingConfig


class StreamRecorder(BaseCallbackHandler):
    def __init__(self):
        self.events = []

    def on_node_execute_stream(self, serialized, chunk=None, **kwargs):
        self.events.append((serialized, chunk))


class TracingStreamRecorder(TracingCallbackHandler):
    def on_node_execute_stream(self, serialized, chunk=None, **kwargs):
        self.events.append(serialized)

    @property
    def events(self):
        return self.metadata.setdefault("events", [])


@pytest.fixture
def node():
    return OpenAI(model="gpt-4o-mini", connection=OpenAIConnection(api_key="k"))


@pytest.fixture
def to_dict_calls():
    calls = Counter()
    original = Node.to_dict

    def counting(self, *args, **kwargs):
        calls[kwargs.get("for_tracing", False)] += 1
        return original(self, *args, **kwargs)

    with patch.object(Node, "to_dict", counting):
        yield calls


def test_callbacks_without_a_stream_hook_are_not_serialized_for(node, to_dict_calls):
    recorder = StreamRecorder()

    node.run_on_node_execute_stream([BaseCallbackHandler(), TracingCallbackHandler(), recorder], {"x": 1})

    assert len(recorder.events) == 1
    assert to_dict_calls == Counter({False: 1})


def test_outside_a_scope_every_event_is_serialized(node, to_dict_calls):
    recorder = StreamRecorder()

    for i in range(5):
        node.run_on_node_execute_stream([recorder], {"i": i})

    assert to_dict_calls[False] == 5


def test_a_scope_serializes_once_per_node_and_tracing_mode(node, to_dict_calls):
    recorder, tracing = StreamRecorder(), TracingStreamRecorder()

    with stream_serialization_scope():
        for i in range(5):
            node.run_on_node_execute_stream([recorder, tracing], {"i": i})

    assert to_dict_calls == Counter({False: 1, True: 1})
    assert [chunk for _, chunk in recorder.events] == [{"i": i} for i in range(5)]
    assert recorder.events[0][0] == node.to_dict()
    assert tracing.events[0] == node.to_dict(for_tracing=True)


def test_a_callback_editing_its_dict_at_any_depth_does_not_change_what_others_receive(node):
    class Editor(BaseCallbackHandler):
        def __init__(self):
            self.seen = []

        def on_node_execute_stream(self, serialized, chunk=None, **kwargs):
            self.seen.append((serialized.pop("id"), serialized["streaming"]["event"], len(serialized["depends"])))
            serialized["streaming"]["event"] = "edited"
            serialized["depends"].append({"node": "edited"})

    first, second = Editor(), Editor()

    with stream_serialization_scope():
        node.run_on_node_execute_stream([first, second], {})
        node.run_on_node_execute_stream([first, second], {})

    fresh = (node.id, node.streaming.event, len(node.depends))
    assert first.seen == second.seen == [fresh, fresh]


@pytest.mark.asyncio
async def test_concurrent_streams_do_not_share_a_scope(to_dict_calls):
    nodes = [OpenAI(model="gpt-4o-mini", connection=OpenAIConnection(api_key="k")) for _ in range(2)]
    recorder = StreamRecorder()
    in_scope = asyncio.Event()

    async def scoped():
        with stream_serialization_scope():
            nodes[0].run_on_node_execute_stream([recorder], {})
            in_scope.set()
            await asyncio.sleep(0)
            nodes[0].run_on_node_execute_stream([recorder], {})

    async def unscoped():
        await in_scope.wait()
        nodes[1].run_on_node_execute_stream([recorder], {})
        nodes[1].run_on_node_execute_stream([recorder], {})

    await asyncio.gather(scoped(), unscoped())

    assert to_dict_calls[False] == 1 + 2
    assert [s["id"] for s, _ in recorder.events].count(nodes[1].id) == 2


@pytest.mark.parametrize("n_chunks", [3, 30])
def test_an_llm_stream_serializes_the_node_independently_of_its_length(node, to_dict_calls, n_chunks):
    class Chunk:
        def __init__(self, i):
            self.i = i

        def model_dump(self):
            return {"choices": [{"delta": {"content": str(self.i)}}]}

    node.streaming = StreamingConfig(enabled=True)
    node._stream_chunk_builder = lambda chunks, messages: None
    node._handle_completion_response = lambda response, config, **kwargs: {"content": "ok"}
    recorder = StreamRecorder()

    node._handle_streaming_completion_response(
        [Chunk(i) for i in range(n_chunks)], [], RunnableConfig(callbacks=[recorder, TracingCallbackHandler()])
    )

    assert len(recorder.events) == n_chunks
    assert to_dict_calls == Counter({False: 1})
