import json
from enum import Enum
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import litellm
import pytest
from openai import AsyncOpenAI
from openai import OpenAI as OpenAIClient

from dynamiq.callbacks.streaming import BaseStreamingCallbackHandler
from dynamiq.connections import OpenAI as OpenAIConnection
from dynamiq.nodes.llms import _openai_sdk
from dynamiq.nodes.llms.openai import NATIVE_SDK_ENV_VAR, OpenAI
from dynamiq.nodes.node import is_retryable_error
from dynamiq.prompts import Prompt
from dynamiq.runnables import RunnableConfig, RunnableStatus
from dynamiq.types.streaming import StreamingConfig

BASE_URL = "https://api.openai.test/v1"
USAGE = {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18}


def _chunk(delta=None, finish_reason=None, usage=None):
    chunk = {
        "id": "chatcmpl-1",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "gpt-4o-mini",
        "choices": [] if delta is None else [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }
    if usage:
        chunk["usage"] = usage
    return chunk


TEXT_CHUNKS = [
    _chunk({"role": "assistant", "content": ""}),
    _chunk({"content": "Hel"}),
    _chunk({"content": "lo"}),
    _chunk({}, finish_reason="stop"),
]
TOOL_CHUNKS = [
    _chunk(
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"index": 0, "id": "call_1", "type": "function", "function": {"name": "search", "arguments": ""}}
            ],
        }
    ),
    _chunk({"tool_calls": [{"index": 0, "function": {"arguments": '{"q":'}}]}),
    _chunk({"tool_calls": [{"index": 0, "function": {"arguments": ' "x"}'}}]}),
    _chunk(
        {
            "tool_calls": [
                {"index": 1, "id": "call_2", "type": "function", "function": {"name": "calc", "arguments": "{}"}}
            ]
        }
    ),
    _chunk({}, finish_reason="tool_calls"),
]


class FakeOpenAI:
    """An OpenAI API on an httpx MockTransport: streams `chunks`, adding usage only when asked, like the API."""

    def __init__(self, chunks=TEXT_CHUNKS, status_code=200, error_message="boom"):
        self.chunks = chunks
        self.status_code = status_code
        self.error_message = error_message
        self.requests: list[dict] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        self.requests.append(body)
        if self.status_code != 200:
            return httpx.Response(self.status_code, json={"error": {"message": self.error_message, "type": "x"}})
        if not body.get("stream"):
            message = {"role": "assistant", "content": "Hello"}
            return httpx.Response(
                200,
                json={
                    "id": "chatcmpl-1",
                    "object": "chat.completion",
                    "created": 1,
                    "model": "gpt-4o-mini",
                    "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
                    "usage": USAGE,
                },
            )
        return self.stream(body, self.chunks)

    @staticmethod
    def stream(body: dict, chunks: list[dict]) -> httpx.Response:
        if (body.get("stream_options") or {}).get("include_usage"):
            chunks = chunks + [_chunk(usage=USAGE)]
        sse = b"".join(b"data: " + json.dumps(c).encode() + b"\n\n" for c in chunks) + b"data: [DONE]\n\n"
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=sse)

    def client(self) -> OpenAIClient:
        transport = httpx.MockTransport(self.handler)
        return OpenAIClient(
            api_key="k", base_url=BASE_URL, max_retries=0, http_client=httpx.Client(transport=transport)
        )

    def async_client(self) -> AsyncOpenAI:
        transport = httpx.MockTransport(self.handler)
        return AsyncOpenAI(
            api_key="k", base_url=BASE_URL, max_retries=0, http_client=httpx.AsyncClient(transport=transport)
        )


class Collector(BaseStreamingCallbackHandler):
    def __init__(self):
        self.chunks = []
        self.usage = None

    def on_node_execute_stream(self, serialized, chunk=None, **kwargs):
        self.chunks.append(chunk)

    def on_node_execute_run(self, serialized, **kwargs):
        if "usage_data" in kwargs:
            self.usage = kwargs["usage_data"]


def make_node(fake: FakeOpenAI, native: bool = True, **kwargs) -> OpenAI:
    node = OpenAI(
        model="gpt-4o-mini",
        connection=OpenAIConnection(api_key="k", url=BASE_URL),
        prompt=Prompt(messages=[{"role": "user", "content": "hi"}]),
        use_native_sdk=native,
        **kwargs,
    )
    node.client = fake.client()
    return node


@pytest.fixture
def async_sdk_client():
    """Routes the connection's async client to `fake` for the test."""

    def use(fake: FakeOpenAI):
        return patch.object(OpenAIConnection, "connect_async", AsyncMock(side_effect=lambda: fake.async_client()))

    return use


def deltas(chunks: list[dict]) -> list[str]:
    return [c["choices"][0]["delta"]["content"] or "" for c in chunks if c["choices"]]


class TestBuildRequest:
    def test_sends_only_set_sdk_params_with_the_bare_model_name(self):
        class Effort(str, Enum):
            LOW = "low"

        params = {
            "model": "openai/gpt-4o-mini",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
            "temperature": None,
            "reasoning_effort": Effort.LOW,
            "drop_params": True,
            "api_base": BASE_URL,
            "api_key": "k",
            "allowed_openai_params": ["tools"],
        }

        request = _openai_sdk.build_request(params, "openai/")

        assert request == {
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
            "reasoning_effort": "low",
        }

    @pytest.mark.parametrize(
        "extra",
        [
            {"thinking": {"type": "enabled", "budget_tokens": 1024}},
            {"num_retries": 3},
            {"response_format": MagicMock()},
            {"messages": [{"role": "user", "content": [{"type": "text", "text": "x", "cache_control": {}}]}]},
        ],
    )
    def test_leaves_what_it_does_not_translate_to_litellm(self, extra):
        params = {"model": "openai/gpt-4o-mini", "messages": [{"role": "user", "content": "hi"}]} | extra

        assert _openai_sdk.build_request(params, "openai/") is None

    def test_puts_params_an_older_sdk_lacks_in_extra_body(self):
        params = {"model": "gpt-5", "messages": [], "verbosity": "low", "extra_body": {"x": 1}}

        with patch.object(_openai_sdk, "_sdk_create_params", return_value=frozenset({"model", "messages"})):
            request = _openai_sdk.build_request(params, "openai/")

        assert request == {"model": "gpt-5", "messages": [], "extra_body": {"verbosity": "low", "x": 1}}


class TestNativeStreaming:
    @pytest.mark.asyncio
    async def test_streams_text_and_takes_usage_from_the_api(self, async_sdk_client):
        fake = FakeOpenAI()
        node = make_node(fake, streaming=StreamingConfig(enabled=True))
        collector = Collector()

        with async_sdk_client(fake):
            result = await node.run_async(input_data={}, config=RunnableConfig(callbacks=[collector]))

        assert result.status == RunnableStatus.SUCCESS, result.error
        assert result.output["content"] == "Hello"
        request = fake.requests[0]
        assert request["model"] == "gpt-4o-mini"
        assert request["stream_options"] == {"include_usage": True}
        assert not {"drop_params", "api_key", "api_base", "temperature"} & request.keys()
        # The usage chunk was asked for only to price the call, so it is not streamed, as LiteLLM does not stream it.
        assert deltas(collector.chunks) == ["", "Hel", "lo", ""]
        assert (collector.usage["prompt_tokens"], collector.usage["completion_tokens"]) == (11, 7)

    @pytest.mark.asyncio
    async def test_streams_the_usage_chunk_with_one_choice_when_asked(self, async_sdk_client):
        fake = FakeOpenAI()
        node = make_node(fake, streaming=StreamingConfig(enabled=True, include_usage=True))
        collector = Collector()

        with async_sdk_client(fake):
            result = await node.run_async(input_data={}, config=RunnableConfig(callbacks=[collector]))

        assert result.status == RunnableStatus.SUCCESS, result.error
        last = collector.chunks[-1]
        assert last["usage"]["total_tokens"] == 18
        assert len(last["choices"]) == 1 and last["choices"][0]["delta"]["content"] is None

    @pytest.mark.asyncio
    async def test_assembles_tool_calls_and_types_every_tool_call_delta(self, async_sdk_client):
        fake = FakeOpenAI(chunks=TOOL_CHUNKS)
        node = make_node(fake, streaming=StreamingConfig(enabled=True))
        collector = Collector()

        with async_sdk_client(fake):
            result = await node.run_async(input_data={}, config=RunnableConfig(callbacks=[collector]))

        assert result.status == RunnableStatus.SUCCESS, result.error
        assert result.output["content"] is None
        assert [(c["id"], c["function"]["name"], c["function"]["arguments"]) for c in result.output["tool_calls"]] == [
            ("call_1", "search", {"q": "x"}),
            ("call_2", "calc", {}),
        ]
        # The agent's function-calling parser reads only deltas typed "function"; the API types only the first.
        streamed = [tc for c in collector.chunks for tc in c["choices"][0]["delta"]["tool_calls"] or []]
        assert len(streamed) == 4 and {tc["type"] for tc in streamed} == {"function"}

    def test_sync_stream_matches_the_litellm_path(self):
        outputs, streamed = {}, {}
        for native in (False, True):
            fake = FakeOpenAI(chunks=TOOL_CHUNKS[:1] + TEXT_CHUNKS[1:3] + TOOL_CHUNKS[1:])
            node = make_node(fake, native=native, streaming=StreamingConfig(enabled=True))
            collector = Collector()
            result = node.run_sync(input_data={}, config=RunnableConfig(callbacks=[collector]))
            assert result.status == RunnableStatus.SUCCESS, result.error
            outputs[native] = result.output
            streamed[native] = [
                (deltas([c]), [(tc["index"], tc["type"], tc["function"]["arguments"]) for tc in tcs])
                for c in collector.chunks
                if c["choices"]
                for tcs in [c["choices"][0]["delta"]["tool_calls"] or []]
            ]

        assert outputs[True] == outputs[False]
        assert streamed[True] == streamed[False]

    def test_unstreamed_call(self):
        fake = FakeOpenAI()
        node = make_node(fake)
        collector = Collector()

        result = node.run_sync(input_data={}, config=RunnableConfig(callbacks=[collector]))

        assert result.status == RunnableStatus.SUCCESS, result.error
        assert result.output == {"content": "Hello"}
        assert "stream_options" not in fake.requests[0]
        assert collector.usage["total_tokens"] == 18

    def test_estimates_usage_when_the_stream_reports_none(self):
        chunks = [ChunkFactory.text("Hello"), ChunkFactory.finish()]

        response = _openai_sdk.build_response(chunks, [{"role": "user", "content": "hi"}], "openai/gpt-4o-mini")

        assert response.usage.prompt_tokens > 0 and response.usage.completion_tokens > 0


class ChunkFactory:
    @staticmethod
    def text(content):
        from openai.types.chat import ChatCompletionChunk

        return ChatCompletionChunk.construct(**_chunk({"content": content}))

    @staticmethod
    def finish():
        from openai.types.chat import ChatCompletionChunk

        return ChatCompletionChunk.construct(**_chunk({}, finish_reason="stop"))


class TestNativeErrors:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "status_code, message, expected, retryable",
        [
            (429, "Rate limit reached", litellm.RateLimitError, True),
            (400, "Unsupported parameter: 'temperature'", litellm.BadRequestError, False),
            (401, "Incorrect API key", litellm.AuthenticationError, False),
            (500, "server error", litellm.InternalServerError, True),
        ],
    )
    async def test_raises_the_litellm_exception_litellm_would(
        self, async_sdk_client, status_code, message, expected, retryable
    ):
        fake = FakeOpenAI(status_code=status_code, error_message=message)
        node = make_node(fake, streaming=StreamingConfig(enabled=True))

        with async_sdk_client(fake), pytest.raises(expected) as raised:
            await node.execute_async(input_data=MagicMock(), config=RunnableConfig(callbacks=[Collector()]))

        assert is_retryable_error(raised.value) is retryable
        assert message in str(raised.value)

    def test_recovers_from_a_rejected_param_like_the_litellm_path(self):
        fake = FakeOpenAI(status_code=400, error_message="Unsupported parameter: 'stop' is not supported")
        node = make_node(fake, stop=["END"])
        original_handler = fake.handler

        def handler(request):
            if "stop" not in json.loads(request.content):
                fake.status_code = 200
            return original_handler(request)

        node.client = OpenAIClient(
            api_key="k",
            base_url=BASE_URL,
            max_retries=0,
            http_client=httpx.Client(transport=httpx.MockTransport(handler)),
        )

        result = node.run_sync(input_data={}, config=RunnableConfig(callbacks=[]))

        assert result.status == RunnableStatus.SUCCESS, result.error
        assert [("stop" in r) for r in fake.requests] == [True, False]


class TestTransportChoice:
    def test_is_off_by_default_and_follows_the_env_var(self, monkeypatch):
        connection = OpenAIConnection(api_key="k")
        monkeypatch.delenv(NATIVE_SDK_ENV_VAR, raising=False)
        assert OpenAI(model="gpt-4o-mini", connection=connection).use_native_sdk is False

        monkeypatch.setenv(NATIVE_SDK_ENV_VAR, "true")
        assert OpenAI(model="gpt-4o-mini", connection=connection).use_native_sdk is True

    @pytest.mark.asyncio
    @pytest.mark.parametrize("native, extra", [(False, {}), (True, {"thinking_enabled": True})])
    async def test_goes_through_litellm_when_off_or_untranslatable(self, async_sdk_client, native, extra):
        fake = FakeOpenAI()
        node = make_node(fake, native=native, **extra)
        response = litellm.ModelResponse(choices=[{"message": {"role": "assistant", "content": "via litellm"}}])
        node._acompletion = AsyncMock(return_value=response)

        with async_sdk_client(fake):
            output = await node.execute_async(input_data=MagicMock(), config=RunnableConfig(callbacks=[]))

        assert output["content"] == "via litellm"
        assert fake.requests == []


class ScriptedOpenAI(FakeOpenAI):
    """Replies with one scripted assistant message per request: tool calls, as a stream when asked for one."""

    def __init__(self, turns):
        super().__init__()
        self.turns = turns

    def handler(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        self.requests.append(body)
        tool_calls = self.turns[min(len(self.requests), len(self.turns)) - 1]
        if not body.get("stream"):
            message = {"role": "assistant", "content": None, "tool_calls": tool_calls}
            choice = {"index": 0, "message": message, "finish_reason": "tool_calls"}
            return httpx.Response(
                200,
                json={
                    "id": "c",
                    "object": "chat.completion",
                    "created": 1,
                    "model": "gpt-4o-mini",
                    "choices": [choice],
                },
            )
        chunks = [_chunk({"role": "assistant", "content": None})]
        for index, call in enumerate(tool_calls):
            arguments = call["function"]["arguments"]
            head = {"index": index, "id": call["id"], "type": "function"}
            chunks.append(
                _chunk({"tool_calls": [head | {"function": {"name": call["function"]["name"], "arguments": ""}}]})
            )
            for start in range(0, len(arguments), 7):
                piece = {"index": index, "function": {"arguments": arguments[start : start + 7]}}
                chunks.append(_chunk({"tool_calls": [piece]}))
        chunks.append(_chunk({}, finish_reason="tool_calls"))
        return self.stream(body, chunks)


@pytest.mark.parametrize("streaming", [False, True])
def test_an_agent_turn_sends_the_same_requests_and_streams_the_same_events_on_both_transports(streaming):
    from typing import ClassVar

    from pydantic import BaseModel

    from dynamiq.callbacks.streaming import StreamingIteratorCallbackHandler
    from dynamiq.nodes.agents import Agent
    from dynamiq.nodes.node import Node, NodeGroup
    from dynamiq.nodes.types import InferenceMode

    class LookupInput(BaseModel):
        city: str = ""

    class Lookup(Node):
        group: NodeGroup = NodeGroup.TOOLS
        name: str = "lookup"
        description: str = "Looks a city up."
        input_schema: ClassVar[type[BaseModel]] = LookupInput

        def execute(self, input_data, config=None, **kwargs):
            return {"content": f"{input_data.city} is sunny"}

    def call(call_id, name, arguments):
        return {"id": call_id, "type": "function", "function": {"name": name, "arguments": json.dumps(arguments)}}

    turns = [
        [call("call_1", "lookup", {"thought": "look it up", "city": "Kyiv"})],
        [call("call_2", "provide_final_answer", {"thought": "done", "answer": "Kyiv is sunny"})],
    ]
    requests, events, answers = {}, {}, {}
    for native in (False, True):
        fake = ScriptedOpenAI(turns)
        agent = Agent(
            name="agent",
            llm=make_node(fake, native=native),
            tools=[Lookup()],
            inference_mode=InferenceMode.FUNCTION_CALLING,
            max_loops=3,
            streaming=StreamingConfig(enabled=streaming),
        )
        handler = StreamingIteratorCallbackHandler()
        result = agent.run(input_data={"input": "Weather in Kyiv?"}, config=RunnableConfig(callbacks=[handler]))

        assert result.status == RunnableStatus.SUCCESS, result.error
        answers[native] = result.output["content"]
        requests[native] = [{k: v for k, v in r.items() if k != "stream_options"} for r in fake.requests]
        events[native] = []
        while not handler.queue.empty():
            event = handler.queue.get_nowait()
            if event.entity_id == agent.id:
                events[native].append(json.dumps(event.data, sort_keys=True, default=str))

    assert answers[True] == answers[False] == "Kyiv is sunny"
    assert len(requests[True]) == 2
    assert requests[True] == requests[False]
    assert events[True] == events[False]
    if streaming:
        assert events[True], "the agent streamed nothing"
