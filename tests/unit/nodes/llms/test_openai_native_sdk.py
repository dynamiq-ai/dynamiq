import copy
import json
from enum import Enum
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import litellm
import pytest
from litellm.completion_extras import responses_api_bridge
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
OPENAI_URL = "https://api.openai.com/v1"
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

    def __init__(self, chunks=TEXT_CHUNKS, status_code=200, error_message="boom", usage=USAGE):
        self.chunks = chunks
        self.status_code = status_code
        self.error_message = error_message
        self.usage = usage
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
                    "usage": self.usage,
                },
            )
        return self.stream(body, self.chunks)

    def stream(self, body: dict, chunks: list[dict]) -> httpx.Response:
        if (body.get("stream_options") or {}).get("include_usage"):
            chunks = chunks + [_chunk(usage=self.usage)]
        sse = b"".join(b"data: " + json.dumps(c).encode() + b"\n\n" for c in chunks) + b"data: [DONE]\n\n"
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=sse)

    def client(self, base_url: str = BASE_URL) -> OpenAIClient:
        transport = httpx.MockTransport(self.handler)
        return OpenAIClient(
            api_key="k", base_url=base_url, max_retries=0, http_client=httpx.Client(transport=transport)
        )

    def async_client(self, base_url: str = BASE_URL) -> AsyncOpenAI:
        transport = httpx.MockTransport(self.handler)
        return AsyncOpenAI(
            api_key="k", base_url=base_url, max_retries=0, http_client=httpx.AsyncClient(transport=transport)
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


def make_node(fake: FakeOpenAI, native: bool = True, model: str = "gpt-4o-mini", **kwargs) -> OpenAI:
    node = OpenAI(
        model=model,
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
            "model": "openai/o3-mini",
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
            "model": "o3-mini",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
            "reasoning_effort": "low",
        }

    @pytest.mark.parametrize(
        "model, kept",
        [
            ("gpt-4o-mini", {"temperature", "top_p", "presence_penalty", "frequency_penalty", "logprobs", "seed"}),
            ("gpt-5-mini", {"seed", "reasoning_effort"}),
            ("o3-mini", {"seed", "reasoning_effort"}),
        ],
    )
    def test_drops_the_params_the_model_rejects_as_litellm_does(self, model, kept):
        sampling = {"temperature": 0.3, "top_p": 0.9, "presence_penalty": 0.5, "frequency_penalty": 0.2}
        params = {"model": model, "messages": [], "logprobs": True, "seed": 7, "reasoning_effort": "low"} | sampling

        request = _openai_sdk.build_request(params, "openai/")

        assert set(request) - {"model", "messages"} == kept

    def test_keeps_request_options_out_of_the_body(self):
        params = {"model": "gpt-4o-mini", "messages": [], "timeout": 30, "extra_headers": {"X-Trace": "1"}}

        request = _openai_sdk.build_request(params, "openai/")

        assert request == {"model": "gpt-4o-mini", "messages": [], "timeout": 30, "extra_headers": {"X-Trace": "1"}}

    @pytest.mark.parametrize(
        "extra",
        [
            {"thinking": {"type": "enabled", "budget_tokens": 1024}},
            {"num_retries": 3},
            {"response_format": MagicMock()},
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

    TOOLS = [{"type": "function", "function": {"name": "f", "parameters": {"type": "object", "properties": {}}}}]

    @pytest.mark.parametrize(
        "model, extra, endpoint, responses",
        [
            ("gpt-5-pro", {}, BASE_URL, True),
            ("o3-pro", {}, BASE_URL, True),
            ("gpt-5.1-codex", {}, BASE_URL, True),
            ("responses/gpt-5.5", {}, BASE_URL, True),
            ("gpt-5.5", {"tools": TOOLS, "reasoning_effort": "low"}, BASE_URL, True),
            # OpenAI's API reasons by default; LiteLLM assumes a compatible server does not.
            ("gpt-5.5", {"tools": TOOLS}, OPENAI_URL, True),
            ("gpt-5.5", {"tools": TOOLS}, BASE_URL, False),
            ("gpt-5.5", {"tools": TOOLS, "reasoning_effort": "none"}, OPENAI_URL, False),
            ("gpt-5-mini", {"tools": TOOLS, "reasoning_effort": "low"}, OPENAI_URL, False),
            ("gpt-4o-mini", {"tools": TOOLS}, OPENAI_URL, False),
        ],
    )
    def test_leaves_what_litellm_sends_to_the_responses_api_to_litellm(self, model, extra, endpoint, responses):
        def params():
            messages = [{"role": "user", "content": "hi"}]
            return {"model": f"openai/{model}", "messages": messages, "api_base": endpoint, "api_key": "k"} | extra

        with patch.object(responses_api_bridge, "completion") as bridge:
            litellm.completion(**params(), client=FakeOpenAI().client(endpoint))

        assert bridge.called is responses
        assert (_openai_sdk.build_request(params(), "openai/") is None) is responses


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

    @pytest.mark.asyncio
    async def test_raises_the_litellm_exception_for_a_transport_error_building_the_body(self):
        fake = FakeOpenAI()
        request = _openai_sdk.build_request({"model": "gpt-4o-mini", "messages": []}, "openai/")
        config = MagicMock()
        # The transform fetches a PDF a file part links to.
        failure = httpx.ConnectError("refused", request=httpx.Request("GET", "https://example.test/a.pdf"))
        config.transform_request.side_effect = config.async_transform_request.side_effect = failure

        with patch.object(_openai_sdk, "_litellm_config", return_value=config):
            with pytest.raises(litellm.APIConnectionError):
                _openai_sdk.create(fake.client(), request)
            with pytest.raises(litellm.APIConnectionError):
                await _openai_sdk.acreate(fake.async_client(), request)

        assert fake.requests == []

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
    @pytest.mark.parametrize("env, native", [(None, False), ("true", True)])
    def test_an_unset_flag_follows_the_env_var_when_the_request_is_sent(self, monkeypatch, env, native):
        monkeypatch.setenv(NATIVE_SDK_ENV_VAR, "false" if native else "true")
        node = make_node(FakeOpenAI(), native=None)
        dumped = node.to_dict()
        if env is None:
            monkeypatch.delenv(NATIVE_SDK_ENV_VAR)
        else:
            monkeypatch.setenv(NATIVE_SDK_ENV_VAR, env)

        with patch.object(_openai_sdk, "create", wraps=_openai_sdk.create) as create:
            result = node.run_sync(input_data={}, config=RunnableConfig(callbacks=[]))

        assert result.status == RunnableStatus.SUCCESS, result.error
        assert create.called is native
        # A dumped workflow keeps following the environment it is loaded in.
        assert dumped["use_native_sdk"] is None

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "native, extra", [(False, {}), (True, {"thinking_enabled": True}), (True, {"model": "gpt-5-pro"})]
    )
    async def test_goes_through_litellm_when_off_or_untranslatable(self, async_sdk_client, native, extra):
        fake = FakeOpenAI()
        node = make_node(fake, native=native, **extra)
        response = litellm.ModelResponse(choices=[{"message": {"role": "assistant", "content": "via litellm"}}])
        node._acompletion = AsyncMock(return_value=response)

        with async_sdk_client(fake):
            output = await node.execute_async(input_data=MagicMock(), config=RunnableConfig(callbacks=[]))

        assert output["content"] == "via litellm"
        assert fake.requests == []

    @pytest.mark.asyncio
    @pytest.mark.parametrize("internal", ["_litellm_model_params", "_litellm_config"])
    @pytest.mark.parametrize("streaming", [False, True])
    async def test_goes_through_litellm_when_a_litellm_internal_it_calls_fails(
        self, async_sdk_client, internal, streaming
    ):
        fake = FakeOpenAI()
        node = make_node(fake, streaming=StreamingConfig(enabled=streaming))
        response = litellm.ModelResponse(choices=[{"message": {"role": "assistant", "content": "via litellm"}}])
        node._completion = MagicMock(return_value=response)
        node._acompletion = AsyncMock(return_value=response)
        # What a LiteLLM upgrade that changes a signature raises.
        failure = TypeError("got an unexpected keyword argument")

        with async_sdk_client(fake), patch.object(_openai_sdk, internal, side_effect=failure):
            sync_result = node.run_sync(input_data={}, config=RunnableConfig(callbacks=[]))
            async_result = await node.run_async(input_data={}, config=RunnableConfig(callbacks=[]))

        for result in (sync_result, async_result):
            assert result.status == RunnableStatus.SUCCESS, result.error
        assert node._completion.called and node._acompletion.called
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


class TestCostReporting:
    """Cost is computed by the node from the reported usage, with LiteLLM's price table, on either transport."""

    BILLED = {
        "prompt_tokens": 1800,
        "completion_tokens": 900,
        "total_tokens": 2700,
        "prompt_tokens_details": {"cached_tokens": 1536},
        "completion_tokens_details": {"reasoning_tokens": 800},
    }

    def usage_data(self, native: bool, streaming: StreamingConfig) -> dict:
        fake = FakeOpenAI(usage=self.BILLED)
        node = make_node(fake, native=native, model="gpt-5-mini", streaming=streaming)
        collector = Collector()
        result = node.run_sync(input_data={}, config=RunnableConfig(callbacks=[collector]))
        assert result.status == RunnableStatus.SUCCESS, result.error
        return collector.usage

    @pytest.mark.parametrize(
        "streaming",
        [StreamingConfig(enabled=False), StreamingConfig(enabled=True, include_usage=True)],
        ids=["unstreamed", "streamed-with-usage"],
    )
    def test_matches_the_litellm_path_whenever_it_has_the_api_usage(self, streaming):
        assert self.usage_data(native=True, streaming=streaming) == self.usage_data(native=False, streaming=streaming)

    def test_a_default_stream_is_priced_from_what_the_api_billed(self):
        usage = self.usage_data(native=True, streaming=StreamingConfig(enabled=True))

        prompt_cost, completion_cost = litellm.cost_per_token(
            model="gpt-5-mini", prompt_tokens=1800, completion_tokens=900, cache_read_input_tokens=1536
        )
        assert (usage["prompt_tokens"], usage["completion_tokens"], usage["cache_read_input_tokens"]) == (
            1800,
            900,
            1536,
        )
        assert usage["total_tokens_cost_usd"] == pytest.approx(prompt_cost + completion_cost)
        assert usage["total_tokens_cost_usd"] > 0


class TestRequestShapeFallbacks:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("override", [{"api_base": "https://other.test/v1"}, {"api_key": "other-key"}])
    async def test_a_per_call_endpoint_or_key_goes_through_litellm(self, async_sdk_client, override):
        fake = FakeOpenAI()
        node = make_node(fake, **override)
        node._acompletion = AsyncMock(
            return_value=litellm.ModelResponse(choices=[{"message": {"role": "assistant", "content": "ok"}}])
        )

        with async_sdk_client(fake):
            await node.execute_async(input_data=MagicMock(), config=RunnableConfig(callbacks=[]))

        assert fake.requests == []
        ((key, value),) = override.items()
        assert node._acompletion.call_args.kwargs[key] == value

    class RejectsStreamOptions(FakeOpenAI):
        def handler(self, request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            if "stream_options" not in body:
                return super().handler(request)
            self.requests.append(body)
            error = {
                "message": "Unrecognized request argument supplied: stream_options",
                "type": "invalid_request_error",
            }
            return httpx.Response(400, json={"error": error})

    @pytest.mark.asyncio
    async def test_an_endpoint_rejecting_stream_options_gets_streams_as_asked(self, async_sdk_client, monkeypatch):
        monkeypatch.setattr(_openai_sdk, "_WITHOUT_STREAM_OPTIONS", set())
        fake = self.RejectsStreamOptions()
        collector = Collector()

        with async_sdk_client(fake):
            for model in ("gpt-4o-mini", "gpt-4o-mini", "gpt-4.1"):
                node = make_node(fake, model=model, streaming=StreamingConfig(enabled=True))
                result = await node.run_async(input_data={}, config=RunnableConfig(callbacks=[collector]))
                assert result.status == RunnableStatus.SUCCESS, result.error
                assert result.output["content"] == "Hello"

        # Rejected once, sent again as asked, and remembered for the model's later streams on the endpoint.
        assert [("stream_options" in r, r["model"]) for r in fake.requests] == [
            (True, "gpt-4o-mini"),
            (False, "gpt-4o-mini"),
            (False, "gpt-4o-mini"),
            (True, "gpt-4.1"),
            (False, "gpt-4.1"),
        ]
        assert collector.usage["prompt_tokens"] > 0 and collector.usage["completion_tokens"] > 0

    @pytest.mark.asyncio
    async def test_an_endpoint_rejecting_stream_options_the_caller_asked_for_fails(self, async_sdk_client, monkeypatch):
        monkeypatch.setattr(_openai_sdk, "_WITHOUT_STREAM_OPTIONS", set())
        fake = self.RejectsStreamOptions()
        node = make_node(fake, streaming=StreamingConfig(enabled=True, include_usage=True))

        with async_sdk_client(fake), pytest.raises(litellm.BadRequestError):
            await node.execute_async(input_data=MagicMock(), config=RunnableConfig(callbacks=[Collector()]))

        assert len(fake.requests) == 1

    class FailsEveryRequest(FakeOpenAI):
        def __init__(self, status_code: int, error):
            super().__init__()
            self.status_code = status_code
            self.error = error

        def handler(self, request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            self.requests.append(body)
            return httpx.Response(self.status_code, json=self.error(body))

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "status_code, error, sent",
        [
            # FastAPI's validation errors echo the body, so only the retry tells this from a rejected option.
            (
                422,
                lambda body: {"detail": [{"type": "missing", "loc": ["body", "x"], "msg": "Missing", "input": body}]},
                [True, False],
            ),
            # An OpenAI-style error names the param it is about.
            (
                400,
                lambda body: {"error": {"message": f"Invalid request: {body}", "type": "x", "param": "messages"}},
                [True],
            ),
        ],
    )
    async def test_an_unrelated_error_mentioning_stream_options_leaves_later_streams_asking_for_usage(
        self, async_sdk_client, monkeypatch, status_code, error, sent
    ):
        monkeypatch.setattr(_openai_sdk, "_WITHOUT_STREAM_OPTIONS", set())
        fake = self.FailsEveryRequest(status_code, error)

        with async_sdk_client(fake):
            for _ in range(2):
                node = make_node(fake, streaming=StreamingConfig(enabled=True))
                with pytest.raises((litellm.BadRequestError, litellm.UnprocessableEntityError)):
                    await node.execute_async(input_data=MagicMock(), config=RunnableConfig(callbacks=[Collector()]))

        assert ["stream_options" in r for r in fake.requests] == sent * 2


@pytest.mark.parametrize("model", ["gpt-4o-mini", "gpt-4.1", "gpt-5-mini", "o3-mini"])
def test_sends_the_params_litellm_sends(model):
    """With drop_params, LiteLLM drops or rewrites what a model rejects; the SDK transport must send the same body."""
    settings = {
        "temperature": 0.3,
        "top_p": 0.9,
        "presence_penalty": 0.5,
        "frequency_penalty": 0.2,
        "max_tokens": 50,
        "seed": 7,
        "stop": ["END"],
        "logprobs": True,
        "top_logprobs": 2,
        "user": "u1",
    }
    bodies = {}
    for native in (False, True):
        fake = FakeOpenAI()
        node = make_node(fake, native=native, model=model, **settings)
        result = node.run_sync(input_data={}, config=RunnableConfig(callbacks=[]))
        assert result.status == RunnableStatus.SUCCESS, result.error
        bodies[native] = fake.requests[0]

    assert bodies[True] == bodies[False]


class TestLiteLLMBodyTransforms:
    """The SDK transport sends messages and tools as LiteLLM rewrites them for the model and endpoint."""

    MESSAGES = [
        {"role": "system", "content": "Be brief."},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Read these.", "cache_control": {"type": "ephemeral"}},
                {"type": "image_url", "image_url": "https://example.test/a.png"},
                {"type": "image_url", "image_url": {"url": "https://example.test/b.png", "format": "image/png"}},
                {"type": "file", "file": {"file_data": "data:application/pdf;base64,JVBERi0="}},
            ],
        },
    ]
    TOOLS = [
        {
            "type": "function",
            "function": {"name": "search", "parameters": {"type": "object", "properties": {"q": {"type": "string"}}}},
            "cache_control": {"type": "ephemeral"},
        }
    ]

    def params(self, model: str, endpoint: str) -> dict:
        return {
            "model": f"openai/{model}",
            "messages": copy.deepcopy(self.MESSAGES),
            "tools": copy.deepcopy(self.TOOLS),
            "api_base": endpoint,
            "api_key": "k",
            "drop_params": True,
        }

    def check(self, bodies: dict, endpoint: str) -> None:
        assert bodies[True] == bodies[False]
        parts = bodies[True]["messages"][1]["content"]
        assert parts[1]["image_url"] == {"url": "https://example.test/a.png"}
        assert "format" not in parts[2]["image_url"]
        assert parts[3]["file"]["filename"]
        # LiteLLM strips cache breakpoints for OpenAI's own host and keeps them for compatible endpoints.
        assert ("cache_control" in json.dumps(bodies[True])) is (endpoint == BASE_URL)

    @pytest.mark.parametrize("model", ["gpt-4o-mini", "o3-mini"])
    @pytest.mark.parametrize("endpoint", ["https://api.openai.com/v1", BASE_URL])
    def test_sync(self, model, endpoint):
        bodies = {}
        fake = FakeOpenAI()
        litellm.completion(**self.params(model, endpoint), client=fake.client(endpoint))
        bodies[False] = fake.requests[-1]
        _openai_sdk.create(fake.client(endpoint), _openai_sdk.build_request(self.params(model, endpoint), "openai/"))
        bodies[True] = fake.requests[-1]

        self.check(bodies, endpoint)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("model", ["gpt-4o-mini", "o3-mini"])
    @pytest.mark.parametrize("endpoint", ["https://api.openai.com/v1", BASE_URL])
    async def test_async(self, model, endpoint):
        bodies = {}
        fake = FakeOpenAI()
        await litellm.acompletion(**self.params(model, endpoint), client=fake.async_client(endpoint))
        bodies[False] = fake.requests[-1]
        request = _openai_sdk.build_request(self.params(model, endpoint), "openai/")
        await _openai_sdk.acreate(fake.async_client(endpoint), request)
        bodies[True] = fake.requests[-1]

        self.check(bodies, endpoint)
