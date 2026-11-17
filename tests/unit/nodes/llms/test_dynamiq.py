import asyncio
import copy
import json
from unittest.mock import AsyncMock, patch

import httpx
import litellm
import pytest
from openai import OpenAI as OpenAIClient
from openai.types.chat import ChatCompletion

from dynamiq import Workflow
from dynamiq.callbacks.streaming import BaseStreamingCallbackHandler, StreamingIteratorCallbackHandler
from dynamiq.connections import Dynamiq as DynamiqConnection
from dynamiq.connections import OpenAI as OpenAIConnection
from dynamiq.connections import connections
from dynamiq.connections.managers import ConnectionManager
from dynamiq.flows import Flow
from dynamiq.nodes.agents import Agent
from dynamiq.nodes.agents.utils import SummarizationConfig
from dynamiq.nodes.llms import AnthropicCacheControl, Dynamiq, DynamiqCacheControl, OpenAI
from dynamiq.nodes.llms.base import LLM_DEFAULT_MAX_TOKENS, FallbackConfig, FallbackTrigger, ModelInfo
from dynamiq.nodes.node import ErrorHandling, is_retryable_error
from dynamiq.nodes.types import InferenceMode
from dynamiq.prompts import Prompt
from dynamiq.runnables import RunnableConfig, RunnableStatus
from dynamiq.types.streaming import StreamingConfig
from tests.helpers.agent_hooks import SearchTool

ROUTER_URL = "https://router.test/v1"
PLATFORM_URL = "https://api.test"
TOKEN = "router-token"
# A slug LiteLLM does not know, so nothing here can lean on LiteLLM's tables.
MODEL = "meta-llama/llama-4-maverick"
USAGE = {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18, "cost": 0.00042}
HELLO = {"role": "assistant", "content": "Hello"}
THINKING = {
    "role": "assistant",
    "content": "Hello",
    "reasoning": "Let me think.",
    "reasoning_details": [{"index": 0, "type": "reasoning.text", "text": "Let me think.", "signature": "sig-1"}],
}


def tool_call(call_id: str, name: str, **arguments) -> dict:
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": json.dumps(arguments)}}


def chunk(model: str, delta: dict, finish_reason: str | None = None, usage: dict | None = None) -> dict:
    data = {
        "id": "gen-1",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }
    if usage:
        data["usage"] = usage
    return data


def stream_of(message: dict, model: str) -> list[dict]:
    """`message` as the router streams it: reasoning, content and tool calls in pieces, usage on the last chunk.

    Reasoning text comes under both of its spellings, the signature of a thinking block in a delta of its own, and
    encrypted reasoning whole, as the router streams it when a reasoning item finishes.
    """
    chunks = [chunk(model, {"role": "assistant"})]
    reasoning = message.get("reasoning") or ""
    for piece in (reasoning[:5], reasoning[5:]) if reasoning else ():
        detail = {"index": 0, "type": "reasoning.text", "text": piece}
        chunks.append(chunk(model, {"reasoning": piece, "reasoning_content": piece, "reasoning_details": [detail]}))
    for detail in message.get("reasoning_details") or ():
        if detail.get("signature"):
            signature = {"index": detail["index"], "type": detail["type"], "signature": detail["signature"]}
            chunks.append(chunk(model, {"reasoning_details": [signature]}))
        if detail.get("data"):
            chunks.append(chunk(model, {"reasoning_details": [detail]}))
    content = message.get("content") or ""
    for piece in (content[:3], content[3:]) if content else ():
        chunks.append(chunk(model, {"content": piece}))
    for index, call in enumerate(message.get("tool_calls") or ()):
        first = {"index": index, "id": call["id"], "type": "function"}
        chunks.append(chunk(model, {"tool_calls": [first | {"function": {"name": call["function"]["name"]}}]}))
        arguments = call["function"]["arguments"]
        for start in range(0, len(arguments), 5):
            piece = {"index": index, "function": {"arguments": arguments[start : start + 5]}}
            chunks.append(chunk(model, {"tool_calls": [piece]}))
    finish_reason = "tool_calls" if message.get("tool_calls") else "stop"
    chunks.append(chunk(model, {}, finish_reason=finish_reason, usage=USAGE))
    return chunks


class FakeRouter:
    """The router's chat completions endpoint on an httpx MockTransport.

    It answers the first requests with the `errors` status codes, then each later one with the next of `replies`,
    the last one repeated. Its errors say "x-should-retry: false", so the SDK does not retry them itself.
    """

    def __init__(self):
        self.replies: list[dict] = [HELLO]
        self.errors: list[int] = []
        self.requests: list[httpx.Request] = []
        self._replied = 0

    @property
    def bodies(self) -> list[dict]:
        return [json.loads(request.content) for request in self.requests]

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if self.errors:
            return httpx.Response(
                self.errors.pop(0),
                headers={"x-should-retry": "false"},
                json={"error": {"code": "provider_error", "message": "boom"}},
            )
        body = json.loads(request.content)
        message = self.replies[min(self._replied, len(self.replies) - 1)]
        self._replied += 1
        if body.get("stream"):
            events = [f"data: {json.dumps(data)}\n\n" for data in stream_of(message, body["model"])]
            content = "".join(events) + "data: [DONE]\n\n"
            return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=content.encode())
        finish_reason = "tool_calls" if message.get("tool_calls") else "stop"
        return httpx.Response(
            200,
            json={
                "id": "gen-1",
                "object": "chat.completion",
                "created": 1,
                "model": body["model"],
                "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
                "usage": USAGE,
            },
        )


@pytest.fixture
def router(monkeypatch) -> FakeRouter:
    """A FakeRouter that every node built in the test calls, sync and async."""
    fake = FakeRouter()
    transport = httpx.MockTransport(fake.handler)
    monkeypatch.setattr(connections, "_openai_http_client", lambda: httpx.Client(transport=transport))
    monkeypatch.setattr(
        DynamiqConnection, "connect_async", AsyncMock(side_effect=lambda: httpx.AsyncClient(transport=transport))
    )
    return fake


class Collector(BaseStreamingCallbackHandler):
    def __init__(self):
        self.chunks = []
        self.usage = None

    def on_node_execute_stream(self, serialized, chunk=None, **kwargs):
        self.chunks.append(chunk)

    def on_node_execute_run(self, serialized, **kwargs):
        if "usage_data" in kwargs:
            self.usage = kwargs["usage_data"]


def make_node(**kwargs) -> Dynamiq:
    kwargs.setdefault("model", MODEL)
    kwargs.setdefault("prompt", Prompt(messages=[{"role": "user", "content": "hi"}]))
    kwargs.setdefault("connection", DynamiqConnection(api_key=TOKEN, url=PLATFORM_URL, router_url=ROUTER_URL))
    return Dynamiq(**kwargs)


def run(node: Dynamiq, run_async: bool, collector: Collector | None = None):
    config = RunnableConfig(callbacks=[collector] if collector else [])
    if run_async:
        return asyncio.run(node.run_async(input_data={}, config=config))
    return node.run(input_data={}, config=config)


class TestRequest:
    def test_sends_the_call_to_the_router_with_the_connection_s_key(self, router):
        result = make_node().run(input_data={})

        assert result.status == RunnableStatus.SUCCESS, result.error
        assert result.output == {"content": "Hello"}
        [request] = router.requests
        assert str(request.url) == f"{ROUTER_URL}/chat/completions"
        assert request.headers["authorization"] == f"Bearer {TOKEN}"
        assert "dynamiq-synapse-handler" not in request.headers
        assert "dynamiq-connection-id" not in request.headers
        assert router.bodies == [{"model": MODEL, "messages": [{"role": "user", "content": "hi"}]}]

    @pytest.mark.parametrize("run_async", [False, True])
    def test_sends_the_connection_s_headers_as_they_are(self, router, run_async):
        connection = DynamiqConnection(
            api_key=TOKEN, router_url=ROUTER_URL, headers={"Dynamiq-Synapse-Handler": "router"}
        )

        run(make_node(connection=connection), run_async)

        [request] = router.requests
        assert request.headers["dynamiq-synapse-handler"] == "router"
        assert request.headers["authorization"] == f"Bearer {TOKEN}"
        assert "dynamiq-connection-id" not in request.headers

    @pytest.mark.asyncio
    async def test_names_the_project_connection_whose_key_the_router_uses(self, router):
        await make_node(provider_connection_id="conn-1").run_async(input_data={})

        [request] = router.requests
        assert request.headers["dynamiq-connection-id"] == "conn-1"
        assert request.headers["authorization"] == f"Bearer {TOKEN}"
        assert "dynamiq-synapse-handler" not in request.headers

    @pytest.mark.parametrize("run_async", [False, True])
    def test_calls_the_connection_s_router_url_not_its_url(self, router, run_async):
        url = "http://synapse.internal:8080/v1"
        node = make_node(connection=DynamiqConnection(api_key=TOKEN, url=PLATFORM_URL, router_url=url))

        result = run(node, run_async)

        assert result.status == RunnableStatus.SUCCESS, result.error
        [request] = router.requests
        assert str(request.url) == f"{url}/chat/completions"
        assert str(node.client.base_url) == f"{url}/"
        assert "router_url" not in Dynamiq.model_fields

    def test_sends_thinking_as_reasoning_and_top_k_in_the_body(self, router):
        make_node(thinking_enabled=True, budget_tokens=2048, top_k=40).run(input_data={})

        [body] = router.bodies
        assert body["reasoning"] == {"max_tokens": 2048}
        assert body["top_k"] == 40
        assert "thinking" not in body

    def test_passes_reasoning_effort_through(self, router):
        make_node(reasoning_effort="low").run(input_data={})

        [body] = router.bodies
        assert body["reasoning_effort"] == "low"
        assert "reasoning" not in body

    def test_sends_none_of_the_arguments_only_litellm_takes(self, router):
        tools = [{"type": "function", "function": {"name": "search", "parameters": {"type": "object"}}}]

        make_node(tools=tools, max_tokens=100, num_retries=3).run(input_data={})

        [body] = router.bodies
        litellm_only = {
            "drop_params",
            "allowed_openai_params",
            "api_base",
            "api_key",
            "headers",
            "client",
            "num_retries",
        }
        assert not litellm_only & set(body)
        assert body["tools"] == tools
        assert body["max_completion_tokens"] == 100


class TestResponse:
    @pytest.mark.parametrize("streaming", [False, True])
    def test_takes_usage_and_cost_from_the_router(self, router, streaming):
        collector = Collector()
        # LiteLLM prices this slug, so the cost shows whose price was taken.
        node = make_node(model="anthropic/claude-sonnet-5", streaming=StreamingConfig(enabled=streaming))

        result = node.run(input_data={}, config=RunnableConfig(callbacks=[collector]))

        assert result.status == RunnableStatus.SUCCESS, result.error
        assert collector.usage["total_tokens_cost_usd"] == USAGE["cost"]
        assert collector.usage["prompt_tokens_cost_usd"] is None
        assert collector.usage["completion_tokens_cost_usd"] is None
        assert (collector.usage["prompt_tokens"], collector.usage["completion_tokens"]) == (11, 7)

    @pytest.mark.parametrize("run_async", [False, True])
    def test_merges_a_streamed_reply(self, router, run_async):
        router.replies = [
            {
                "role": "assistant",
                "content": "Searching",
                "tool_calls": [tool_call("call_1", "search", query="kyiv"), tool_call("call_2", "calc", expr="1+1")],
            }
        ]
        collector = Collector()

        result = run(make_node(streaming=StreamingConfig(enabled=True)), run_async, collector)

        assert result.status == RunnableStatus.SUCCESS, result.error
        assert result.output["content"] == "Searching"
        assert result.output["tool_calls"] == [
            {"id": "call_1", "type": "function", "function": {"name": "search", "arguments": {"query": "kyiv"}}},
            {"id": "call_2", "type": "function", "function": {"name": "calc", "arguments": {"expr": "1+1"}}},
        ]
        [body] = router.bodies
        assert body["stream"] is True
        assert body["stream_options"] == {"include_usage": True}
        assert collector.chunks, "the node streamed no chunk"
        assert collector.usage["total_tokens_cost_usd"] == USAGE["cost"]

    @pytest.mark.parametrize("streaming", [False, True])
    def test_keeps_reasoning_and_signed_thinking_readable(self, router, streaming):
        router.replies = [THINKING]
        node = make_node(streaming=StreamingConfig(enabled=streaming))
        config = RunnableConfig(callbacks=[Collector()])

        with patch.object(
            Dynamiq, "_handle_completion_response", autospec=True, side_effect=Dynamiq._handle_completion_response
        ) as handle:
            result = node.run(input_data={}, config=config)

        assert result.status == RunnableStatus.SUCCESS, result.error
        message = handle.call_args.kwargs["response"].choices[0].message
        assert message.content == "Hello"
        assert message.reasoning == THINKING["reasoning"]
        assert message.reasoning_details == THINKING["reasoning_details"]
        assert result.output["reasoning_details"] == [
            detail | {"model": MODEL} for detail in THINKING["reasoning_details"]
        ]


class TestErrors:
    @pytest.mark.parametrize(
        "status, error, retryable",
        [
            (429, litellm.RateLimitError, True),
            (500, litellm.InternalServerError, True),
            (503, litellm.ServiceUnavailableError, True),
            (400, litellm.BadRequestError, False),
            (401, litellm.AuthenticationError, False),
        ],
    )
    def test_raises_the_litellm_error_retries_and_fallbacks_classify(self, router, status, error, retryable):
        router.errors = [status]

        with pytest.raises(error) as raised:
            make_node().execute(input_data={})

        assert is_retryable_error(raised.value) is retryable

    @pytest.mark.parametrize("status, attempts, outcome", [(429, 2, RunnableStatus.SUCCESS), (400, 1, "failure")])
    def test_retries_what_can_succeed_on_a_retry(self, router, status, attempts, outcome):
        router.errors = [status]
        node = make_node(error_handling=ErrorHandling(max_retries=1, retry_interval_seconds=0))

        result = node.run(input_data={})

        assert result.status == (RunnableStatus.FAILURE if outcome == "failure" else outcome)
        assert len(router.requests) == attempts

    @pytest.mark.asyncio
    async def test_falls_back_on_a_rate_limit(self, router):
        router.errors = [429]
        fallback = make_node(model="openai/gpt-6-sol")
        node = make_node(fallback=FallbackConfig(llm=fallback, enabled=True, triggers=[FallbackTrigger.RATE_LIMIT]))

        result = await node.run_async(input_data={})

        assert result.status == RunnableStatus.SUCCESS, result.error
        assert [body["model"] for body in router.bodies] == [MODEL, "openai/gpt-6-sol"]

    def test_a_connection_failure_is_a_connection_error(self):
        def refuse(request):
            raise httpx.ConnectError("connection refused", request=request)

        node = make_node()
        node.client = OpenAIClient(
            api_key=TOKEN,
            base_url=ROUTER_URL,
            max_retries=0,
            http_client=httpx.Client(transport=httpx.MockTransport(refuse)),
        )

        with pytest.raises(litellm.APIConnectionError) as raised:
            node.execute(input_data={})

        assert node._is_connection_error(raised.type, str(raised.value).lower())


class TestModelCapabilities:
    def test_answers_without_litellm(self, router):
        def unknown(*args, **kwargs):
            raise AssertionError("LiteLLM was asked about a router model")

        with (
            patch.multiple(
                "dynamiq.nodes.llms.base",
                get_model_info=unknown,
                get_max_tokens=unknown,
                get_supported_openai_params=unknown,
                supports_response_schema=unknown,
                supports_vision=unknown,
                supports_pdf_input=unknown,
                litellm_video_input_flag=unknown,
            ),
            patch("litellm.cost_per_token", unknown),
        ):
            node = make_node(tools=[{"type": "function", "function": {"name": "search", "parameters": {}}}])
            for mode in (InferenceMode.FUNCTION_CALLING, InferenceMode.STRUCTURED_OUTPUT):
                Agent(name="agent", llm=node, tools=[SearchTool()], inference_mode=mode)
            result = node.run(input_data={})

            assert result.status == RunnableStatus.SUCCESS, result.error
            assert node.get_token_limit() == LLM_DEFAULT_MAX_TOKENS
            assert node.is_function_calling_supported is True
            assert node.is_response_schema_supported is True
            assert node.is_vision_supported is False
            assert node.is_pdf_input_supported is False
            assert node.is_video_input_supported is False

    def test_model_info_overrides_the_defaults(self):
        node = make_node(
            model_info=ModelInfo(max_input_tokens=200_000, supports_vision=True, supports_function_calling=False)
        )

        assert node.get_token_limit() == 200_000
        assert node.is_vision_supported is True
        assert node.is_function_calling_supported is False
        assert node.is_pdf_input_supported is False


EPHEMERAL = {"type": "ephemeral", "ttl": "5m"}
SYSTEM = {"role": "system", "content": "Be brief."}


def text(content: str, cache_control: dict | None = None) -> dict:
    part = {"type": "text", "text": content}
    return part | {"cache_control": cache_control} if cache_control else part


class TestPromptCaching:
    def test_marks_the_system_message_and_the_last_message(self, router):
        prompt = Prompt(messages=[SYSTEM, {"role": "user", "content": "hi"}])

        make_node(prompt=prompt, cache_control=AnthropicCacheControl()).run(input_data={})

        [body] = router.bodies
        assert body["messages"] == [
            {"role": "system", "content": [text("Be brief.", EPHEMERAL)]},
            {"role": "user", "content": [text("hi", EPHEMERAL)]},
        ]
        assert "cache_control_injection_points" not in body
        assert "cache_control" not in body

    @pytest.mark.parametrize("tool_result", [True, False], ids=["tool-result", "tool-calls-only"])
    def test_marks_a_tool_result_or_the_last_message_with_text_before_a_tool_call(self, router, tool_result):
        calling = {"role": "assistant", "content": "", "tool_calls": [tool_call("call_1", "search", query="kyiv")]}
        result = {"role": "tool", "content": "Sunny", "tool_call_id": "call_1"}
        messages = [SYSTEM, {"role": "user", "content": "Weather?"}, calling] + ([result] if tool_result else [])

        make_node(prompt=Prompt(messages=messages), cache_control=AnthropicCacheControl()).run(input_data={})

        [body] = router.bodies
        sent = body["messages"]
        assert sent[0]["content"] == [text("Be brief.", EPHEMERAL)]
        assert sent[2]["content"] in ("", None)
        if tool_result:
            assert sent[1]["content"] == "Weather?"
            assert sent[3] == {"role": "tool", "content": [text("Sunny", EPHEMERAL)], "tool_call_id": "call_1"}
        else:
            assert sent[1]["content"] == [text("Weather?", EPHEMERAL)]

    def test_marks_the_last_text_part_and_leaves_the_caller_s_messages_alone(self):
        image = {"type": "image_url", "image_url": {"url": "https://example.test/a.png"}}
        messages = [SYSTEM, {"role": "user", "content": [text("first"), text("second"), image]}]
        sent_before = copy.deepcopy(messages)

        request, _ = make_node(cache_control=AnthropicCacheControl())._router_request(
            {"model": MODEL, "messages": messages}
        )

        assert messages == sent_before
        assert request["messages"][1]["content"] == [text("first"), text("second", EPHEMERAL), image]

    def test_stays_within_four_breakpoints_keeping_the_system_one(self):
        marked = [{"role": "user", "content": [text(f"turn {n}", EPHEMERAL)]} for n in range(3)]
        messages = [SYSTEM, *marked, {"role": "user", "content": "last"}]

        request, _ = make_node(cache_control=AnthropicCacheControl())._router_request(
            {"model": MODEL, "messages": messages}
        )

        assert request["messages"][0]["content"] == [text("Be brief.", EPHEMERAL)]
        assert request["messages"][-1]["content"] == "last"

    def test_leaves_the_rolling_breakpoint_to_the_provider_with_automatic_caching(self, router):
        prompt = Prompt(messages=[SYSTEM, {"role": "user", "content": "hi"}])

        make_node(prompt=prompt, cache_control=DynamiqCacheControl(automatic=True)).run(input_data={})

        [body] = router.bodies
        assert body["cache_control"] == EPHEMERAL
        assert body["messages"] == [
            {"role": "system", "content": [text("Be brief.", EPHEMERAL)]},
            {"role": "user", "content": "hi"},
        ]

    def test_automatic_caching_takes_one_of_the_four_breakpoints(self):
        marked = [{"role": "user", "content": [text(f"turn {n}", EPHEMERAL)]} for n in range(3)]
        messages = [SYSTEM, *marked, {"role": "user", "content": "last"}]

        request, _ = make_node(cache_control=DynamiqCacheControl(automatic=True))._router_request(
            {"model": MODEL, "messages": messages}
        )

        assert request["messages"][0] == SYSTEM
        assert request["extra_body"]["cache_control"] == EPHEMERAL

    @pytest.mark.parametrize("existing, automatic", [(3, True), (4, False)])
    def test_automatic_caching_is_left_out_when_the_messages_use_every_breakpoint(self, existing, automatic):
        marked = [{"role": "user", "content": [text(f"turn {n}", EPHEMERAL)]} for n in range(existing)]
        messages = [*marked, {"role": "user", "content": [text("new turn")]}]

        request, _ = make_node(
            model="anthropic/claude-sonnet-5", cache_control=DynamiqCacheControl(automatic=True)
        )._router_request({"model": "anthropic/claude-sonnet-5", "messages": messages})

        assert request["messages"] == messages
        assert ("cache_control" in request.get("extra_body", {})) is automatic

    def test_reads_cache_reads_and_writes_as_the_router_reports_them(self):
        details = {"cached_tokens": 4334, "cached_read_tokens": 4334, "cache_write_tokens": 120}
        usage = {"prompt_tokens": 4487, "completion_tokens": 30, "total_tokens": 4517, "prompt_tokens_details": details}
        completion = ChatCompletion.construct(id="gen-1", choices=[], created=1, model=MODEL, usage=usage)

        usage_data = Dynamiq.get_usage_data(model=MODEL, completion=completion)

        assert (usage_data.cache_read_input_tokens, usage_data.cache_creation_input_tokens) == (4334, 120)

    @pytest.mark.parametrize("cache_control", [False, None])
    def test_sends_no_breakpoints_unless_configured(self, router, cache_control):
        messages = [SYSTEM, {"role": "user", "content": "hi"}]

        make_node(prompt=Prompt(messages=messages), cache_control=cache_control).run(input_data={})

        [body] = router.bodies
        assert body["messages"] == messages

    @pytest.mark.parametrize(
        "model, model_info, cached",
        [
            ("anthropic/claude-sonnet-5", None, True),
            ("openai/gpt-6-sol", None, False),
            (MODEL, None, False),
            ("anthropic/claude-sonnet-5", ModelInfo(supports_prompt_caching=False), False),
            (MODEL, ModelInfo(supports_prompt_caching=True), True),
        ],
    )
    def test_an_agent_turns_caching_on_for_models_that_need_breakpoints(self, model, model_info, cached):
        agent = Agent(name="agent", llm=make_node(model=model, model_info=model_info), tools=[SearchTool()])

        assert agent.llm.cache_control == (DynamiqCacheControl() if cached else None)

    def test_an_agent_on_a_claude_slug_sends_breakpoints(self, router):
        router.replies = [
            {"role": "assistant", "content": "", "tool_calls": [tool_call("call_1", "search", thought="t", query="x")]},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [tool_call("call_2", "provide_final_answer", answer="y")],
            },
        ]
        agent = Agent(
            name="agent",
            llm=make_node(model="anthropic/claude-sonnet-5"),
            tools=[SearchTool()],
            inference_mode=InferenceMode.FUNCTION_CALLING,
            max_loops=3,
        )

        result = agent.run(input_data={"input": "Find x."})

        assert result.status == RunnableStatus.SUCCESS, result.error
        for body in router.bodies:
            assert body["messages"][0]["role"] == "system"
            assert body["messages"][0]["content"][-1]["cache_control"] == EPHEMERAL
            assert body["messages"][-1]["content"][-1]["cache_control"] == EPHEMERAL
        assert router.bodies[-1]["messages"][-1]["role"] == "tool"


ENCRYPTED = [{"id": "rs_1", "index": 0, "type": "reasoning.encrypted", "data": "enc-1"}]


class TestReasoningReplay:
    @pytest.mark.parametrize("streaming", [False, True])
    def test_an_agent_sends_the_reasoning_back_with_its_tool_calls(self, router, streaming):
        router.replies = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [tool_call("call_1", "search", thought="t", query="kyiv")],
                "reasoning_details": ENCRYPTED,
            },
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [tool_call("call_2", "provide_final_answer", thought="done", answer="Sunny")],
            },
        ]
        agent = Agent(
            name="agent",
            llm=make_node(model="openai/gpt-6-sol"),
            tools=[SearchTool()],
            inference_mode=InferenceMode.FUNCTION_CALLING,
            max_loops=3,
            streaming=StreamingConfig(enabled=streaming),
        )

        result = agent.run(
            input_data={"input": "Weather in Kyiv?"},
            config=RunnableConfig(callbacks=[StreamingIteratorCallbackHandler()]),
        )

        assert result.status == RunnableStatus.SUCCESS, result.error
        calling = next(m for m in router.bodies[1]["messages"] if m.get("tool_calls"))
        assert calling["reasoning_details"] == ENCRYPTED
        assert "reasoning_details" not in router.bodies[0]["messages"][-1]

    def test_a_fallback_to_another_model_gets_no_reasoning_from_the_first(self, router):
        router.replies = [
            {
                "role": "assistant",
                "content": "",
                "reasoning_details": ENCRYPTED,
                "tool_calls": [tool_call("call_1", "search")],
            },
            HELLO,
        ]
        primary = make_node(model="openai/gpt-6-sol")
        calling = primary.run(input_data={}).output
        messages = [
            {"role": "user", "content": "Weather?"},
            {
                "role": "assistant",
                "content": "Calling search",
                "tool_calls": [tool_call("call_1", "search")],
                "reasoning_details": calling["reasoning_details"],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "Sunny"},
        ]
        router.errors = [429]
        primary = make_node(
            model="openai/gpt-6-sol",
            prompt=Prompt(messages=messages),
            fallback=FallbackConfig(
                llm=make_node(model="anthropic/claude-sonnet-5"), enabled=True, triggers=[FallbackTrigger.RATE_LIMIT]
            ),
        )

        assert primary.run(input_data={}).status == RunnableStatus.SUCCESS

        _, failed, fallback = router.bodies
        assert failed["messages"][1]["reasoning_details"] == ENCRYPTED
        assert "reasoning_details" not in fallback["messages"][1]

    def test_an_agent_compacts_its_history_when_the_reasoning_it_sends_back_fills_the_context(self, router):
        details = [{"index": 0, "type": "reasoning.text", "text": "ponder " * 20000, "signature": "sig-1"}]
        router.replies = [
            {
                "role": "assistant",
                "content": "",
                "reasoning_details": details,
                "tool_calls": [tool_call("call_1", "search", thought="t", query="x")],
            },
            {"role": "assistant", "content": "Searched for x."},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [tool_call("call_2", "provide_final_answer", answer="y")],
            },
        ]
        agent = Agent(
            name="agent",
            llm=make_node(model="anthropic/claude-sonnet-5"),
            tools=[SearchTool()],
            inference_mode=InferenceMode.FUNCTION_CALLING,
            max_loops=3,
            summarization_config=SummarizationConfig(enabled=True, max_token_context_length=10000),
        )

        result = agent.run(input_data={"input": "Find x."})

        assert result.status == RunnableStatus.SUCCESS, result.error
        calling, summarizing, answering = router.bodies
        assert not any("reasoning_details" in message for message in calling["messages"])
        assert "Provide a concise summary" in json.dumps(summarizing["messages"])
        assert any(
            str(message.get("content")).startswith("Observation: Searched for x.") for message in answering["messages"]
        )

    def test_other_nodes_drop_the_reasoning(self):
        node = OpenAI(model="gpt-6-sol", connection=OpenAIConnection(api_key="sk-test"))
        calling = {"role": "assistant", "content": "Calling: search", "tool_calls": [tool_call("call_1", "search")]}
        messages = [{"role": "user", "content": "hi"}, calling | {"reasoning_details": ENCRYPTED}]

        params = node._build_completion_params(
            messages=messages, config=RunnableConfig(callbacks=[]), prompt=Prompt(messages=messages)
        )

        assert params["messages"] == [{"role": "user", "content": "hi"}, calling]
        assert messages[1]["reasoning_details"] == ENCRYPTED


@pytest.mark.parametrize("streaming", [False, True])
def test_runs_an_agent_that_calls_a_tool(router, streaming):
    router.replies = [
        {"role": "assistant", "content": "", "tool_calls": [tool_call("call_1", "search", thought="t", query="kyiv")]},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [tool_call("call_2", "provide_final_answer", thought="done", answer="Sunny in Kyiv")],
        },
    ]
    agent = Agent(
        name="agent",
        llm=make_node(),
        tools=[SearchTool()],
        inference_mode=InferenceMode.FUNCTION_CALLING,
        max_loops=3,
        streaming=StreamingConfig(enabled=streaming),
    )

    result = agent.run(
        input_data={"input": "Weather in Kyiv?"}, config=RunnableConfig(callbacks=[StreamingIteratorCallbackHandler()])
    )

    assert result.status == RunnableStatus.SUCCESS, result.error
    assert result.output["content"] == "Sunny in Kyiv"
    first, second = router.bodies
    assert {tool["function"]["name"] for tool in first["tools"]} >= {"search", "provide_final_answer"}
    assert first.get("stream", False) is streaming
    assert "allowed_openai_params" not in first
    assert {"role": "tool", "tool_call_id": "call_1"}.items() <= second["messages"][-1].items()
    assert "results for kyiv" in second["messages"][-1]["content"]


class TestClient:
    def test_the_connection_manager_closes_the_router_client(self, router):
        manager = ConnectionManager()
        node = make_node(is_postponed_component_init=True)
        node.init_components(manager)
        assert node.run(input_data={}).status == RunnableStatus.SUCCESS
        client = node.client

        manager.close()

        assert client.is_closed()

    def test_reconnects_to_the_router_after_its_client_closes(self, router):
        node = make_node()
        assert node.run(input_data={}).status == RunnableStatus.SUCCESS
        node.client.close()

        result = node.run(input_data={})

        assert result.status == RunnableStatus.SUCCESS, result.error
        assert len(router.requests) == 2

    def test_nodes_sharing_a_client_send_their_own_project_connection(self, router):
        manager = ConnectionManager()
        connection = DynamiqConnection(api_key=TOKEN, url=PLATFORM_URL, router_url=ROUTER_URL)
        nodes = [
            make_node(connection=connection, provider_connection_id=connection_id, is_postponed_component_init=True)
            for connection_id in ("conn-1", None)
        ]
        for node in nodes:
            node.init_components(manager)

        for node in nodes:
            node.run(input_data={})

        assert nodes[0].client is nodes[1].client
        first, second = router.requests
        assert first.headers["dynamiq-connection-id"] == "conn-1"
        assert "dynamiq-connection-id" not in second.headers


def test_survives_a_yaml_round_trip(router, tmp_path):
    node = make_node(
        id="router-llm", provider_connection_id="conn-1", model_info=ModelInfo(supports_vision=True), prompt=None
    )
    path = tmp_path / "workflow.yaml"
    Workflow(flow=Flow(nodes=[node])).to_yaml_file(path)

    loaded = Workflow.from_yaml_file(str(path), init_components=True).flow.nodes[0]

    assert isinstance(loaded, Dynamiq)
    assert (loaded.model, loaded.provider_connection_id) == (MODEL, "conn-1")
    assert (loaded.connection.url, loaded.connection.router_url) == (PLATFORM_URL, ROUTER_URL)
    assert loaded.model_info.supports_vision is True
    assert loaded.connection.api_key == TOKEN
    loaded.prompt = Prompt(messages=[{"role": "user", "content": "hi"}])
    assert loaded.run(input_data={}).output == {"content": "Hello"}
    assert router.requests[-1].headers["dynamiq-connection-id"] == "conn-1"
