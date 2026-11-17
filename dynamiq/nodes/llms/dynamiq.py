from typing import TYPE_CHECKING, Any, ClassVar, Literal

from pydantic import Field

from dynamiq.connections import Dynamiq as DynamiqConnection
from dynamiq.connections.managers import ConnectionClientInitType
from dynamiq.nodes.llms.anthropic import AnthropicCacheControl
from dynamiq.nodes.llms.base import LLM_DEFAULT_MAX_TOKENS, BaseLLM, BaseLLMUsageData, ModelInfo
from dynamiq.runnables import RunnableConfig

if TYPE_CHECKING:
    from openai import AsyncOpenAI

CONNECTION_ID_HEADER = "Dynamiq-Connection-Id"

# The key the node adds to each reasoning detail it returns, naming the model that produced it. The router turns a
# detail into the provider's own reasoning block, which another provider or model cannot read, so a detail goes back
# only to its model, and the key is removed before it is sent.
_REASONING_MODEL_KEY = "model"

# What BaseLLM sends for LiteLLM alone: the connection's endpoint, key and headers, which the SDK client holds, and
# LiteLLM's param filtering and retries.
_LITELLM_ONLY_PARAMS = frozenset(
    {"allowed_openai_params", "api_base", "api_key", "api_version", "client", "drop_params", "headers", "num_retries"}
)

# The slugs of models that cache nothing without a breakpoint. The router may serve Claude from Anthropic, Bedrock or
# Vertex, and converts the markers for each.
_BREAKPOINT_MODEL_PREFIXES = ("anthropic/",)

# Anthropic rejects a request with more cache breakpoints than this.
_MAX_CACHE_BREAKPOINTS = 4


def _cache_breakpoints(message: dict) -> int:
    content = message.get("content")
    if not isinstance(content, list):
        return 0
    return sum(1 for part in content if isinstance(part, dict) and part.get("cache_control"))


def _with_cache_breakpoint(message: dict, control: dict) -> dict | None:
    """A copy of `message` with `control` on its last text part, or None when it has no text to mark.

    String content becomes a single text part, since a breakpoint goes on a content part.
    """
    content = message.get("content")
    if isinstance(content, str):
        return message | {"content": [{"type": "text", "text": content, "cache_control": control}]} if content else None
    if isinstance(content, list):
        for index in range(len(content) - 1, -1, -1):
            part = content[index]
            if isinstance(part, dict) and part.get("type") == "text" and part.get("text"):
                parts = list(content)
                parts[index] = part | {"cache_control": control}
                return message | {"content": parts}
    return None


class DynamiqCacheControl(AnthropicCacheControl):
    """Prompt caching config for the Dynamiq router node.

    Attributes:
        automatic: Ask the provider to place the rolling breakpoint itself, through one top-level
            ``cache_control``, instead of marking the message at ``cache_injection_point_index``. The provider
            puts it on the last block it can cache and moves it forward as the conversation grows. The system
            breakpoint still goes in the messages when ``cache_system`` is set. The model's provider must support
            automatic caching: Anthropic's API does, and the router turns it into a cache point on Bedrock.
    """

    automatic: bool = False


def _with_cache_breakpoints(messages: list[dict], cache_control: AnthropicCacheControl) -> tuple[list[dict], bool]:
    """A copy of `messages` with prompt caching breakpoints, as the Anthropic node has LiteLLM place them, and whether
    the request also asks the provider for the automatic one.

    The last system message with text gets one when `cache_system` is set, and the message at
    `cache_injection_point_index` gets the rolling one, or the nearest message before it with text, such as the one
    before an assistant turn that only calls tools. The system breakpoint comes first, so it is the one kept when the
    request has no room left for both. With automatic caching, the rolling breakpoint is the provider's, and it takes
    one of the request's breakpoints, so it is left out when the messages already carry all of them.
    """
    control = cache_control.model_dump(exclude_none=True, include={"type", "ttl"})
    existing = sum(_cache_breakpoints(message) for message in messages)
    automatic = isinstance(cache_control, DynamiqCacheControl) and cache_control.automatic
    automatic = automatic and existing < _MAX_CACHE_BREAKPOINTS
    room = _MAX_CACHE_BREAKPOINTS - existing - (1 if automatic else 0)
    starts = []
    if cache_control.cache_system:
        systems = [index for index, message in enumerate(messages) if message.get("role") == "system"]
        starts += [index for index in systems if _with_cache_breakpoint(messages[index], control)][-1:]
    if not automatic and -len(messages) <= cache_control.cache_injection_point_index < len(messages):
        starts.append(cache_control.cache_injection_point_index % len(messages))

    marked = list(messages)
    done: set[int] = set()
    for start in starts:
        for index in range(start, -1, -1):
            if room <= 0 or index in done:
                break
            if (message := _with_cache_breakpoint(marked[index], control)) is not None:
                marked[index] = message
                done.add(index)
                room -= 1
                break
    return marked, automatic


class Dynamiq(BaseLLM):
    """LLM node that calls models through the Dynamiq router.

    The router serves OpenAI chat completions for the models in its catalog, picks the provider for each call,
    meters it to the caller's project and returns its cost. Requests go through the OpenAI SDK, not LiteLLM.

    Attributes:
        connection (DynamiqConnection): The router's address and credentials. The node calls its `router_url`, never
            its `url`, which is the platform API. On the platform, nexus fills `router_url` with the internal router
            address; outside it, `router_url` defaults to the public router. Its API key, a Dynamiq access key or a
            router token, authenticates the calls. Its `headers` go with every request as they are, so the platform
            passes any routing headers the router needs through them; the node adds none of its own.
        model (str): The model's slug in the router's catalog, such as "anthropic/claude-sonnet-5".
        provider_connection_id (str | None): A connection of the project whose provider key the router uses instead
            of Dynamiq's.
        model_info (ModelInfo | None): What the model supports. Unset fields take `DEFAULT_MODEL_INFO`, since
            LiteLLM does not know the router's slugs.
        cache_control (DynamiqCacheControl | AnthropicCacheControl | Literal[False] | None): Prompt caching config,
            as on the Anthropic node: breakpoints on the system message and on a rolling message, placed in the
            messages themselves, or the rolling one left to the provider with `automatic`. None chooses nothing,
            leaving an Agent free to turn caching on for a model that `supports_prompt_caching`; False opts out.

    An assistant message's `reasoning_details`, which the node returns with a reply, go back to the router with the
    message, so a model keeps its reasoning across the calls of a tool loop. OpenAI stores nothing for the router's
    calls, so its reasoning models need them to continue from their earlier reasoning; Claude's signed thinking needs
    them too. Each detail the node returns names its model, and goes back only to that model, so a fallback to
    another model or a model switch drops it. Details without a model, from the caller, go back as they are.
    """

    connection: DynamiqConnection
    provider_connection_id: str | None = Field(
        default=None,
        description="A connection of the project whose provider key the router uses instead of Dynamiq's.",
    )
    # DynamiqCacheControl first: an Agent turning caching on builds the first config type listed.
    cache_control: DynamiqCacheControl | AnthropicCacheControl | Literal[False] | None = None

    REPLAYS_REASONING_DETAILS: ClassVar[bool] = True
    # The connection's default client is the requests module, for the platform API's clients.
    CLIENT_INIT_TYPE: ClassVar[ConnectionClientInitType] = ConnectionClientInitType.ROUTER

    # Input beyond text stays off until `model_info` turns it on. Tools and JSON schema output are on, as BaseLLM
    # assumes for a model it does not know: off, an agent would refuse function calling for every router model.
    DEFAULT_MODEL_INFO: ClassVar[ModelInfo] = ModelInfo(
        max_input_tokens=LLM_DEFAULT_MAX_TOKENS,
        supports_vision=False,
        supports_pdf_input=False,
        supports_video_input=False,
        supports_function_calling=True,
        supports_response_schema=True,
    )

    def __init__(self, **kwargs):
        if kwargs.get("client") is None and kwargs.get("connection") is None:
            kwargs["connection"] = DynamiqConnection()
        super().__init__(**kwargs)

    async def _async_client(self) -> "AsyncOpenAI":
        """An SDK client on the connection's httpx client for the running event loop, which holds the connections."""
        from openai import AsyncOpenAI  # deferred: importing the node must not load the OpenAI SDK

        return AsyncOpenAI(
            api_key=self.connection.api_key,
            base_url=self.connection.router_url,
            default_headers=self.connection.headers,
            http_client=await self.get_async_client(),
        )

    def replayed_reasoning_details(self, details: list[dict] | None) -> list[dict]:
        """The details this node's model produced, or that name no model, without the model they name."""
        return [
            {key: value for key, value in detail.items() if key != _REASONING_MODEL_KEY}
            for detail in details or ()
            if detail.get(_REASONING_MODEL_KEY, self.model) == self.model
        ]

    def _model_capability(self, name: str) -> Any:
        """`model_info`'s value for `name`, or `DEFAULT_MODEL_INFO`'s when it has none."""
        value = getattr(self.model_info, name, None) if self.model_info else None
        return getattr(self.DEFAULT_MODEL_INFO, name) if value is None else value

    def get_token_limit(self) -> int:
        return self._model_capability("max_input_tokens")

    @property
    def is_vision_supported(self) -> bool:
        return self._model_capability("supports_vision")

    @property
    def is_function_calling_supported(self) -> bool:
        return self._model_capability("supports_function_calling")

    @property
    def is_pdf_input_supported(self) -> bool:
        return self._model_capability("supports_pdf_input")

    @property
    def is_video_input_supported(self) -> bool:
        return self._model_capability("supports_video_input")

    @property
    def is_response_schema_supported(self) -> bool:
        return self._model_capability("supports_response_schema")

    def supports_prompt_caching(self) -> bool:
        """Whether the model caches nothing without a breakpoint: an "anthropic/" slug, unless `model_info` says."""
        if self.model_info and self.model_info.supports_prompt_caching is not None:
            return self.model_info.supports_prompt_caching
        return self.model.startswith(_BREAKPOINT_MODEL_PREFIXES)

    def _tool_params_to_force(self) -> list[str]:
        # The router gets every param the node sends: there is no LiteLLM filter to push tools through.
        return []

    @classmethod
    def get_usage_data(cls, model: str, completion: Any) -> BaseLLMUsageData:
        """Usage data, with the cache writes the router reports in OpenAI's `cache_write_tokens`."""
        usage_data = super().get_usage_data(model, completion)
        if usage_data.cache_creation_input_tokens is None:
            details = cls._usage_value(cls._extract_usage(completion), "prompt_tokens_details", None)
            usage_data.cache_creation_input_tokens = cls._usage_value(details, "cache_write_tokens", None)
        return usage_data

    @classmethod
    def _usage_costs(
        cls, completion: Any, usage: Any, **tokens: Any
    ) -> tuple[float | None, float | None, float | None]:
        """The cost the router reported for the call, which it prices from its own catalog."""
        return None, None, cls._reported_cost(completion, usage)

    def _router_request(self, params: dict[str, Any]) -> tuple[dict[str, Any], bool]:
        """The SDK arguments for BaseLLM's completion `params`, and whether the caller asked to stream usage.

        The router silently drops what it does not know, so params are sent in its format: Anthropic's `thinking`
        becomes `reasoning`, and arguments the SDK does not take, such as `reasoning` and `top_k`, go in the body
        through `extra_body`.
        """
        from dynamiq.nodes.llms import _openai_sdk  # deferred: importing the node must not load the OpenAI SDK

        request = {
            key: value
            for key, value in params.items()
            if value is not None and key not in _LITELLM_ONLY_PARAMS and not (key == "stream" and not value)
        }
        if request.get("messages"):
            request["messages"] = [self._replayable(message) for message in request["messages"]]
        if self.cache_control and request.get("messages"):
            # In the messages, since the router drops LiteLLM's cache_control_injection_points.
            request["messages"], automatic = _with_cache_breakpoints(request["messages"], self.cache_control)
            if automatic:
                request["cache_control"] = self.cache_control.model_dump(exclude_none=True, include={"type", "ttl"})
        if self.provider_connection_id:
            # Per request: nodes with different project connections share the connection's client.
            request["extra_headers"] = {CONNECTION_ID_HEADER: self.provider_connection_id}
        if "max_tokens" in request:
            request.setdefault("max_completion_tokens", request.pop("max_tokens"))
        thinking = request.pop("thinking", None)
        if isinstance(thinking, dict) and thinking.get("type") == "enabled" and thinking.get("budget_tokens"):
            request["reasoning"] = {"max_tokens": thinking["budget_tokens"]} | (request.get("reasoning") or {})

        emit_usage = False
        if request.get("stream"):
            stream_options = request.get("stream_options") or {}
            emit_usage = bool(stream_options.get("include_usage"))
            # The usage, and the cost in it, arrives only in a stream's last chunk, and only when asked for.
            request["stream_options"] = stream_options | {"include_usage": True}
        return _openai_sdk.with_extra_body(request), emit_usage

    def _replayable(self, message: dict) -> dict:
        """`message` with only the reasoning details that go back to this node's model."""
        if "reasoning_details" not in message:
            return message
        others = {key: value for key, value in message.items() if key != "reasoning_details"}
        details = self.replayed_reasoning_details(message["reasoning_details"])
        return others | {"reasoning_details": details} if details else others

    def _call_completion(self, params: dict[str, Any]) -> Any:
        from dynamiq.nodes.llms import _openai_sdk  # deferred: importing the node must not load the OpenAI SDK

        request, emit_usage = self._router_request(params)
        return _openai_sdk.send(self.client, request, emit_usage)

    async def _acall_completion(self, params: dict[str, Any]) -> Any:
        from dynamiq.nodes.llms import _openai_sdk  # deferred: importing the node must not load the OpenAI SDK

        request, emit_usage = self._router_request(params)
        return await _openai_sdk.asend(await self._async_client(), request, emit_usage)

    def _handle_completion_response(self, response: Any, config: RunnableConfig = None, **kwargs) -> dict:
        """The reply, with the `reasoning_details` it came with, each naming the model, which an agent sends back
        with the turn."""
        result = super()._handle_completion_response(response=response, config=config, **kwargs)
        details = getattr(response.choices[0].message, "reasoning_details", None)
        if isinstance(details, list) and details:
            result["reasoning_details"] = [detail | {_REASONING_MODEL_KEY: self.model} for detail in details]
        return result

    def _build_streamed_response(self, chunks: list, messages: list[dict], stream: Any = None) -> Any:
        return stream.build_response(messages, self.model)
