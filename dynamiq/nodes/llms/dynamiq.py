from typing import TYPE_CHECKING, Any, ClassVar

from pydantic import Field

from dynamiq.connections import Dynamiq as DynamiqConnection
from dynamiq.connections.connections import _openai_http_client
from dynamiq.nodes.llms.base import LLM_DEFAULT_MAX_TOKENS, BaseLLM, ModelInfo

if TYPE_CHECKING:
    from openai import AsyncOpenAI
    from openai import OpenAI as OpenAIClient

CONNECTION_ID_HEADER = "Dynamiq-Connection-Id"

# What BaseLLM sends for LiteLLM alone: the connection's endpoint, key and headers, which the SDK client holds, and
# LiteLLM's param filtering and retries.
_LITELLM_ONLY_PARAMS = frozenset(
    {"allowed_openai_params", "api_base", "api_key", "api_version", "client", "drop_params", "headers", "num_retries"}
)


class Dynamiq(BaseLLM):
    """LLM node that calls models through the Dynamiq router.

    The router serves OpenAI chat completions for the models in its catalog, picks the provider for each call,
    meters it to the caller's project and returns its cost. Requests go through the OpenAI SDK, not LiteLLM.

    Attributes:
        connection (DynamiqConnection): The router's address and credentials. Its `url` must be the router's
            OpenAI-compatible base, such as "https://router.getdynamiq.ai/v1", not the platform API the connection
            defaults to. Its API key, a Dynamiq access key or a router token, authenticates the calls. Its `headers`
            go with every request as they are, so the platform passes any routing headers the router needs through
            them; the node adds none of its own.
        model (str): The model's slug in the router's catalog, such as "anthropic/claude-sonnet-5".
        provider_connection_id (str | None): A connection of the project whose provider key the router uses instead
            of Dynamiq's.
        model_info (ModelInfo | None): What the model supports. Unset fields take `DEFAULT_MODEL_INFO`, since
            LiteLLM does not know the router's slugs.
    """

    connection: DynamiqConnection
    provider_connection_id: str | None = Field(
        default=None,
        description="A connection of the project whose provider key the router uses instead of Dynamiq's.",
    )

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

    def init_components(self, connection_manager=None):
        # The connection's own client is the requests module, so the node builds the SDK client it calls with.
        if self.client is None:
            self.client = self._create_client()
        super().init_components(connection_manager)

    @property
    def _headers(self) -> dict[str, str]:
        headers = dict(self.connection.headers)
        if self.provider_connection_id:
            headers[CONNECTION_ID_HEADER] = self.provider_connection_id
        return headers

    def _create_client(self) -> "OpenAIClient":
        from openai import OpenAI as OpenAIClient  # deferred: importing the node must not load the OpenAI SDK

        return OpenAIClient(
            api_key=self.connection.api_key,
            base_url=self.connection.url,
            default_headers=self._headers,
            http_client=_openai_http_client(),
        )

    async def _async_client(self) -> "AsyncOpenAI":
        """An SDK client on the connection's httpx client for the running event loop, which holds the connections."""
        from openai import AsyncOpenAI  # deferred: importing the node must not load the OpenAI SDK

        return AsyncOpenAI(
            api_key=self.connection.api_key,
            base_url=self.connection.url,
            default_headers=self._headers,
            http_client=await self.get_async_client(),
        )

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

    def _tool_params_to_force(self) -> list[str]:
        # The router gets every param the node sends: there is no LiteLLM filter to push tools through.
        return []

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

    def _call_completion(self, params: dict[str, Any]) -> Any:
        from dynamiq.nodes.llms import _openai_sdk  # deferred: importing the node must not load the OpenAI SDK

        request, emit_usage = self._router_request(params)
        return _openai_sdk.send(self.client, request, emit_usage)

    async def _acall_completion(self, params: dict[str, Any]) -> Any:
        from dynamiq.nodes.llms import _openai_sdk  # deferred: importing the node must not load the OpenAI SDK

        request, emit_usage = self._router_request(params)
        return await _openai_sdk.asend(await self._async_client(), request, emit_usage)

    def _build_streamed_response(self, chunks: list, messages: list[dict], stream: Any = None) -> Any:
        return stream.build_response(messages, self.model)
