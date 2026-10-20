import enum
import os
from functools import cached_property
from typing import Any, ClassVar

from openai import OpenAI as OpenAIClient
from pydantic import Field

from dynamiq.connections import OpenAI as OpenAIConnection
from dynamiq.nodes.llms import _openai_sdk
from dynamiq.nodes.llms._strict import OpenAIStrictToolsMixin
from dynamiq.nodes.llms.base import BaseLLM

NATIVE_SDK_ENV_VAR = "DYNAMIQ_OPENAI_NATIVE_SDK"


def _native_sdk_default() -> bool:
    """`use_native_sdk` default: on when the environment variable is set to a true value."""
    return os.getenv(NATIVE_SDK_ENV_VAR, "").strip().lower() in ("1", "true", "yes", "on")


class ReasoningEffort(str, enum.Enum):
    """
    The reasoning effort to use for the OpenAI LLM.
    """

    AUTO = "auto"
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    MINIMAL = "minimal"


# OpenAI models whose native default for 'reasoning_effort' is "none"
_MODELS_DEFAULTING_TO_NONE: frozenset[str] = frozenset(
    {
        "gpt-5.1",
        "gpt-5.1-2025-11-13",
        "gpt-5.2",
        "gpt-5.2-2025-12-11",
        "gpt-5.4",
        "gpt-5.4-2026-03-05",
        "gpt-5.4-mini",
        "gpt-5.4-mini-2026-03-17",
        "gpt-5.4-nano",
        "gpt-5.4-nano-2026-03-17",
        "gpt-5.5",
        "gpt-5.5-2026-04-23",
        "gpt-5.6",
        "gpt-5.6-terra",
        "gpt-5.6-luna",
        "gpt-5.6-sol",
        "gpt-6-sol",
        "gpt-6-luna",
    }
)


def _resolve_default_reasoning_effort(model_lower: str) -> ReasoningEffort:
    """Return the effort ``AUTO`` resolves to for ``model_lower``.

    ``model_lower`` is the model identifier with the ``openai/`` prefix stripped
    and lowercased. Models on the "defaults to none" list get ``NONE`` sent
    explicitly rather than the param left out: litellm takes an unset effort on
    gpt-5.4+ models as reasoning on, and then routes any call with function tools
    through the Responses API instead of Chat Completions. Other models fall back
    to the historical default of ``MEDIUM``.
    """
    if model_lower in _MODELS_DEFAULTING_TO_NONE:
        return ReasoningEffort.NONE
    return ReasoningEffort.MEDIUM


class Verbosity(str, enum.Enum):
    """
    The verbosity level for the OpenAI LLM.
    """

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class OpenAI(OpenAIStrictToolsMixin, BaseLLM):
    """OpenAI LLM node.

    This class provides an implementation for the OpenAI Language Model node.

    Attributes:
        connection (OpenAIConnection | None): The connection to use for the OpenAI LLM.
        strict_tools: Inherited from :class:`BaseLLM`. False (default, or an empty
            list) ships every tool as-is; True converts each tool's parameter schema
            into OpenAI structured-outputs strict form (every property required,
            optionals re-encoded as nullable types, ``additionalProperties: false``
            on every object) and attaches ``strict: true``; a list of tool (function)
            names converts only those tools and ships the rest untouched.
        use_native_sdk (bool): Send requests through the OpenAI SDK rather than LiteLLM. Streams cost several
            times less CPU per chunk, so one process serves several times more concurrent streams. Requests using
            params the SDK transport does not translate still go through LiteLLM. Defaults to the
            ``DYNAMIQ_OPENAI_NATIVE_SDK`` environment variable, off when unset.
    """
    connection: OpenAIConnection | None = None
    reasoning_effort: ReasoningEffort | None = ReasoningEffort.AUTO
    verbosity: Verbosity | None = Verbosity.MEDIUM
    use_native_sdk: bool = Field(
        default_factory=_native_sdk_default,
        description=(
            "Send requests through the OpenAI SDK rather than LiteLLM, which costs several times less CPU per "
            f"streamed chunk. Defaults to the {NATIVE_SDK_ENV_VAR} environment variable."
        ),
    )
    O_SERIES_MODEL_PREFIXES: ClassVar[tuple[str, ...]] = ("o1", "o3", "o4")
    GPT_REASONING_MODEL_PREFIXES: ClassVar[tuple[str, ...]] = ("gpt-5", "gpt-6")
    MODEL_PREFIX = "openai/"

    def __init__(self, **kwargs):
        """Initialize the OpenAI LLM node.

        Args:
            **kwargs: Additional keyword arguments.
        """
        if kwargs.get("client") is None and kwargs.get("connection") is None:
            kwargs["connection"] = OpenAIConnection()
        super().__init__(**kwargs)

    @cached_property
    def is_o_series_model(self) -> bool:
        """
        Determine if the model belongs to the O-series (e.g. o1 or o3, o4)
        by checking if the model starts with any of the O-series prefixes.
        """
        model_lower = self.model.lower().removeprefix(self.MODEL_PREFIX)
        return any(model_lower.startswith(prefix) for prefix in self.O_SERIES_MODEL_PREFIXES)

    def _effective_reasoning_effort(self, model_lower: str) -> ReasoningEffort | None:
        """Resolve reasoning_effor parameter for the model."""
        if self.reasoning_effort == ReasoningEffort.AUTO:
            return _resolve_default_reasoning_effort(model_lower)
        return self.reasoning_effort

    @staticmethod
    def _apply_reasoning_effort(params: dict[str, Any], effort: ReasoningEffort | None) -> None:
        """Set or remove ``reasoning_effort`` on ``params`` based on ``effort``."""
        if effort is None:
            params.pop("reasoning_effort", None)
        else:
            params["reasoning_effort"] = effort

    def update_completion_params(self, params: dict[str, Any]) -> dict[str, Any]:
        """
        Override the base method to update the completion parameters for OpenAI.
        For new series models, use "max_completion_tokens" instead of "max_tokens".
        """
        params = super().update_completion_params(params)

        new_params = params.copy()
        model_lower = self.model.lower().removeprefix(self.MODEL_PREFIX)
        if self.is_o_series_model:
            new_params["max_completion_tokens"] = self.max_tokens
            if model_lower.startswith("o3") or model_lower.startswith("o4"):
                self._apply_reasoning_effort(new_params, self._effective_reasoning_effort(model_lower))
            if model_lower not in ["o3-mini"]:
                new_params.pop("stop", None)
            new_params.pop("max_tokens", None)
            new_params.pop("temperature", None)
        elif model_lower.startswith(self.GPT_REASONING_MODEL_PREFIXES):
            if "chat" not in model_lower:
                new_params["verbosity"] = self.verbosity
                if "pro" in model_lower:
                    self._apply_reasoning_effort(new_params, ReasoningEffort.HIGH)
                else:
                    self._apply_reasoning_effort(new_params, self._effective_reasoning_effort(model_lower))
            new_params["max_completion_tokens"] = self.max_tokens
            new_params.pop("stop", None)
            new_params.pop("max_tokens", None)

        return new_params

    def _native_request(self, params: dict[str, Any]) -> dict[str, Any] | None:
        """The OpenAI SDK request for `params`, or None to send them through LiteLLM.

        The SDK clients are the connection's, so a call whose endpoint or key differs from it (an `api_base` or
        `api_key` set on the node) goes through LiteLLM, which sends it where it asks.
        """
        if not self.use_native_sdk or self.connection is None:
            return None
        if (params.get("api_base"), params.get("api_key")) != (self.connection.url, self.connection.api_key):
            return None
        return _openai_sdk.build_request(params, self.MODEL_PREFIX)

    def _call_completion(self, params: dict[str, Any]) -> Any:
        request = self._native_request(params)
        if request is None or not isinstance(self.client, OpenAIClient):
            return super()._call_completion(params)
        return _openai_sdk.create(self.client, request)

    async def _acall_completion(self, params: dict[str, Any]) -> Any:
        request = self._native_request(params)
        if request is None:
            return await super()._acall_completion(params)
        return await _openai_sdk.acreate(await self.get_async_client(), request)

    def _build_streamed_response(self, chunks: list, messages: list[dict]) -> Any:
        if _openai_sdk.is_sdk_stream(chunks):
            return _openai_sdk.build_response(chunks, messages, self.model)
        return super()._build_streamed_response(chunks, messages)
