import enum
from functools import cached_property
from typing import Any, ClassVar

from dynamiq.connections import OpenAI as OpenAIConnection
from dynamiq.nodes.llms.base import BaseLLM


class ReasoningEffort(str, enum.Enum):
    """
    The reasoning effort to use for the OpenAI LLM.
    """

    AUTO = "auto"
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
    }
)


def _resolve_default_reasoning_effort(model_lower: str) -> ReasoningEffort | None:
    """Return the implicit default for ``model_lower``, or ``None`` to omit the param.

    ``model_lower`` is the model identifier with the ``openai/`` prefix stripped
    and lowercased. Models not on the explicit "defaults to none" list fall back
    to the historical default of ``MEDIUM``.
    """
    if model_lower in _MODELS_DEFAULTING_TO_NONE:
        return None
    return ReasoningEffort.MEDIUM


class Verbosity(str, enum.Enum):
    """
    The verbosity level for the OpenAI LLM.
    """

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class OpenAI(BaseLLM):
    """OpenAI LLM node.

    This class provides an implementation for the OpenAI Language Model node.

    Attributes:
        connection (OpenAIConnection | None): The connection to use for the OpenAI LLM.
    """
    connection: OpenAIConnection | None = None
    reasoning_effort: ReasoningEffort | None = ReasoningEffort.AUTO
    verbosity: Verbosity | None = Verbosity.MEDIUM
    O_SERIES_MODEL_PREFIXES: ClassVar[tuple[str, ...]] = ("o1", "o3", "o4")
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
        elif model_lower.startswith("gpt-5"):
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
