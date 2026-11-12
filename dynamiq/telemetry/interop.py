"""Detection of other OpenTelemetry instrumentation that records the same model calls as Dynamiq."""

import sys
from typing import Any

from dynamiq.utils.logger import logger

_CALLBACK_LISTS = ("callbacks", "success_callback", "failure_callback", "_async_success_callback")
_OTEL_CALLBACK_NAME = "otel"
_OTEL_CALLBACK_CLASS = "OpenTelemetry"

_warned = False
# Lengths of the callback lists when they were last searched. litellm registers a callback by appending to a
# list, so an unchanged length means there is nothing new to look at.
_searched_lengths: tuple[int, ...] | None = None


def _is_otel_callback(callback: Any) -> bool:
    if callback == _OTEL_CALLBACK_NAME:
        return True
    return any(cls.__name__ == _OTEL_CALLBACK_CLASS for cls in type(callback).__mro__)


def warn_if_litellm_otel_callback() -> None:
    """Warn once per process when LiteLLM's OpenTelemetry callback is enabled.

    The emitter calls this at the start of every recording ``chat`` span, not only the first: litellm is imported
    by the first model call, after the first span has started. It must stay cheap and must not import litellm.
    After the warning it returns at once. Before it, a call costs a ``sys.modules`` lookup and the lengths of four
    lists; the lists are searched again only when one of them has changed size.
    """
    global _warned, _searched_lengths
    if _warned:
        return
    try:
        litellm = sys.modules.get("litellm")
        if litellm is None:
            return
        lists = [getattr(litellm, name, None) for name in _CALLBACK_LISTS]
        lists = [callbacks if isinstance(callbacks, (list, tuple)) else () for callbacks in lists]
        lengths = tuple(len(callbacks) for callbacks in lists)
        if lengths == _searched_lengths:
            return
        _searched_lengths = lengths
        if not any(_is_otel_callback(callback) for callbacks in lists for callback in list(callbacks)):
            return
        _warned = True
        logger.warning(
            "LiteLLM's OpenTelemetry callback is enabled alongside Dynamiq tracing. Inside a Dynamiq `chat` span "
            "it writes attributes and a final status onto that span, and with USE_OTEL_LITELLM_REQUEST_SPAN=true "
            "it adds a second span for the same call. Remove 'otel' from litellm's callbacks, or accept duplicate "
            "inference spans."
        )
    except Exception as e:
        logger.debug(f"Telemetry: LiteLLM callbacks were not checked: {e!r}")
