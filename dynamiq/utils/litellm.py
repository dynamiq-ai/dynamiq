"""Deferred access to litellm.

Importing litellm takes seconds, so dynamiq imports it when it first needs it instead of when dynamiq is imported.
The functions below forward to the litellm functions of the same name, so modules can import them at the top and
tests can patch them where they are used. Code that changes litellm's global state registers a setup step with
:func:`on_litellm_import` instead of importing litellm itself.
"""

import sys
import threading
from collections.abc import Callable
from types import ModuleType
from typing import Any

from dynamiq.utils.lazy_imports import eager_imports_enabled
from dynamiq.utils.logger import logger

_lock = threading.RLock()
_litellm: ModuleType | None = None
_setup_steps: list[Callable[[], Any]] = []


def get_litellm() -> ModuleType:
    """Return the litellm module, importing it on the first call.

    The setup steps registered with :func:`on_litellm_import` run before litellm is returned.
    """
    global _litellm
    if _litellm is None or _setup_steps:
        with _lock:
            import litellm  # deferred: importing litellm takes seconds

            while _setup_steps:
                step = _setup_steps.pop(0)
                try:
                    step()
                except Exception as error:
                    logger.warning(
                        f"Failed to set up litellm with {getattr(step, '__qualname__', step)}: {error}", exc_info=True
                    )
            _litellm = litellm
    return _litellm


def on_litellm_import(step: Callable[[], Any]) -> None:
    """Run ``step`` before dynamiq first uses litellm.

    The step runs now if litellm is already imported, since that costs nothing, and otherwise on the first call to
    :func:`get_litellm`.

    Args:
        step: Changes litellm's global state, such as registering models. An exception it raises is logged. It runs
            while dynamiq holds its litellm lock, so it must not import a module that registers a step itself.
    """
    with _lock:
        _setup_steps.append(step)
    if _litellm is not None or "litellm" in sys.modules:
        get_litellm()


def token_counter(*args: Any, **kwargs: Any) -> int:
    return get_litellm().token_counter(*args, **kwargs)


def get_max_tokens(*args: Any, **kwargs: Any) -> int | None:
    return get_litellm().get_max_tokens(*args, **kwargs)


def get_model_info(*args: Any, **kwargs: Any) -> Any:
    return get_litellm().get_model_info(*args, **kwargs)


def get_supported_openai_params(*args: Any, **kwargs: Any) -> list[str] | None:
    return get_litellm().get_supported_openai_params(*args, **kwargs)


def supports_vision(*args: Any, **kwargs: Any) -> bool:
    return get_litellm().supports_vision(*args, **kwargs)


def supports_response_schema(*args: Any, **kwargs: Any) -> bool:
    return get_litellm().supports_response_schema(*args, **kwargs)


def supports_pdf_input(*args: Any, **kwargs: Any) -> bool:
    return get_litellm().utils.supports_pdf_input(*args, **kwargs)


def supports_prompt_caching(*args: Any, **kwargs: Any) -> bool:
    return get_litellm().utils.supports_prompt_caching(*args, **kwargs)


if eager_imports_enabled():
    get_litellm()
