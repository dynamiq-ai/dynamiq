"""The router's model list, which the Dynamiq LLM node reads its model's capabilities from."""

import threading
import time
from collections.abc import Awaitable, Callable, Iterable
from typing import Any

from pydantic import BaseModel

from dynamiq.nodes.llms.base import ModelInfo
from dynamiq.utils.logger import logger

# How long a router's model list is used before it is fetched again: a catalog edit reaches running nodes within it.
MODEL_LIST_TTL_SECONDS = 600

# How long a failed fetch is not retried, so an unreachable router does not slow every capability lookup.
MODEL_LIST_RETRY_SECONDS = 60

# The model list request's timeout, which bounds how long a capability lookup waits for it.
MODEL_LIST_TIMEOUT_SECONDS = 3.0


def _now() -> float:
    return time.monotonic()


def model_info_from_catalog(model: dict[str, Any]) -> ModelInfo:
    """The capabilities of a model in the router's list.

    Prompt caching is left out: the catalog says whether a model caches at all, while the node asks whether it needs
    cache breakpoints, which only Claude does.
    """
    inputs = model.get("input_modalities") or ()
    return ModelInfo(
        max_input_tokens=model.get("context_length"),
        supports_vision="image" in inputs,
        supports_pdf_input="file" in inputs,
        supports_video_input="video" in inputs,
        supports_function_calling=model.get("supports_tools"),
        supports_response_schema=model.get("supports_structured_output"),
    )


class _RouterModels(BaseModel):
    models: dict[str, ModelInfo] | None = None
    fetched_at: float | None = None
    failed_at: float | None = None


class RouterCatalog:
    """The model lists of the routers the process calls, by router address.

    A router lists the same models to every caller, so the cache holds no caller's data. A list is fetched again after
    `MODEL_LIST_TTL_SECONDS`, and a failed fetch keeps the last list and is not retried for `MODEL_LIST_RETRY_SECONDS`.
    Safe for concurrent use: one thread fetches a router's list while the others wait for it.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._fetch_locks: dict[str, threading.Lock] = {}
        self._routers: dict[str, _RouterModels] = {}

    def lookup(self, router_url: str, slug: str) -> ModelInfo | None:
        """The model's capabilities as last fetched, or None for a model the list lacks or a router not yet fetched."""
        with self._lock:
            router = self._routers.get(router_url)
        return (router.models or {}).get(slug) if router else None

    def get(self, router_url: str, slug: str, fetch: Callable[[], Iterable[dict[str, Any]]]) -> ModelInfo | None:
        """`lookup`, after fetching the router's list with `fetch` when it is due."""
        if self._due(router_url):
            with self._fetch_lock(router_url):
                if self._due(router_url):
                    self._store(router_url, self._fetched(router_url, fetch))
        return self.lookup(router_url, slug)

    async def aget(
        self, router_url: str, slug: str, fetch: Callable[[], Awaitable[Iterable[dict[str, Any]]]]
    ) -> ModelInfo | None:
        """`get` with an async `fetch`. No lock is held across it, so two coroutines may fetch a due list at once."""
        if self._due(router_url):
            try:
                models = list(await fetch())
            except Exception as e:
                models = self._failed(router_url, e)
            self._store(router_url, models)
        return self.lookup(router_url, slug)

    def clear(self) -> None:
        with self._lock:
            self._routers.clear()

    def _fetch_lock(self, router_url: str) -> threading.Lock:
        with self._lock:
            return self._fetch_locks.setdefault(router_url, threading.Lock())

    def _due(self, router_url: str) -> bool:
        now = _now()
        with self._lock:
            router = self._routers.get(router_url)
        if router is None:
            return True
        if router.failed_at is not None and now - router.failed_at < MODEL_LIST_RETRY_SECONDS:
            return False
        return router.fetched_at is None or now - router.fetched_at >= MODEL_LIST_TTL_SECONDS

    def _fetched(self, router_url: str, fetch: Callable[[], Iterable[dict[str, Any]]]) -> list[dict[str, Any]] | None:
        try:
            return list(fetch())
        except Exception as e:
            return self._failed(router_url, e)

    @staticmethod
    def _failed(router_url: str, error: Exception) -> None:
        logger.warning(f"Failed to list the models of the router at {router_url}, keeping the last list: {error}")

    def _store(self, router_url: str, models: list[dict[str, Any]] | None) -> None:
        now = _now()
        with self._lock:
            router = self._routers.setdefault(router_url, _RouterModels())
            if models is None:
                router.failed_at = now
                return
            router.models = {model["id"]: model_info_from_catalog(model) for model in models if model.get("id")}
            router.fetched_at, router.failed_at = now, None


ROUTER_CATALOG = RouterCatalog()
