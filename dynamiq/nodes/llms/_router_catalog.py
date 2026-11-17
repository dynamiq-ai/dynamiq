"""The router's model list, which the Dynamiq LLM node reads its model's capabilities from."""

import hashlib
import threading
import time
from collections.abc import Awaitable, Callable, Iterable
from typing import Any
from urllib.parse import urlsplit, urlunsplit

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
    """The capabilities of a model in the router's list. Raises for an entry that does not describe them.

    Prompt caching is left out: the catalog says whether a model caches at all, while the node asks whether it needs
    cache breakpoints, which only Claude does.
    """
    inputs = model.get("input_modalities") or []
    if not isinstance(inputs, list):
        raise ValueError("input_modalities is not a list")
    return ModelInfo(
        max_input_tokens=model.get("context_length"),
        supports_vision="image" in inputs,
        supports_pdf_input="file" in inputs,
        supports_video_input="video" in inputs,
        supports_function_calling=model.get("supports_tools"),
        supports_response_schema=model.get("supports_structured_output"),
    )


def _parsed(models: Any) -> dict[str, ModelInfo] | None:
    """The capabilities of each entry of a model list that describes them, or None for a list with none that does.

    An entry that does not parse costs only its own model.
    """
    if not isinstance(models, list):
        return None
    parsed = {}
    for model in models:
        try:
            if not isinstance(model.get("id"), str):
                raise ValueError("id is not a string")
            parsed[model["id"]] = model_info_from_catalog(model)
        except Exception as e:
            logger.debug(f"Skipped a router model list entry that does not parse: {type(e).__name__}")
    return parsed if parsed or not models else None


def _redacted(url: str) -> str:
    """`url` without the credentials it may carry in its userinfo or query."""
    try:
        parts = urlsplit(url)
        host = parts.hostname or ""
        if parts.port:
            host = f"{host}:{parts.port}"
        return urlunsplit((parts.scheme, host, parts.path, "", ""))
    except ValueError:
        return "an invalid URL"


def _refused(error: Exception) -> bool:
    """Whether the router refused the caller, rather than failing for everyone: a 4xx other than 404."""
    status = getattr(error, "status_code", None)
    return isinstance(status, int) and 400 <= status < 500 and status != 404


class _RouterModels(BaseModel):
    models: dict[str, ModelInfo] | None = None
    fetched_at: float | None = None
    failed_at: float | None = None


class RouterCatalog:
    """The model lists of the routers the process calls, by router address.

    A router lists the same models to every caller, so the cache holds no caller's data. A list is fetched again after
    `MODEL_LIST_TTL_SECONDS`, and a failed fetch keeps the last list and is not retried for `MODEL_LIST_RETRY_SECONDS`.
    A fetch the router refuses, such as for a revoked key, is retried later only for that caller's key, since it says
    nothing about the list other callers get. Safe for concurrent use: one thread fetches a router's list while the
    others wait for it.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._fetch_locks: dict[str, threading.Lock] = {}
        self._routers: dict[str, _RouterModels] = {}
        self._refused: dict[tuple[str, str], float] = {}

    def lookup(self, router_url: str, slug: str) -> ModelInfo | None:
        """The model's capabilities as last fetched, or None for a model the list lacks or a router not yet fetched."""
        with self._lock:
            router = self._routers.get(router_url)
        return (router.models or {}).get(slug) if router else None

    def get(
        self, router_url: str, api_key: str, slug: str, fetch: Callable[[], Iterable[dict[str, Any]]]
    ) -> ModelInfo | None:
        """`lookup`, after fetching the router's list with `fetch`, which sends `api_key`, when it is due."""
        caller = _fingerprint(api_key)
        if self._due(router_url, caller):
            with self._fetch_lock(router_url):
                if self._due(router_url, caller):
                    try:
                        self._store(router_url, caller, list(fetch()))
                    except Exception as e:
                        self._failed(router_url, caller, e)
        return self.lookup(router_url, slug)

    async def aget(
        self, router_url: str, api_key: str, slug: str, fetch: Callable[[], Awaitable[Iterable[dict[str, Any]]]]
    ) -> ModelInfo | None:
        """`get` with an async `fetch`. No lock is held across it, so two coroutines may fetch a due list at once."""
        caller = _fingerprint(api_key)
        if self._due(router_url, caller):
            try:
                self._store(router_url, caller, list(await fetch()))
            except Exception as e:
                self._failed(router_url, caller, e)
        return self.lookup(router_url, slug)

    def clear(self) -> None:
        with self._lock:
            self._routers.clear()
            self._refused.clear()

    def _fetch_lock(self, router_url: str) -> threading.Lock:
        with self._lock:
            return self._fetch_locks.setdefault(router_url, threading.Lock())

    def _due(self, router_url: str, caller: str) -> bool:
        now = _now()
        with self._lock:
            router = self._routers.get(router_url)
            refused_at = self._refused.get((router_url, caller))
        if refused_at is not None and now - refused_at < MODEL_LIST_RETRY_SECONDS:
            return False
        if router is None:
            return True
        if router.failed_at is not None and now - router.failed_at < MODEL_LIST_RETRY_SECONDS:
            return False
        return router.fetched_at is None or now - router.fetched_at >= MODEL_LIST_TTL_SECONDS

    def _failed(self, router_url: str, caller: str, error: Exception) -> None:
        status = getattr(error, "status_code", None)
        logger.warning(
            f"Failed to list the models of the router at {_redacted(router_url)}, keeping the last list: "
            f"{type(error).__name__}" + (f" {status}" if isinstance(status, int) else "")
        )
        now = _now()
        with self._lock:
            if _refused(error):
                # Kept only while it holds a caller back, so revoked or rotated keys do not pile up.
                self._refused = {key: at for key, at in self._refused.items() if now - at < MODEL_LIST_RETRY_SECONDS}
                self._refused[(router_url, caller)] = now
            else:
                self._routers.setdefault(router_url, _RouterModels()).failed_at = now

    def _store(self, router_url: str, caller: str, models: list[Any]) -> None:
        parsed = _parsed(models)
        if parsed is None:
            raise ValueError("the model list has no entry that parses")
        with self._lock:
            router = self._routers.setdefault(router_url, _RouterModels())
            router.models = parsed
            router.fetched_at, router.failed_at = _now(), None
            self._refused.pop((router_url, caller), None)


def _fingerprint(api_key: str) -> str:
    """A key that names the caller in the cache without holding its credential."""
    return hashlib.sha256((api_key or "").encode()).hexdigest()


ROUTER_CATALOG = RouterCatalog()
