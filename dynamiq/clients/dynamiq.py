import asyncio
import atexit
import threading
import weakref
from queue import SimpleQueue
from typing import TYPE_CHECKING, Any
from urllib.parse import urljoin

import httpx
import orjson
import requests
from httpx import Response
from httpx._types import URLTypes

from dynamiq.clients import BaseTracingClient
from dynamiq.utils import is_called_from_async_context
from dynamiq.utils.env import get_env_var
from dynamiq.utils.logger import logger
from dynamiq.utils.utils import orjson_encode

if TYPE_CHECKING:
    from dynamiq.callbacks.tracing import Run

DYNAMIQ_BASE_URL = "https://collector.getdynamiq.ai"

_FLUSH_TIMEOUT = 0.5  # seconds to wait for remaining traces on shutdown


class HttpBaseError(Exception):
    pass


class HttpConnectionError(HttpBaseError):
    pass


class HttpServerError(HttpBaseError):
    pass


class HttpClientError(HttpBaseError):
    pass


class DynamiqTracingClient(BaseTracingClient):

    def __init__(self, base_url: str | None = None, access_key: str | None = None, timeout: float = 60.0):
        self.base_url = base_url or DYNAMIQ_BASE_URL
        self.access_key = access_key or get_env_var("DYNAMIQ_ACCESS_KEY") or get_env_var("DYNAMIQ_SERVICE_TOKEN")
        if self.access_key is None:
            raise ValueError("No API key provided")
        self.timeout = timeout

        # Background queue and thread for non-blocking sync trace dispatch
        self._trace_queue: SimpleQueue[list["Run"] | None] = SimpleQueue()
        self._bg_thread = threading.Thread(target=self._trace_worker, daemon=True)
        self._bg_thread.start()
        atexit.register(self.close)

        # Lazily initialised async HTTP client (reused across calls).
        # Both the lock and the client are bound to a specific event loop;
        # we track the loop via weakref so we can detect when it changes
        # (e.g. successive asyncio.run() calls in a test suite).
        self._async_client: httpx.AsyncClient | None = None
        self._async_client_lock: asyncio.Lock | None = None
        self._async_client_loop: weakref.ref | None = None
        self._async_init_mutex = threading.Lock()  # guards loop-change detection

    # ------------------------------------------------------------------
    # Background worker
    # ------------------------------------------------------------------

    def _trace_worker(self) -> None:
        """Drain *_trace_queue* in a background daemon thread."""
        while True:
            runs = self._trace_queue.get()
            if runs is None:  # sentinel -> shut down
                break
            self._send_traces_sync(runs)

    def close(self) -> None:
        """Flush pending traces and stop the background thread."""
        self._trace_queue.put(None)  # send sentinel
        self._bg_thread.join(timeout=_FLUSH_TIMEOUT)
        # Best-effort async client cleanup
        client = self._async_client
        if client is not None and not client.is_closed:
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(client.aclose())
            except RuntimeError:
                # No running loop - run synchronously in a fresh loop
                try:
                    asyncio.run(client.aclose())
                except Exception:
                    pass
            self._async_client = None

    async def aclose(self) -> None:
        """Async counterpart of *close()* - shuts down the async HTTP client."""
        client = self._async_client
        if client is not None and not client.is_closed:
            await client.aclose()
            self._async_client = None

    # ------------------------------------------------------------------
    # Sync transport
    # ------------------------------------------------------------------

    def _send_traces_sync(self, runs: list["Run"]) -> None:
        """Synchronous method to send traces using requests"""
        try:
            trace_data = orjson.dumps(
                {"runs": [run.to_dict() for run in runs]},
                default=orjson_encode,
                option=orjson.OPT_NON_STR_KEYS,
            )
            response = requests.post(  # nosec
                urljoin(self.base_url, "/v1/traces"),
                data=trace_data,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.access_key}",
                },
                timeout=self.timeout,
            )
            response.raise_for_status()
        except Exception as e:
            logger.error(f"Failed to send traces (sync). Error: {e}")

    # ------------------------------------------------------------------
    # Async transport
    # ------------------------------------------------------------------

    def _is_same_loop(self, loop: asyncio.AbstractEventLoop) -> bool:
        """Check if *loop* is the same event loop we last bound to."""
        if self._async_client_loop is None:
            return False
        prev = self._async_client_loop()  # dereference weakref
        return prev is loop

    async def _get_async_client(self) -> httpx.AsyncClient:
        """Return a shared *httpx.AsyncClient*, creating it lazily.

        The lock and client are bound to the current event loop.  When the
        loop changes (e.g. a new ``asyncio.run()``), both are recreated so
        we never use an ``asyncio.Lock`` from a dead loop.
        """
        current_loop = asyncio.get_running_loop()

        # Fast path - same loop, client alive
        if (
            self._async_client is not None
            and not self._async_client.is_closed
            and self._is_same_loop(current_loop)
        ):
            return self._async_client

        # Slow path - need to (re)create lock and/or client
        with self._async_init_mutex:
            # Loop changed -> discard old lock (bound to old loop) and client
            if not self._is_same_loop(current_loop):
                old_client = self._async_client
                if old_client is not None and not old_client.is_closed:
                    try:
                        await old_client.aclose()
                    except Exception:
                        pass
                self._async_client = None
                self._async_client_lock = asyncio.Lock()
                self._async_client_loop = weakref.ref(current_loop)

        # Now acquire the (loop-local) async lock for client creation
        async with self._async_client_lock:  # type: ignore[union-attr]
            if self._async_client is None or self._async_client.is_closed:
                self._async_client = httpx.AsyncClient()  # nosec B113
        return self._async_client

    async def request(self, method: str, url_path: URLTypes, **kwargs: Any) -> Response:
        logger.debug(f'[{self.__class__.__name__}] REQ "{method} {url_path}". Kwargs: {kwargs}')
        url = f"{self.base_url}/{str(url_path).lstrip('/')}" if self.base_url else str(url_path).lstrip("/")
        try:
            client = await self._get_async_client()
            response = await client.request(method, url=url, timeout=self.timeout, **kwargs)
        except (httpx.TimeoutException, httpx.NetworkError) as e:
            raise HttpConnectionError(e) from e

        try:
            response.raise_for_status()
        except httpx.HTTPError as e:
            if httpx.codes.is_client_error(response.status_code):
                raise HttpClientError(e, response) from e
            else:
                raise HttpServerError(e, response) from e

        return response

    async def _send_traces_async(self, runs: list["Run"]) -> None:
        """Async method to send traces using httpx"""
        try:
            trace_data = orjson.dumps(
                {"runs": [run.to_dict() for run in runs]},
                default=orjson_encode,
                option=orjson.OPT_NON_STR_KEYS,
            )
            await self.request(
                method="POST",
                url_path="/v1/traces",
                content=trace_data,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.access_key}",
                },
            )
        except Exception as e:
            logger.error(f"Failed to send traces (async). Error: {e}")

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def trace(self, runs: list["Run"]) -> None:
        """Sync method required by BaseTracingClient interface"""
        if not runs:
            return

        try:
            if is_called_from_async_context():
                loop = asyncio.get_running_loop()
                loop.create_task(self._send_traces_async(runs))
            else:
                self._trace_queue.put(runs)
        except Exception as e:
            logger.error(f"Failed to send traces. Error: {e}")
