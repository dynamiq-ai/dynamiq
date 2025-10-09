import asyncio
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

    async def request(self, method: str, url_path: URLTypes, **kwargs: Any) -> Response:
        logger.debug(f'[{self.__class__.__name__}] REQ "{method} {url_path}". Kwargs: {kwargs}')
        url = f"{self.base_url}/{str(url_path).lstrip('/')}" if self.base_url else str(url_path).lstrip("/")
        try:
            async with httpx.AsyncClient() as client:  # nosec B113
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

    def trace(self, runs: list["Run"]) -> None:
        """Sync method required by BaseTracingClient interface"""
        if not runs:
            return

        try:
            if is_called_from_async_context():
                loop = asyncio.get_running_loop()
                loop.create_task(self._send_traces_async(runs))
            else:
                self._send_traces_sync(runs)
        except Exception as e:
            logger.error(f"Failed to send traces. Error: {e}")
