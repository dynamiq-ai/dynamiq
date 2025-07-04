from __future__ import annotations

import logging
from typing import Any
from urllib.parse import urljoin

import requests
from tenacity import retry, retry_if_exception_type, retry_if_result, stop_after_attempt, wait_exponential

from dynamiq.connections import HTTPMethod

from .config import Settings


class HTTPError(RuntimeError):
    """Raised for non-2xx responses after retries."""


_RETRY_STATUS = {502, 503, 504}


class ApiClient:

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client = requests.Session()

    def get(self, path: str, *, params: dict[str, Any] | None = None) -> Any:
        return self._request("GET", path, params=params)

    def post(
        self,
        path: str,
        *,
        headers: dict[str, Any] | None = None,
        json: dict[str, Any] | None = None,
        data: dict[str, Any] | None = None,
        files: dict[str, Any] | None = None,
    ) -> Any:
        return self._request("POST", path, headers=headers, json=json, data=data, files=files)

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=0.25, min=0.25, max=4),
        retry=(
            retry_if_exception_type(requests.RequestException)
            | retry_if_result(lambda r: r is not None and r.status_code in _RETRY_STATUS)
        ),
        reraise=True,
    )
    def _request(
        self,
        method: str | HTTPMethod,
        path: str,
        *,
        headers: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
        json: dict[str, Any] | None = None,
        data: dict[str, Any] | None = None,
        files: dict[str, Any] | None = None,
        timeout: float = 30.0,
    ) -> Any:
        url = urljoin(self._settings.api_host, path.lstrip("/"))
        if headers is None:
            headers = {}
        headers["Authorization"] = f"Bearer {self._settings.api_key}"
        try:
            response = self._client.request(
                method,
                url,
                params=params,
                json=json,
                data=data,
                files=files,
                headers=headers,
                timeout=timeout,
            )
            if response.status_code != 200:
                logging.error(f"{method} {path} failed with {response.status_code}: {response.text.strip()}")
            return response
        except Exception as e:
            logging.error(str(e))
            raise
