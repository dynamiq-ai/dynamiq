from __future__ import annotations

import time
from typing import Any

import requests

from .config import Settings

__all__ = ["ApiClient", "HTTPError"]

from ..connections import HTTPMethod


class HTTPError(RuntimeError):
    """Raised for non-2xx responses after retries."""


_RETRY_STATUS = {502, 503, 504}
_RETRY_BACKOFF = (0.25, 1.0, 2.0, 4.0)  # seconds


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
        url = f"{self._settings.api_host}{path}"
        if headers is None:
            headers = {}
        headers["Authorization"] = f"Bearer {self._settings.api_token}"
        for attempt, backoff in enumerate((*_RETRY_BACKOFF, None), start=1):
            try:
                with self._client.request(
                    method,
                    url,
                    params=params,
                    json=json,
                    data=data,
                    files=files,
                    headers=headers,
                    timeout=timeout,
                ) as resp:
                    if resp.status_code < 400:
                        return resp

                    if resp.status_code in _RETRY_STATUS and backoff is not None:
                        time.sleep(backoff)
                        continue

                    raise HTTPError(f"{method} {path} failed with {resp.status_code}: {resp.text.strip()}")
            except HTTPError as e:
                print(e)
        raise HTTPError("Exhausted retries")
