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


def rewind(files) -> None:
    """Seek every file handle in a multipart payload back to the start.

    requests reads each handle to EOF while building the body, and a retry re-enters with the
    same objects. Without this, attempt 2 onwards sends zero-byte parts - and the API answers
    2xx, so an upload that hit exactly the gateway error the retry exists for reports success
    over an empty file. Ingestion is asynchronous, so it surfaces much later as an indexed
    item with no content.
    """
    if not files:
        return
    entries = files.values() if isinstance(files, dict) else files
    for entry in entries:
        value = entry[1] if isinstance(entry, (tuple, list)) and len(entry) > 1 else entry
        handle = value[1] if isinstance(value, (tuple, list)) and len(value) > 1 else value
        if hasattr(handle, "seek"):
            try:
                handle.seek(0)
            except (OSError, ValueError):
                # A non-seekable stream cannot be replayed; let the request fail loudly
                # rather than silently uploading nothing.
                raise


def ok(response) -> bool:
    """True for any 2xx.

    Checking `status_code == 200` alone reports a created resource as a failure: `POST /v1/apps`
    answers 201 and the app is deployed, but the caller sees `HTTP 201: {...}` raised as an
    error and cannot tell it apart from a real one.
    """
    return 200 <= response.status_code < 300


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
        timeout: float | None = None,
        retry: bool = True,
    ) -> Any:
        """`retry=False` for anything that is not safe to repeat.

        The retry below re-sends on a ReadTimeout, and a POST that already reached the server
        is not undone by the client giving up on the response: a workflow test would execute
        up to five times, tools really acting each time. An upload is worse than useless on a
        retry - requests has read the file handles to EOF and nothing rewinds them, so the
        repeat sends empty parts and the API answers 2xx.
        """
        send = self._request if retry else self._request_once
        return send("POST", path, headers=headers, json=json, data=data, files=files,
                    **({"timeout": timeout} if timeout is not None else {}))

    def put(
        self,
        path: str,
        *,
        headers: dict[str, Any] | None = None,
        json: dict[str, Any] | None = None,
        data: dict[str, Any] | None = None,
        files: dict[str, Any] | None = None,
    ) -> Any:
        return self._request("PUT", path, headers=headers, json=json, data=data, files=files)

    def delete(
        self,
        path: str,
        *,
        headers: dict[str, Any] | None = None,
        json: dict[str, Any] | None = None,
        data: dict[str, Any] | None = None,
    ) -> Any:
        return self._request("DELETE", path, headers=headers, json=json, data=data)

    def _request_once(self, method, path, **kwargs):
        """One attempt, no retry. Same request, same error handling."""
        return self._request.retry_with(stop=stop_after_attempt(1))(self, method, path, **kwargs)

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
        rewind(files)
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
