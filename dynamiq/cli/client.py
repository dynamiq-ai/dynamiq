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


def _last_response(retry_state):
    """What to hand back when the retries are used up.

    A gateway error is a result, not an exception, so `reraise=True` has nothing to re-raise
    and tenacity would raise RetryError naming a Future - hiding the status and body the
    caller needs to report. Return the last response and let them handle it.
    """
    outcome = retry_state.outcome
    if outcome is not None and not outcome.failed:
        return outcome.result()
    raise outcome.exception() if outcome is not None else RuntimeError("no attempt was made")


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
        retry: bool = False,
    ) -> Any:
        """POST does not retry unless you ask. `retry=True` only when repeating is harmless.

        A POST that already reached the server is not undone by the client giving up on the
        response, and almost every POST here creates or runs something: a fine-tuning job, an
        evaluation, a deployment, a trigger run. Retrying one of those on a timeout bills for
        it twice, and none of these endpoints takes an idempotency key to tell the duplicate
        apart. GET, PUT and DELETE keep the retry - they are safe to repeat by definition.

        Opting back in is for calls where a second one costs nothing: minting a short-lived
        token, a search, a status transition that is already where it is going.
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
        """One attempt, no retry - the undecorated call.

        Not `retry_with(stop=...)`: that narrows the stop condition and keeps the result
        predicate, so a 502/503/504 still matches "retry" while the stop says "stop", and
        tenacity raises RetryError over a perfectly good response. `reraise=True` does not help
        - there is no exception to re-raise. The caller would get a traceback naming a Future
        instead of the gateway's status and body.
        """
        return self._request.__wrapped__(self, method, path, **kwargs)

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=0.25, min=0.25, max=4),
        retry=(
            retry_if_exception_type(requests.RequestException)
            | retry_if_result(lambda r: r is not None and r.status_code in _RETRY_STATUS)
        ),
        retry_error_callback=_last_response,
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
