"""Bounded fixed-endpoint transport for host-owned control requests."""

import json
import time
from typing import Any
from urllib.parse import urlsplit

from requests.auth import AuthBase

from dynamiq.connections import Http, HTTPMethod


class _NoEnvironmentAuth(AuthBase):
    def __call__(self, request):
        return request


def validate_control_connection(connection: Http) -> None:
    parsed = urlsplit(connection.url)
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or connection.method != HTTPMethod.POST
    ):
        raise ValueError("Remote control requires a fixed credential-free HTTPS POST URL")


def post_control_json(client: Any, connection: Http, payload: dict, timeout: float, max_response_bytes: int) -> dict:
    """Return a bounded JSON object; errors contain neither payloads nor endpoints."""
    try:
        deadline = time.monotonic() + timeout
        with client.post(
            connection.url,
            json=payload,
            headers=connection.headers,
            timeout=timeout,
            allow_redirects=False,
            proxies={"http": "", "https": "", "all": ""},
            auth=_NoEnvironmentAuth(),
            stream=True,
        ) as response:
            if response.status_code != 200:
                raise ValueError("Remote control request was not accepted")
            body = bytearray()
            for chunk in response.iter_content(chunk_size=1024):
                if time.monotonic() > deadline:
                    raise ValueError("Remote control response timed out")
                body.extend(chunk)
                if len(body) > max_response_bytes:
                    raise ValueError("Remote control response exceeds its bound")
            result = json.loads(body)
            if not isinstance(result, dict):
                raise ValueError("Remote control response must be an object")
            return result
    except Exception:
        raise RuntimeError("Remote control request failed") from None
