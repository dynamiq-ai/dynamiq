import io
import re
from enum import Enum
from typing import Any

import requests


def resolve_http_client(client: Any | None) -> Any:
    """Pick the object used for HTTP calls.

    The connection manager hands adapters whatever the connection's ``connect()`` returned: the
    ``requests`` module for HTTP connections, an SDK client for OpenAI, or ``None`` for connections
    that only carry credentials. Anything exposing ``request`` that is not an OpenAI SDK client is
    used as-is, so a pre-configured ``requests.Session`` can be injected; otherwise ``requests``.
    """
    if client is not None and callable(getattr(client, "request", None)) and not hasattr(client, "audio"):
        return client
    return requests


def raise_for_status(response: requests.Response, provider: str) -> None:
    """Raise ``requests.HTTPError`` for failed calls, keeping the provider's error body in the message."""
    if response.status_code < 400:
        return
    detail = (response.text or "").strip()[:500] or response.reason
    raise requests.HTTPError(
        f"{provider} request failed with status {response.status_code}: {detail}", response=response
    )


def prepare_audio_file(audio: io.BytesIO | bytes, default_name: str, default_content_type: str) -> io.BytesIO:
    """Normalize node audio input to a named, typed ``BytesIO`` that provider clients accept."""
    if isinstance(audio, bytes):
        audio = io.BytesIO(audio)
    if not isinstance(audio, io.BytesIO):
        raise ValueError("Audio must be a BytesIO object or bytes.")
    if not getattr(audio, "name", None):
        audio.name = default_name
    if not getattr(audio, "content_type", None):
        audio.content_type = default_content_type
    audio.seek(0)
    return audio


def split_terms(prompt: str) -> list[str]:
    """Turn a free-text prompt into the term list used by providers with key-term boosting."""
    return [term.strip() for term in re.split(r"[,\n]", prompt) if term.strip()]


def format_param_value(value: Any) -> Any:
    """Render a parameter the way query strings and multipart forms expect it."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, (list, tuple)):
        return [format_param_value(item) for item in value]
    return value


def multipart_fields(data: dict[str, Any]) -> list[tuple[str, tuple[None, str]]]:
    """Flatten a dict into ``requests`` multipart tuples, repeating list values as separate fields."""
    fields: list[tuple[str, tuple[None, str]]] = []
    for key, value in data.items():
        if value is None:
            continue
        formatted = format_param_value(value)
        items = formatted if isinstance(formatted, list) else [formatted]
        fields.extend((key, (None, str(item))) for item in items)
    return fields
