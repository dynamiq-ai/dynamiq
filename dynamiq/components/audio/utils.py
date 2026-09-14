import io
import mimetypes
import re
from enum import Enum
from typing import Any

import requests

from dynamiq.utils.logger import logger

# Providers take speech out of video containers too, so both count as a recording.
AUDIO_CONTENT_TYPE_PREFIXES = ("audio/", "video/")

# What a store writes when nobody told it the type. It says "bytes", not "not audio", so the file
# name gets the final word: the agent's own upload path stamps this on every file it stores.
GENERIC_CONTENT_TYPES = ("application/octet-stream", "binary/octet-stream")


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


def _looks_like_audio(file: Any) -> bool | None:
    """Whether a file is a recording. ``None`` when it carries nothing to judge by."""
    content_type = getattr(file, "content_type", None)
    if not content_type or content_type in GENERIC_CONTENT_TYPES:
        name = getattr(file, "name", None)
        content_type = (mimetypes.guess_type(name)[0] if name else None) or None
    if not content_type:
        return None
    return content_type.startswith(AUDIO_CONTENT_TYPE_PREFIXES)


def select_audio_file(audio: Any) -> Any:
    """Pick the recording out of whatever was handed to a transcription node.

    Agents inject every file their store holds, so the node is routinely given a contract and a
    call recording together and has to choose. A file that identifies itself as audio or video
    wins; raw bytes with nothing to judge by are taken as a last resort; a set of files that are
    all provably something else is an error rather than a transcription of a PDF.
    """
    if not isinstance(audio, (list, tuple)):
        return audio
    if not audio:
        return None

    unidentified = []
    for item in audio:
        verdict = _looks_like_audio(item)
        if verdict is True:
            if len(audio) > 1:
                logger.debug(f"Transcribing {getattr(item, 'name', 'the audio file')} out of {len(audio)} files.")
            return item
        if verdict is None:
            unidentified.append(item)

    if unidentified:
        return unidentified[0]

    names = ", ".join(str(getattr(item, "name", "unnamed file")) for item in audio)
    raise ValueError(f"Received {len(audio)} file(s) but none of them look like audio: {names}.")


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
