from typing import Any

from dynamiq.connections import BaseConnection
from dynamiq.connections import Groq as GroqConnection
from dynamiq.connections import HttpApiKey as HttpApiKeyConnection
from dynamiq.connections import OpenAI as OpenAIConnection
from dynamiq.connections import Whisper as WhisperConnection

GROQ_OPENAI_BASE_URL = "https://api.groq.com/openai/v1"


def resolve_openai_client(connection: BaseConnection | None, client: Any | None) -> Any:
    """Return an OpenAI SDK client for any connection that speaks the OpenAI audio contract.

    The SDK is used for the whole OpenAI-compatible family (OpenAI, Groq, self-hosted servers behind
    ``HttpApiKey`` or ``Whisper`` connections) because it already handles multipart uploads, retries
    and every transcription parameter, including diarization.
    """
    if client is not None and hasattr(client, "audio"):
        return client

    # Imported lazily: the SDK is only needed by this provider family.
    from openai import OpenAI as OpenAIClient

    if isinstance(connection, OpenAIConnection):
        return OpenAIClient(api_key=connection.api_key, base_url=connection.url)
    if isinstance(connection, GroqConnection):
        return OpenAIClient(api_key=connection.api_key, base_url=GROQ_OPENAI_BASE_URL)
    if isinstance(connection, (HttpApiKeyConnection, WhisperConnection)):
        # The SDK sets the bearer header itself; any other custom headers are kept.
        headers = {
            key: value
            for key, value in (getattr(connection, "headers", None) or {}).items()
            if key.lower() != "authorization"
        }
        return OpenAIClient(api_key=connection.api_key, base_url=connection.url, default_headers=headers or None)
    raise ValueError(f"Connection {type(connection).__name__} cannot be used with an OpenAI-compatible audio API.")
