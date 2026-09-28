from typing import Any

from dynamiq.components.audio.stt import (
    BaseSTTAdapter,
    DeepgramSTTAdapter,
    ElevenLabsSTTAdapter,
    GroqSTTAdapter,
    MistralSTTAdapter,
    OpenAICompatibleSTTAdapter,
    OpenAISTTAdapter,
)
from dynamiq.components.audio.tts import (
    BaseTTSAdapter,
    DeepgramTTSAdapter,
    ElevenLabsTTSAdapter,
    GroqTTSAdapter,
    MistralTTSAdapter,
    OpenAICompatibleTTSAdapter,
    OpenAITTSAdapter,
)
from dynamiq.connections import BaseConnection
from dynamiq.connections import Deepgram as DeepgramConnection
from dynamiq.connections import ElevenLabs as ElevenLabsConnection
from dynamiq.connections import Groq as GroqConnection
from dynamiq.connections import HttpApiKey as HttpApiKeyConnection
from dynamiq.connections import Mistral as MistralConnection
from dynamiq.connections import OpenAI as OpenAIConnection
from dynamiq.connections import Whisper as WhisperConnection

# The provider is the connection type: adding a provider means one adapter and one entry here.
STT_ADAPTERS: dict[type[BaseConnection], type[BaseSTTAdapter]] = {
    OpenAIConnection: OpenAISTTAdapter,
    GroqConnection: GroqSTTAdapter,
    HttpApiKeyConnection: OpenAICompatibleSTTAdapter,
    WhisperConnection: OpenAICompatibleSTTAdapter,
    MistralConnection: MistralSTTAdapter,
    DeepgramConnection: DeepgramSTTAdapter,
    ElevenLabsConnection: ElevenLabsSTTAdapter,
}

TTS_ADAPTERS: dict[type[BaseConnection], type[BaseTTSAdapter]] = {
    OpenAIConnection: OpenAITTSAdapter,
    GroqConnection: GroqTTSAdapter,
    HttpApiKeyConnection: OpenAICompatibleTTSAdapter,
    WhisperConnection: OpenAICompatibleTTSAdapter,
    ElevenLabsConnection: ElevenLabsTTSAdapter,
    MistralConnection: MistralTTSAdapter,
    DeepgramConnection: DeepgramTTSAdapter,
}


def _resolve(registry: dict[type[BaseConnection], Any], connection: BaseConnection | None, client: Any, kind: str):
    if connection is None:
        # A bare OpenAI SDK client is the only client-only configuration the audio nodes accept.
        if client is not None and hasattr(client, "audio"):
            return registry[OpenAIConnection]
        raise ValueError(f"A connection is required to pick the {kind} provider.")
    # Walk the MRO so subclasses of a registered connection (MistralOCR, Dynamiq) map to the same adapter.
    for base in type(connection).__mro__:
        if base in registry:
            return registry[base]
    supported = ", ".join(sorted(item.__name__ for item in registry))
    raise ValueError(
        f"Connection {type(connection).__name__} is not supported for {kind}. Supported connections: {supported}."
    )


def resolve_stt_adapter(connection: BaseConnection | None, client: Any = None) -> type[BaseSTTAdapter]:
    return _resolve(STT_ADAPTERS, connection, client, "speech-to-text")


def resolve_tts_adapter(connection: BaseConnection | None, client: Any = None) -> type[BaseTTSAdapter]:
    return _resolve(TTS_ADAPTERS, connection, client, "text-to-speech")
