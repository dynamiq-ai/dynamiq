from typing import Any

from dynamiq.components.audio.openai_client import resolve_openai_client
from dynamiq.components.audio.tts.base import BaseTTSAdapter, SpeechRequest, SpeechResult, TTSCapabilities
from dynamiq.types.audio import AudioFormat

OPENAI_FORMATS = {
    AudioFormat.MP3,
    AudioFormat.OPUS,
    AudioFormat.AAC,
    AudioFormat.FLAC,
    AudioFormat.WAV,
    AudioFormat.PCM,
}


class OpenAICompatibleTTSAdapter(BaseTTSAdapter):
    """Any server that speaks the OpenAI ``/audio/speech`` contract (Kokoro, Speaches, vLLM-Omni)."""

    provider = "OpenAI-compatible endpoint"
    capabilities = TTSCapabilities(formats=OPENAI_FORMATS, speed=True, instructions=True)
    default_model = "tts-1"
    default_voice = "alloy"

    def build_params(self, request: SpeechRequest) -> dict[str, Any]:
        params: dict[str, Any] = {
            "model": request.model,
            "voice": request.voice or self.default_voice,
            "input": request.text,
            "response_format": request.output_format.value,
        }
        if request.speed is not None:
            params["speed"] = request.speed
        if request.instructions:
            params["instructions"] = request.instructions
        params.update(request.provider_options)
        return params

    def synthesize(self, request: SpeechRequest) -> SpeechResult:
        client = resolve_openai_client(self.connection, self.client)
        response = client.audio.speech.create(**self.build_params(request))
        return self.result(response.content, request.output_format)


class OpenAITTSAdapter(OpenAICompatibleTTSAdapter):
    """OpenAI speech models; ``instructions`` steer tone on ``gpt-4o-mini-tts``."""

    provider = "OpenAI"
    capabilities = TTSCapabilities(formats=OPENAI_FORMATS, speed=True, instructions=True, max_characters=4096)
    default_model = "gpt-4o-mini-tts"


class GroqTTSAdapter(OpenAICompatibleTTSAdapter):
    """Groq-hosted Orpheus voices through the OpenAI speech contract (WAV only, short inputs)."""

    provider = "Groq"
    capabilities = TTSCapabilities(formats={AudioFormat.WAV}, max_characters=200)
    default_model = "canopylabs/orpheus-v1-english"
    default_voice = "autumn"
