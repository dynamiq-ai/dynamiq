from typing import Any

from dynamiq.components.audio.tts.base import BaseTTSAdapter, SpeechRequest, SpeechResult, TTSCapabilities
from dynamiq.components.audio.utils import format_param_value, raise_for_status, resolve_http_client
from dynamiq.types.audio import AudioFormat

# Deepgram splits a format into encoding + container; containerless encodings carry their own framing.
ENCODINGS: dict[AudioFormat, tuple[str, str | None]] = {
    AudioFormat.MP3: ("mp3", None),
    AudioFormat.WAV: ("linear16", "wav"),
    AudioFormat.PCM: ("linear16", "none"),
    AudioFormat.OPUS: ("opus", "ogg"),
    AudioFormat.FLAC: ("flac", None),
    AudioFormat.AAC: ("aac", None),
    AudioFormat.MULAW: ("mulaw", "none"),
    AudioFormat.ALAW: ("alaw", "none"),
}


class DeepgramTTSAdapter(BaseTTSAdapter):
    """Deepgram Aura text to speech (``/v1/speak``). The voice is the model name (``aura-2-thalia-en``)."""

    provider = "Deepgram"
    capabilities = TTSCapabilities(speed=True, sample_rate=True, max_characters=2000)
    default_model = "aura-2-thalia-en"

    def build_params(self, request: SpeechRequest) -> dict[str, Any]:
        encoding, container = ENCODINGS[request.output_format]
        params: dict[str, Any] = {"model": request.voice or request.model, "encoding": encoding}
        if container:
            params["container"] = container
        if request.sample_rate is not None:
            params["sample_rate"] = request.sample_rate
        if request.speed is not None:
            params["speed"] = request.speed
        params.update(request.provider_options)
        return {key: format_param_value(value) for key, value in params.items() if value is not None}

    def synthesize(self, request: SpeechRequest) -> SpeechResult:
        http = resolve_http_client(self.client)
        response = http.post(
            f"{self.connection.url.rstrip('/')}/speak",
            params=self.build_params(request),
            headers={"Authorization": f"Token {self.connection.api_key}", "Content-Type": "application/json"},
            json={"text": request.text},
        )
        raise_for_status(response, self.provider)
        return self.result(response.content, request.output_format)
