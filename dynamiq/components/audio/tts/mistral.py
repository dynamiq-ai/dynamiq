import base64
from typing import Any

from dynamiq.components.audio.stt.mistral import MISTRAL_API_BASE_URL
from dynamiq.components.audio.tts.base import BaseTTSAdapter, SpeechRequest, SpeechResult, TTSCapabilities
from dynamiq.components.audio.utils import raise_for_status, resolve_http_client
from dynamiq.types.audio import AudioFormat


class MistralTTSAdapter(BaseTTSAdapter):
    """Mistral Voxtral text to speech (``/v1/audio/speech``); audio arrives base64-encoded."""

    provider = "Mistral"
    capabilities = TTSCapabilities(
        formats={AudioFormat.MP3, AudioFormat.WAV, AudioFormat.PCM, AudioFormat.FLAC, AudioFormat.OPUS}
    )
    default_model = "voxtral-mini-tts-latest"
    # Voxtral has no implicit voice: a request without one is rejected, so the node falls back to a
    # global preset. Any preset slug, a cloned voice id, or a `ref_audio` provider option works too.
    default_voice = "en_paul_neutral"

    def build_body(self, request: SpeechRequest) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": request.model,
            "input": request.text,
            "response_format": request.output_format.value,
        }
        if request.voice:
            body["voice_id"] = request.voice
        elif "ref_audio" not in request.provider_options and "voice_id" not in request.provider_options:
            body["voice_id"] = self.default_voice
        body.update(request.provider_options)
        return body

    def synthesize(self, request: SpeechRequest) -> SpeechResult:
        http = resolve_http_client(self.client)
        response = http.post(
            f"{MISTRAL_API_BASE_URL}/audio/speech",
            headers={"Authorization": f"Bearer {self.connection.api_key}", "Content-Type": "application/json"},
            json=self.build_body(request),
        )
        raise_for_status(response, self.provider)
        encoded = (response.json() or {}).get("audio_data")
        if not encoded:
            raise ValueError("Mistral returned no audio_data in the speech response.")
        return self.result(base64.b64decode(encoded), request.output_format)
