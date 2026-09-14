from typing import Any

from dynamiq.components.audio.stt.elevenlabs import elevenlabs_api_base
from dynamiq.components.audio.tts.base import BaseTTSAdapter, SpeechRequest, SpeechResult, TTSCapabilities
from dynamiq.components.audio.utils import raise_for_status, resolve_http_client
from dynamiq.types.audio import AudioFormat

# ElevenLabs names formats as codec_rate[_bitrate]; the node speaks in codec + sample rate.
OUTPUT_FORMATS: dict[AudioFormat, dict[int, str]] = {
    AudioFormat.MP3: {22050: "mp3_22050_32", 24000: "mp3_24000_48", 44100: "mp3_44100_128"},
    AudioFormat.PCM: {rate: f"pcm_{rate}" for rate in (8000, 16000, 22050, 24000, 32000, 44100, 48000)},
    AudioFormat.WAV: {rate: f"wav_{rate}" for rate in (8000, 16000, 22050, 24000, 32000, 44100, 48000)},
    AudioFormat.OPUS: {48000: "opus_48000_128"},
    AudioFormat.MULAW: {8000: "ulaw_8000"},
    AudioFormat.ALAW: {8000: "alaw_8000"},
}
DEFAULT_SAMPLE_RATES: dict[AudioFormat, int] = {
    AudioFormat.MP3: 44100,
    AudioFormat.PCM: 24000,
    AudioFormat.WAV: 44100,
    AudioFormat.OPUS: 48000,
    AudioFormat.MULAW: 8000,
    AudioFormat.ALAW: 8000,
}


def elevenlabs_output_format(output_format: AudioFormat, sample_rate: int | None) -> str:
    rates = OUTPUT_FORMATS[output_format]
    rate = sample_rate or DEFAULT_SAMPLE_RATES[output_format]
    if rate not in rates:
        supported = ", ".join(str(item) for item in sorted(rates))
        raise ValueError(f"ElevenLabs {output_format.value} supports sample rates {supported}; got {rate}.")
    return rates[rate]


class ElevenLabsTTSAdapter(BaseTTSAdapter):
    """ElevenLabs text to speech (``/v1/text-to-speech/{voice_id}``)."""

    provider = "ElevenLabs"
    capabilities = TTSCapabilities(formats=set(OUTPUT_FORMATS), speed=True, language=True, sample_rate=True)
    default_model = "eleven_multilingual_v2"
    default_voice = "21m00Tcm4TlvDq8ikWAM"

    def build_body(self, request: SpeechRequest) -> dict[str, Any]:
        options = dict(request.provider_options)
        voice_settings = dict(options.pop("voice_settings", None) or {})
        if request.speed is not None:
            voice_settings["speed"] = request.speed
        body: dict[str, Any] = {"text": request.text, "model_id": request.model}
        if request.language:
            body["language_code"] = request.language
        if voice_settings:
            body["voice_settings"] = voice_settings
        body.update(options)
        body.update(self.connection.data or {})
        return body

    def synthesize(self, request: SpeechRequest) -> SpeechResult:
        http = resolve_http_client(self.client)
        voice = request.voice or self.default_voice
        response = http.post(
            f"{elevenlabs_api_base(self.connection.url)}/text-to-speech/{voice}",
            params={"output_format": elevenlabs_output_format(request.output_format, request.sample_rate)},
            headers=self.connection.headers,
            json=self.build_body(request),
        )
        raise_for_status(response, self.provider)
        return self.result(response.content, request.output_format)
