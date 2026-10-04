from typing import Any

from dynamiq.components.audio.tts.base import BaseTTSAdapter, SpeechRequest, SpeechResult, TTSCapabilities
from dynamiq.components.audio.utils import raise_for_status, resolve_http_client
from dynamiq.types.audio import AudioFormat

# MiniMax takes the codec in audio_setting.format; the node's names match for the formats it shares.
AUDIO_FORMATS: dict[AudioFormat, str] = {
    AudioFormat.MP3: "mp3",
    AudioFormat.WAV: "wav",
    AudioFormat.PCM: "pcm",
    AudioFormat.FLAC: "flac",
}
SAMPLE_RATES = (8000, 16000, 22050, 24000, 32000, 44100)
MINIMAX_SUCCESS = 0


class MiniMaxTTSAdapter(BaseTTSAdapter):
    """MiniMax text to speech (``/v1/t2a_v2``), non-streaming.

    Audio arrives hex-encoded in ``data.audio``, or as a download link when ``output_format`` is set
    to ``url`` in the provider options. MiniMax reports failures in ``base_resp`` with HTTP 200, so
    the status code there is checked on every response.
    """

    provider = "MiniMax"
    capabilities = TTSCapabilities(formats=set(AUDIO_FORMATS), speed=True, sample_rate=True, max_characters=9999)
    default_model = "speech-2.8-hd"
    # MiniMax rejects a request without a voice, so the node falls back to a system voice.
    default_voice = "English_expressive_narrator"

    def build_body(self, request: SpeechRequest) -> dict[str, Any]:
        if request.sample_rate is not None and request.sample_rate not in SAMPLE_RATES:
            supported = ", ".join(str(rate) for rate in SAMPLE_RATES)
            raise ValueError(f"MiniMax supports sample rates {supported}; got {request.sample_rate}.")

        options = dict(request.provider_options)
        voice_setting = dict(options.pop("voice_setting", None) or {})
        if request.voice or ("voice_id" not in voice_setting and "timbre_weights" not in options):
            voice_setting["voice_id"] = request.voice or self.default_voice
        if request.speed is not None:
            voice_setting["speed"] = request.speed

        audio_setting = dict(options.pop("audio_setting", None) or {})
        audio_setting["format"] = AUDIO_FORMATS[request.output_format]
        if request.sample_rate is not None:
            audio_setting["sample_rate"] = request.sample_rate

        body: dict[str, Any] = {
            "model": request.model,
            "text": request.text,
            "stream": False,
            "voice_setting": voice_setting,
            "audio_setting": audio_setting,
        }
        body.update(options)
        body.update(self.connection.data or {})
        # Streaming responses are server-sent events, which this adapter does not read.
        body["stream"] = False
        return body

    def synthesize(self, request: SpeechRequest) -> SpeechResult:
        http = resolve_http_client(self.client)
        body = self.build_body(request)
        response = http.post(
            f"{self.connection.url.rstrip('/')}/t2a_v2",
            headers=self.connection.headers,
            json=body,
        )
        raise_for_status(response, self.provider)

        payload = response.json() or {}
        base_resp = payload.get("base_resp") or {}
        status_code = base_resp.get("status_code")
        if status_code != MINIMAX_SUCCESS:
            message = base_resp.get("status_msg") or "no status message"
            raise ValueError(f"MiniMax speech synthesis failed with status {status_code}: {message}")

        audio = (payload.get("data") or {}).get("audio")
        if not isinstance(audio, str) or not audio:
            raise ValueError("MiniMax returned no audio in the speech response.")

        if body.get("output_format") == "url":
            download = http.get(audio)
            raise_for_status(download, self.provider)
            return self.result(download.content, request.output_format)

        try:
            return self.result(bytes.fromhex(audio), request.output_format)
        except ValueError as error:
            raise ValueError("MiniMax returned audio that is not valid hex.") from error
