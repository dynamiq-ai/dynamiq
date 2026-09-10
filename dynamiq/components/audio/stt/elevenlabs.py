from typing import Any

from dynamiq.components.audio.stt.base import BaseSTTAdapter, STTCapabilities, TranscriptionRequest
from dynamiq.components.audio.stt.normalize import build_transcript, optional_str
from dynamiq.components.audio.utils import format_param_value, raise_for_status, resolve_http_client
from dynamiq.types.audio import TimestampGranularity, Transcript, TranscriptWord


def elevenlabs_api_base(url: str) -> str:
    """Base API URL from a connection URL that may point at a specific endpoint (legacy configs)."""
    marker = "/v1"
    if marker in url:
        return url[: url.index(marker) + len(marker)]
    return url.rstrip("/")


class ElevenLabsSTTAdapter(BaseSTTAdapter):
    """ElevenLabs Scribe (``/v1/speech-to-text``) with word-level speaker ids."""

    provider = "ElevenLabs"
    capabilities = STTCapabilities(diarization=True, speaker_hints=True, word_timestamps=True)
    default_model = "scribe_v2"

    def build_fields(self, request: TranscriptionRequest) -> dict[str, Any]:
        fields: dict[str, Any] = {"model_id": request.model}
        if request.language:
            fields["language_code"] = request.language
        if request.diarize:
            fields["diarize"] = True
            hints = request.speakers
            speakers = (hints.expected or hints.max) if hints else None
            if speakers:
                fields["num_speakers"] = speakers
        if request.timestamps != TimestampGranularity.NONE:
            fields["timestamps_granularity"] = "word"
        fields.update(request.provider_options)
        fields.update(self.connection.data or {})
        return {key: format_param_value(value) for key, value in fields.items() if value is not None}

    def transcribe(self, request: TranscriptionRequest) -> Transcript:
        http = resolve_http_client(self.client)
        audio = request.audio
        response = http.post(
            f"{elevenlabs_api_base(self.connection.url)}/speech-to-text",
            headers=self.connection.headers,
            data=self.build_fields(request),
            files={"file": (audio.name, audio, getattr(audio, "content_type", None))},
        )
        raise_for_status(response, self.provider)
        return self.normalize(response.json(), request)

    @staticmethod
    def normalize(data: dict[str, Any], request: TranscriptionRequest) -> Transcript:
        words: list[TranscriptWord] = []
        for word in data.get("words") or []:
            kind = word.get("type") or "word"
            if kind == "spacing":
                continue
            words.append(
                TranscriptWord(
                    word=word.get("text") or "",
                    start=word.get("start"),
                    end=word.get("end"),
                    speaker=optional_str(word.get("speaker_id")),
                    type="audio_event" if kind == "audio_event" else "word",
                )
            )
        return build_transcript(
            content=data.get("text") or "",
            words=words,
            language=data.get("language_code") or request.language,
            raw=data,
        )
