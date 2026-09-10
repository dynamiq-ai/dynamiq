from typing import Any

from dynamiq.components.audio.stt.base import BaseSTTAdapter, STTCapabilities, TranscriptionRequest
from dynamiq.components.audio.stt.normalize import build_transcript, optional_str
from dynamiq.components.audio.utils import multipart_fields, raise_for_status, resolve_http_client, split_terms
from dynamiq.types.audio import SpeakerHints, TimestampGranularity, Transcript, TranscriptSegment, TranscriptWord

MISTRAL_API_BASE_URL = "https://api.mistral.ai/v1"


class MistralSTTAdapter(BaseSTTAdapter):
    """Mistral Voxtral transcription (``/v1/audio/transcriptions``) with speaker diarization."""

    provider = "Mistral"
    capabilities = STTCapabilities(diarization=True, word_timestamps=True, prompt=True, audio_url_input=True)
    default_model = "voxtral-mini-latest"

    @classmethod
    def check_config(
        cls,
        *,
        diarize: bool,
        timestamps: TimestampGranularity,
        speakers: SpeakerHints | None,
        prompt: str | None,
        language: str | None = None,
    ) -> None:
        super().check_config(
            diarize=diarize, timestamps=timestamps, speakers=speakers, prompt=prompt, language=language
        )
        if diarize and timestamps == TimestampGranularity.WORD:
            raise ValueError("Mistral diarization returns segment timings only; use timestamps='segment' with diarize.")

    def build_fields(self, request: TranscriptionRequest) -> dict[str, Any]:
        fields: dict[str, Any] = {"model": request.model}
        if request.language:
            fields["language"] = request.language
        if request.diarize:
            fields["diarize"] = True
            # Diarization is rejected outright unless segment granularity comes with it, whatever the
            # node asked for: the timed segments are what carries the speaker labels.
            fields["timestamp_granularities"] = ["segment"]
        elif request.timestamps != TimestampGranularity.NONE:
            fields["timestamp_granularities"] = [request.timestamps.value]
        if request.prompt:
            fields["context_bias"] = split_terms(request.prompt)
        if request.audio_url:
            fields["file_url"] = request.audio_url
        fields.update(request.provider_options)
        return fields

    def transcribe(self, request: TranscriptionRequest) -> Transcript:
        http = resolve_http_client(self.client)
        files: list[tuple[str, Any]] = multipart_fields(self.build_fields(request))
        if request.audio is not None:
            audio = request.audio
            files.append(("file", (audio.name, audio, getattr(audio, "content_type", None))))
        response = http.post(
            f"{MISTRAL_API_BASE_URL}/audio/transcriptions",
            headers={"Authorization": f"Bearer {self.connection.api_key}"},
            files=files,
        )
        raise_for_status(response, self.provider)
        return self.normalize(response.json(), request)

    @staticmethod
    def normalize(data: dict[str, Any], request: TranscriptionRequest) -> Transcript:
        segments = [
            TranscriptSegment(
                id=str(index),
                text=segment.get("text", ""),
                start=segment.get("start"),
                end=segment.get("end"),
                speaker=optional_str(segment.get("speaker_id")),
                confidence=segment.get("score"),
            )
            for index, segment in enumerate(data.get("segments") or [])
        ]
        words = [
            TranscriptWord(
                word=word.get("word") or word.get("text") or "",
                start=word.get("start"),
                end=word.get("end"),
                speaker=optional_str(word.get("speaker_id")),
            )
            for word in data.get("words") or []
        ]
        usage = data.get("usage") or {}
        return build_transcript(
            content=data.get("text") or "",
            segments=segments,
            words=words,
            language=data.get("language") or request.language,
            duration=usage.get("prompt_audio_seconds"),
            usage=usage,
            raw=data,
        )
