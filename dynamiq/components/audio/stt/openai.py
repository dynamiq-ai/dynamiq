from typing import Any

from dynamiq.components.audio.openai_client import resolve_openai_client
from dynamiq.components.audio.stt.base import BaseSTTAdapter, STTCapabilities, TranscriptionRequest
from dynamiq.components.audio.stt.normalize import build_transcript, optional_str
from dynamiq.types.audio import TimestampGranularity, Transcript, TranscriptSegment, TranscriptWord
from dynamiq.utils.logger import logger

# The gpt-4o-transcribe family only returns plain JSON; verbose_json (segments, words) is a Whisper feature.
JSON_ONLY_MODEL_MARKER = "transcribe"


def normalize_openai_response(data: dict[str, Any], request: TranscriptionRequest) -> Transcript:
    """Map ``json``, ``verbose_json`` and ``diarized_json`` responses to a ``Transcript``."""
    segments = [
        TranscriptSegment(
            id=str(segment.get("id", index)),
            text=segment.get("text", ""),
            start=segment.get("start"),
            end=segment.get("end"),
            speaker=optional_str(segment.get("speaker")),
        )
        for index, segment in enumerate(data.get("segments") or [])
    ]
    words = [
        TranscriptWord(word=word.get("word", ""), start=word.get("start"), end=word.get("end"))
        for word in data.get("words") or []
    ]
    languages = [item["code"] for item in data.get("languages") or [] if isinstance(item, dict) and item.get("code")]
    usage = data.get("usage") or {}
    duration = data.get("duration")
    if duration is None and usage.get("type") == "duration":
        duration = usage.get("seconds")
    return build_transcript(
        content=data.get("text") or "",
        segments=segments,
        words=words,
        language=data.get("language") or request.language,
        languages=languages,
        duration=duration,
        usage=usage,
        raw=data,
    )


class OpenAICompatibleSTTAdapter(BaseSTTAdapter):
    """Any server that speaks the OpenAI ``/audio/transcriptions`` contract: Groq, vLLM, Speaches, NIM."""

    provider = "OpenAI-compatible endpoint"
    capabilities = STTCapabilities(word_timestamps=True, prompt=True)
    default_model = "whisper-1"

    def build_params(self, request: TranscriptionRequest) -> dict[str, Any]:
        audio = request.audio
        params: dict[str, Any] = {
            "model": request.model,
            "file": (audio.name, audio, getattr(audio, "content_type", None)),
        }
        if request.language:
            params["language"] = request.language
        if request.prompt:
            params["prompt"] = request.prompt
        if request.diarize:
            params["response_format"] = "diarized_json"
            params["chunking_strategy"] = "auto"
        elif request.timestamps != TimestampGranularity.NONE and JSON_ONLY_MODEL_MARKER not in request.model:
            params["response_format"] = "verbose_json"
            params["timestamp_granularities"] = (
                ["word", "segment"] if request.timestamps == TimestampGranularity.WORD else ["segment"]
            )
        else:
            params["response_format"] = "json"
        params.update(request.provider_options)
        return params

    def transcribe(self, request: TranscriptionRequest) -> Transcript:
        client = resolve_openai_client(self.connection, self.client)
        response = client.audio.transcriptions.create(**self.build_params(request))
        data = response.model_dump() if hasattr(response, "model_dump") else dict(response)
        return normalize_openai_response(data, request)


class OpenAISTTAdapter(OpenAICompatibleSTTAdapter):
    """OpenAI transcription models, including ``gpt-4o-transcribe-diarize`` for speaker diarization."""

    provider = "OpenAI"
    capabilities = STTCapabilities(diarization=True, word_timestamps=True, prompt=True)
    default_model = "gpt-4o-transcribe"
    DIARIZATION_MODEL = "gpt-4o-transcribe-diarize"

    def build_params(self, request: TranscriptionRequest) -> dict[str, Any]:
        if request.diarize and self.DIARIZATION_MODEL not in request.model:
            raise ValueError(f"OpenAI diarization requires model '{self.DIARIZATION_MODEL}', got '{request.model}'.")
        if request.timestamps == TimestampGranularity.WORD and JSON_ONLY_MODEL_MARKER in request.model:
            raise ValueError(
                f"Word timestamps are only returned by whisper-1 on OpenAI; '{request.model}' returns text only."
            )
        if request.prompt and self.DIARIZATION_MODEL in request.model:
            logger.warning(f"{self.DIARIZATION_MODEL} does not accept a prompt; it is ignored for this run.")
            request = request.model_copy(update={"prompt": None})
        return super().build_params(request)


class GroqSTTAdapter(OpenAICompatibleSTTAdapter):
    """Groq's OpenAI-compatible transcription endpoint; only the model catalogue differs."""

    provider = "Groq"
    default_model = "whisper-large-v3-turbo"
