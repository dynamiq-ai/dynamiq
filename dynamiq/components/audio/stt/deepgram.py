from typing import Any

from dynamiq.components.audio.stt.base import BaseSTTAdapter, STTCapabilities, TranscriptionRequest
from dynamiq.components.audio.stt.normalize import build_transcript, optional_str
from dynamiq.components.audio.utils import format_param_value, raise_for_status, resolve_http_client, split_terms
from dynamiq.types.audio import TimestampGranularity, Transcript, TranscriptSegment, TranscriptWord


class DeepgramSTTAdapter(BaseSTTAdapter):
    """Deepgram pre-recorded transcription (``/v1/listen``) with word-level diarization."""

    provider = "Deepgram"
    capabilities = STTCapabilities(diarization=True, word_timestamps=True, prompt=True, audio_url_input=True)
    default_model = "nova-3"

    def build_params(self, request: TranscriptionRequest) -> dict[str, Any]:
        params: dict[str, Any] = {"model": request.model, "smart_format": True}
        if request.language:
            params["language"] = request.language
        if request.diarize:
            params["diarize"] = True
        if request.diarize or request.timestamps != TimestampGranularity.NONE:
            # Utterances are the only timed segments Deepgram returns; without them word timing
            # would collapse the transcript into a single synthetic segment.
            params["utterances"] = True
        if request.prompt:
            params["keyterm"] = split_terms(request.prompt)
        params.update(request.provider_options)
        # diarize_model is Deepgram's newer switch; sending the legacy flag alongside it would pin v1.
        if "diarize_model" in params:
            params.pop("diarize", None)
        return {key: format_param_value(value) for key, value in params.items() if value is not None}

    def transcribe(self, request: TranscriptionRequest) -> Transcript:
        http = resolve_http_client(self.client)
        url = f"{self.connection.url.rstrip('/')}/listen"
        headers = {"Authorization": f"Token {self.connection.api_key}"}
        params = self.build_params(request)
        if request.audio_url:
            headers["Content-Type"] = "application/json"
            response = http.post(url, params=params, headers=headers, json={"url": request.audio_url})
        else:
            headers["Content-Type"] = getattr(request.audio, "content_type", None) or "application/octet-stream"
            response = http.post(url, params=params, headers=headers, data=request.audio.getvalue())
        raise_for_status(response, self.provider)
        return self.normalize(response.json(), request)

    @staticmethod
    def normalize(data: dict[str, Any], request: TranscriptionRequest) -> Transcript:
        results = data.get("results") or {}
        channels = results.get("channels") or []
        channel = channels[0] if channels else {}
        alternatives = channel.get("alternatives") or []
        alternative = alternatives[0] if alternatives else {}

        words = [
            TranscriptWord(
                word=word.get("punctuated_word") or word.get("word") or "",
                start=word.get("start"),
                end=word.get("end"),
                speaker=optional_str(word.get("speaker")),
                confidence=word.get("confidence"),
            )
            for word in alternative.get("words") or []
        ]
        segments = [
            TranscriptSegment(
                id=str(utterance.get("id") or index),
                text=utterance.get("transcript", ""),
                start=utterance.get("start"),
                end=utterance.get("end"),
                speaker=optional_str(utterance.get("speaker")),
                confidence=utterance.get("confidence"),
            )
            for index, utterance in enumerate(results.get("utterances") or [])
        ]
        metadata = data.get("metadata") or {}
        duration = metadata.get("duration")
        detected = channel.get("detected_language")
        requested = request.language if request.language and request.language != "multi" else None
        return build_transcript(
            content=alternative.get("transcript") or "",
            segments=segments,
            words=words,
            language=detected or requested,
            languages=list(channel.get("languages") or []),
            duration=duration,
            usage={"seconds": duration} if duration is not None else {},
            raw=data,
        )
