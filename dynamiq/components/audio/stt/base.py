import io
from abc import ABC, abstractmethod
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict, Field

from dynamiq.connections import BaseConnection
from dynamiq.types.audio import SpeakerHints, TimestampGranularity, Transcript
from dynamiq.utils.logger import logger


class STTCapabilities(BaseModel):
    """What a transcription provider can do. Checked before any request is sent."""

    diarization: bool = False
    speaker_hints: bool = False
    word_timestamps: bool = False
    prompt: bool = False
    audio_url_input: bool = False


class TranscriptionRequest(BaseModel):
    """Provider-neutral transcription request built by the SpeechToText node."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    audio: io.BytesIO | None = None
    audio_url: str | None = None
    model: str
    language: str | None = None
    diarize: bool = False
    timestamps: TimestampGranularity = TimestampGranularity.SEGMENT
    speakers: SpeakerHints | None = None
    prompt: str | None = None
    provider_options: dict[str, Any] = Field(default_factory=dict)


class BaseSTTAdapter(ABC):
    """Translates a ``TranscriptionRequest`` into one provider's API call and back into a ``Transcript``.

    Attributes:
        provider (str): Human-readable provider name used in error messages.
        capabilities (STTCapabilities): Features the provider supports.
        default_model (str): Model used when the node does not set one.
    """

    provider: ClassVar[str]
    capabilities: ClassVar[STTCapabilities] = STTCapabilities()
    default_model: ClassVar[str]

    def __init__(self, connection: BaseConnection | None, client: Any | None = None):
        self.connection = connection
        self.client = client

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
        """Reject node configuration the provider cannot honour, before any audio is sent."""
        capabilities = cls.capabilities
        if diarize and not capabilities.diarization:
            raise ValueError(f"{cls.provider} does not support speaker diarization.")
        if timestamps == TimestampGranularity.WORD and not capabilities.word_timestamps:
            raise ValueError(f"{cls.provider} does not support word-level timestamps.")
        if prompt and not capabilities.prompt:
            raise ValueError(f"{cls.provider} does not accept a transcription prompt.")
        if speakers is not None and diarize and not capabilities.speaker_hints:
            logger.warning(f"{cls.provider} ignores speaker hints; diarization runs without them.")

    def check_request(self, request: TranscriptionRequest) -> None:
        """Reject per-run input the provider cannot accept."""
        if request.audio_url and not self.capabilities.audio_url_input:
            raise ValueError(f"{self.provider} does not accept audio by URL; pass the audio file instead.")
        if request.audio is None and not request.audio_url:
            raise ValueError("Either audio or audio_url is required.")

    @abstractmethod
    def transcribe(self, request: TranscriptionRequest) -> Transcript:
        """Run the transcription and return the normalized transcript."""
        raise NotImplementedError
