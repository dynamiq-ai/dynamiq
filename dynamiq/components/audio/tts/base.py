from abc import ABC, abstractmethod
from typing import Any, ClassVar

from pydantic import BaseModel, Field

from dynamiq.connections import BaseConnection
from dynamiq.types.audio import AUDIO_FORMAT_EXTENSIONS, AUDIO_FORMAT_MIME_TYPES, AudioFormat


class TTSCapabilities(BaseModel):
    """What a speech provider can do. Checked before any request is sent."""

    formats: set[AudioFormat] = Field(default_factory=lambda: set(AudioFormat))
    speed: bool = False
    instructions: bool = False
    language: bool = False
    sample_rate: bool = False
    max_characters: int | None = None


class SpeechRequest(BaseModel):
    """Provider-neutral speech request built by the TextToSpeech node."""

    text: str
    model: str
    voice: str | None = None
    language: str | None = None
    speed: float | None = None
    instructions: str | None = None
    output_format: AudioFormat = AudioFormat.MP3
    sample_rate: int | None = None
    provider_options: dict[str, Any] = Field(default_factory=dict)


class SpeechResult(BaseModel):
    """Synthesized audio plus the metadata needed to name and serve the file."""

    audio: bytes
    mime_type: str
    extension: str


class BaseTTSAdapter(ABC):
    """Translates a ``SpeechRequest`` into one provider's API call and returns the audio bytes."""

    provider: ClassVar[str]
    capabilities: ClassVar[TTSCapabilities] = TTSCapabilities()
    default_model: ClassVar[str]
    default_voice: ClassVar[str | None] = None

    def __init__(self, connection: BaseConnection | None, client: Any | None = None):
        self.connection = connection
        self.client = client

    @classmethod
    def check_config(
        cls,
        *,
        output_format: AudioFormat,
        speed: float | None,
        instructions: str | None,
        language: str | None,
        sample_rate: int | None,
    ) -> None:
        """Reject node configuration the provider cannot honour, before any text is sent."""
        capabilities = cls.capabilities
        if output_format not in capabilities.formats:
            supported = ", ".join(sorted(item.value for item in capabilities.formats))
            raise ValueError(f"{cls.provider} cannot produce '{output_format.value}' audio; supported: {supported}.")
        if speed is not None and not capabilities.speed:
            raise ValueError(f"{cls.provider} does not support a speed setting.")
        if instructions and not capabilities.instructions:
            raise ValueError(f"{cls.provider} does not accept voice instructions.")
        if language and not capabilities.language:
            raise ValueError(f"{cls.provider} does not accept a language setting; the voice defines it.")
        if sample_rate is not None and not capabilities.sample_rate:
            raise ValueError(f"{cls.provider} does not accept a sample rate.")

    def check_request(self, request: SpeechRequest) -> None:
        """Reject per-run input the provider cannot accept."""
        limit = self.capabilities.max_characters
        if limit is not None and len(request.text) > limit:
            raise ValueError(
                f"{self.provider} accepts at most {limit} characters per request; got {len(request.text)}."
            )

    @staticmethod
    def result(audio: bytes, output_format: AudioFormat) -> SpeechResult:
        return SpeechResult(
            audio=audio,
            mime_type=AUDIO_FORMAT_MIME_TYPES[output_format],
            extension=AUDIO_FORMAT_EXTENSIONS[output_format],
        )

    @abstractmethod
    def synthesize(self, request: SpeechRequest) -> SpeechResult:
        """Run the synthesis and return the audio."""
        raise NotImplementedError
