import enum
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


class AudioFormat(str, enum.Enum):
    """Audio encodings a text-to-speech node can request.

    Adapters translate these into the provider's own format names, so one node configuration
    works across providers.
    """

    MP3 = "mp3"
    WAV = "wav"
    PCM = "pcm"
    OPUS = "opus"
    FLAC = "flac"
    AAC = "aac"
    MULAW = "mulaw"
    ALAW = "alaw"


AUDIO_FORMAT_MIME_TYPES: dict[AudioFormat, str] = {
    AudioFormat.MP3: "audio/mpeg",
    AudioFormat.WAV: "audio/wav",
    AudioFormat.PCM: "audio/pcm",
    AudioFormat.OPUS: "audio/ogg",
    AudioFormat.FLAC: "audio/flac",
    AudioFormat.AAC: "audio/aac",
    AudioFormat.MULAW: "audio/basic",
    AudioFormat.ALAW: "audio/alaw",
}

AUDIO_FORMAT_EXTENSIONS: dict[AudioFormat, str] = {
    AudioFormat.MP3: "mp3",
    AudioFormat.WAV: "wav",
    AudioFormat.PCM: "pcm",
    AudioFormat.OPUS: "opus",
    AudioFormat.FLAC: "flac",
    AudioFormat.AAC: "aac",
    AudioFormat.MULAW: "ulaw",
    AudioFormat.ALAW: "alaw",
}


class TimestampGranularity(str, enum.Enum):
    """How much timing detail a transcription should carry."""

    NONE = "none"
    SEGMENT = "segment"
    WORD = "word"


class SpeakerHints(BaseModel):
    """Guidance for speaker diarization.

    Each provider adapter maps the hints its API accepts and ignores the rest, so the same
    configuration is valid for every provider.
    """

    expected: int | None = Field(default=None, ge=1, description="Exact number of speakers, when known.")
    min: int | None = Field(default=None, ge=1, description="Lower bound on the number of speakers.")
    max: int | None = Field(default=None, ge=1, description="Upper bound on the number of speakers.")

    @model_validator(mode="after")
    def validate_bounds(self):
        if self.min is not None and self.max is not None and self.min > self.max:
            raise ValueError("Speaker hint 'min' must not exceed 'max'.")
        return self


class TranscriptWord(BaseModel):
    """A single word (or punctuation mark / audio event) with optional timing and speaker."""

    word: str
    start: float | None = None
    end: float | None = None
    speaker: str | None = None
    confidence: float | None = None
    type: Literal["word", "punctuation", "audio_event"] = "word"


class TranscriptSegment(BaseModel):
    """A contiguous span of speech. Times are seconds from the start of the audio."""

    id: str
    text: str
    start: float | None = None
    end: float | None = None
    speaker: str | None = None
    confidence: float | None = None
    language: str | None = None
    synthetic: bool = Field(
        default=False,
        description="True when the segment was rebuilt from word-level output instead of returned by the provider.",
    )


class Speaker(BaseModel):
    """A speaker detected by diarization. ``id`` is the provider's own label."""

    id: str
    label: str


class Transcript(BaseModel):
    """Provider-neutral transcription result.

    ``content`` is the plain text every provider returns. ``transcript`` is the same text rendered
    with speaker labels when diarization produced them. ``segments`` and ``words`` carry timing and
    speakers; ``words`` is empty when the provider returns none.
    """

    content: str = Field(description="Plain transcript text.")
    transcript: str = Field(description="Speaker-labelled transcript; equals content without diarization.")
    language: str | None = None
    languages: list[str] = Field(default_factory=list)
    duration: float | None = Field(default=None, description="Audio duration in seconds.")
    speakers: list[Speaker] = Field(default_factory=list)
    segments: list[TranscriptSegment] = Field(default_factory=list)
    words: list[TranscriptWord] = Field(default_factory=list)
    usage: dict[str, Any] = Field(default_factory=dict)
    raw: dict[str, Any] = Field(default_factory=dict, description="Untouched provider response.")
