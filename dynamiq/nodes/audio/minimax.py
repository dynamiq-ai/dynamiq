import enum
import io
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, Field

from dynamiq.connections import MiniMax as MiniMaxConnection
from dynamiq.nodes import ErrorHandling
from dynamiq.nodes.exceptions import NodeFailedException
from dynamiq.nodes.node import ConnectionNode, NodeGroup, ensure_config
from dynamiq.runnables import RunnableConfig
from dynamiq.types.cancellation import check_cancellation


class MiniMaxSpeechModel(str, enum.Enum):
    """MiniMax speech synthesis models."""

    SPEECH_2_8_HD = "speech-2.8-hd"
    SPEECH_2_8_TURBO = "speech-2.8-turbo"
    SPEECH_2_6_HD = "speech-2.6-hd"
    SPEECH_2_6_TURBO = "speech-2.6-turbo"
    SPEECH_02_HD = "speech-02-hd"
    SPEECH_02_TURBO = "speech-02-turbo"
    SPEECH_01_HD = "speech-01-hd"
    SPEECH_01_TURBO = "speech-01-turbo"


class MiniMaxAudioFormat(str, enum.Enum):
    """Supported MiniMax speech audio formats."""

    MP3 = "mp3"
    WAV = "wav"
    FLAC = "flac"
    PCM = "pcm"


class MiniMaxOutputFormat(str, enum.Enum):
    """MiniMax non-streaming response encodings."""

    HEX = "hex"
    URL = "url"


_CONTENT_TYPE_BY_FORMAT = {
    MiniMaxAudioFormat.MP3: "audio/mpeg",
    MiniMaxAudioFormat.WAV: "audio/wav",
    MiniMaxAudioFormat.FLAC: "audio/flac",
    MiniMaxAudioFormat.PCM: "audio/pcm",
}


class MiniMaxTTSInputSchema(BaseModel):
    """Input for MiniMax speech synthesis."""

    text: str = Field(..., max_length=9999, description="Text to synthesize into speech.")
    output_file_name: str | None = Field(
        default=None,
        description="Optional filename for the generated audio file.",
    )


class MiniMaxTTS(ConnectionNode):
    """Synthesizes speech with the MiniMax synchronous text-to-audio API."""

    group: Literal[NodeGroup.AUDIO] = NodeGroup.AUDIO
    name: str = "minimax-tts"
    connection: MiniMaxConnection | None = None
    error_handling: ErrorHandling = Field(default_factory=lambda: ErrorHandling(timeout_seconds=600))
    model: MiniMaxSpeechModel | str = MiniMaxSpeechModel.SPEECH_2_8_HD
    stream: Literal[False] = Field(default=False, description="Use the synchronous non-streaming response.")
    language_boost: str | None = Field(default=None, description="Language or dialect recognition boost.")
    output_format: MiniMaxOutputFormat = MiniMaxOutputFormat.HEX
    voice_setting: dict[str, Any] | None = Field(default=None, description="Voice selection and delivery settings.")
    pronunciation_dict: dict[str, Any] | None = Field(default=None, description="Pronunciation replacements.")
    audio_setting: dict[str, Any] = Field(
        default_factory=lambda: {"format": MiniMaxAudioFormat.MP3.value},
        description="Audio format, sample rate, bitrate, and channel settings.",
    )
    voice_modify: dict[str, Any] | None = Field(default=None, description="Voice effect settings.")
    subtitle_enable: bool = Field(default=False, description="Include subtitle data in the response.")
    output_file_name: str | None = None
    input_schema: ClassVar[type[MiniMaxTTSInputSchema]] = MiniMaxTTSInputSchema

    def __init__(self, **kwargs: Any) -> None:
        """Initialize MiniMax speech synthesis.

        Args:
            **kwargs: Node configuration, including an optional client or connection.
        """
        if kwargs.get("client") is None and kwargs.get("connection") is None:
            kwargs["connection"] = MiniMaxConnection()
        super().__init__(**kwargs)

    def execute(
        self, input_data: MiniMaxTTSInputSchema, config: RunnableConfig = None, **kwargs
    ) -> dict[str, bytes | list[io.BytesIO]]:
        """Generate an audio file from text.

        Args:
            input_data: Text and optional output filename.
            config: Runtime configuration.
            **kwargs: Callback metadata.

        Returns:
            Raw audio bytes and a named in-memory audio file.

        Raises:
            NodeFailedException: If the API reports an error or returns malformed audio data.
        """
        config = ensure_config(config)
        check_cancellation(config)
        self.run_on_node_execute_run(config.callbacks, **kwargs)

        request = {
            "model": self.model.value if isinstance(self.model, MiniMaxSpeechModel) else self.model,
            "text": input_data.text,
            "stream": self.stream,
            "output_format": self.output_format.value,
            "audio_setting": self.audio_setting,
            "subtitle_enable": self.subtitle_enable,
        }
        request.update(
            {
                key: value
                for key, value in {
                    "language_boost": self.language_boost,
                    "voice_setting": self.voice_setting,
                    "pronunciation_dict": self.pronunciation_dict,
                    "voice_modify": self.voice_modify,
                }.items()
                if value is not None
            }
        )

        response = self.client.request(
            method=self.connection.method,
            url=self.connection.url,
            headers=self.connection.headers,
            json={**request, **(self.connection.data or {})},
        )
        if response.status_code != 200:
            response.raise_for_status()

        body = response.json()
        base = body.get("base_resp") or {}
        data = body.get("data") or {}
        if base.get("status_code") != 0 or data.get("status") != 2:
            message = base.get("status_msg") or "The speech synthesis request did not complete."
            raise NodeFailedException(message=f"MiniMax speech synthesis failed: {message}")

        audio = data.get("audio")
        if not isinstance(audio, str) or not audio:
            raise NodeFailedException(message="MiniMax speech synthesis returned no audio data.")

        if self.output_format == MiniMaxOutputFormat.URL:
            audio_response = self.client.get(audio)
            if audio_response.status_code != 200:
                audio_response.raise_for_status()
            audio_bytes = audio_response.content
        else:
            try:
                audio_bytes = bytes.fromhex(audio)
            except ValueError as error:
                raise NodeFailedException(
                    message="MiniMax speech synthesis returned invalid hex audio data."
                ) from error

        audio_format = MiniMaxAudioFormat(self.audio_setting.get("format", MiniMaxAudioFormat.MP3.value))
        output_file_name = input_data.output_file_name or self.output_file_name or f"audio.{audio_format.value}"
        audio_file = io.BytesIO(audio_bytes)
        audio_file.name = output_file_name
        audio_file.content_type = _CONTENT_TYPE_BY_FORMAT[audio_format]

        return {"content": audio_bytes, "files": [audio_file]}
