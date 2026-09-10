import io
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_validator

from dynamiq.components.audio.registry import resolve_stt_adapter
from dynamiq.components.audio.stt import BaseSTTAdapter, TranscriptionRequest
from dynamiq.components.audio.utils import prepare_audio_file
from dynamiq.connections import Deepgram as DeepgramConnection
from dynamiq.connections import ElevenLabs as ElevenLabsConnection
from dynamiq.connections import Groq as GroqConnection
from dynamiq.connections import HttpApiKey as HttpApiKeyConnection
from dynamiq.connections import Mistral as MistralConnection
from dynamiq.connections import OpenAI as OpenAIConnection
from dynamiq.connections import Whisper as WhisperConnection
from dynamiq.connections.managers import ConnectionManager
from dynamiq.nodes import ErrorHandling
from dynamiq.nodes.node import ConnectionNode, NodeGroup, ensure_config
from dynamiq.runnables import RunnableConfig
from dynamiq.types.audio import SpeakerHints, TimestampGranularity
from dynamiq.types.cancellation import check_cancellation
from dynamiq.utils.logger import logger

DEFAULT_FILE_NAME = "audio.wav"
DEFAULT_CONTENT_TYPE = "audio/wav"


class SpeechToTextInputSchema(BaseModel):
    audio: io.BytesIO | bytes | list[io.BytesIO | bytes] | None = Field(
        default=None,
        description="Audio file to transcribe, as bytes or a file object.",
        # Agents fill this field from their file store; the LLM must not be asked to type raw audio.
        json_schema_extra={"map_from_storage": True, "is_accessible_to_agent": False},
    )
    audio_url: str | None = Field(
        default=None, description="Public URL of the audio, for providers that download it themselves."
    )
    model_config = ConfigDict(arbitrary_types_allowed=True)

    @model_validator(mode="after")
    def validate_source(self):
        if isinstance(self.audio, list):
            # Agent file injection hands over every stored file; the node transcribes one recording.
            if len(self.audio) > 1:
                logger.warning("SpeechToText received several files; transcribing the first one.")
            self.audio = self.audio[0] if self.audio else None
        if self.audio is None and not self.audio_url:
            raise ValueError("Either `audio` or `audio_url` must be provided.")
        return self


class SpeechToText(ConnectionNode):
    """
    Transcribes audio with any supported speech provider.

    The connection type selects the provider: OpenAI, Groq, an OpenAI-compatible server (HttpApiKey or
    Whisper connection), Mistral Voxtral, Deepgram or ElevenLabs Scribe. Every provider returns the same
    output shape, so downstream nodes do not change when the provider does.

    Attributes:
        group (Literal[NodeGroup.AUDIO]): The group the node belongs to.
        name (str): The name of the node.
        connection: The provider connection. Defaults to OpenAI when neither client nor connection is given.
        model (str | None): Provider model id. Defaults to the provider's recommended batch model.
        language (str | None): Language hint; the provider auto-detects when omitted.
        diarize (bool): Label speakers in segments and words.
        timestamps (TimestampGranularity): Timing detail to request: none, segment or word.
        speakers (SpeakerHints | None): Speaker count hints for diarization.
        prompt (str | None): Context or key terms that improve recognition of names and jargon.
        provider_options (dict): Extra request fields passed to the provider verbatim.
        include_raw_response (bool): Include the provider's untouched response under ``raw``.
        error_handling (ErrorHandling): Error handling configuration.

    Output:
        content (str): Plain transcript.
        transcript (str): Speaker-labelled transcript (equals content without diarization).
        language, languages, duration, speakers, segments, words, usage: see ``dynamiq.types.audio.Transcript``.
    """

    group: Literal[NodeGroup.AUDIO] = NodeGroup.AUDIO
    name: str = "speech-to-text"
    description: str = (
        "Transcribe an audio file to text with the configured provider, optionally with speaker "
        "diarization and word timestamps."
    )
    connection: (
        OpenAIConnection
        | GroqConnection
        | HttpApiKeyConnection
        | WhisperConnection
        | MistralConnection
        | DeepgramConnection
        | ElevenLabsConnection
        | None
    ) = None
    model: str | None = Field(
        default=None, description="Provider model id. Defaults to the provider's recommended batch model."
    )
    language: str | None = Field(
        default=None, description="Language hint (ISO-639-1 or BCP-47). Auto-detected when omitted."
    )
    diarize: bool = Field(default=False, description="Label speakers in segments and words.")
    timestamps: TimestampGranularity = Field(
        default=TimestampGranularity.SEGMENT, description="Timing detail to request: none, segment or word."
    )
    speakers: SpeakerHints | None = Field(default=None, description="Speaker count hints for diarization.")
    prompt: str | None = Field(
        default=None, description="Context or key terms that improve recognition of names and jargon."
    )
    provider_options: dict[str, Any] = Field(
        default_factory=dict, description="Extra request fields passed to the provider verbatim."
    )
    include_raw_response: bool = Field(
        default=False, description="Include the provider's untouched response under `raw`."
    )
    error_handling: ErrorHandling = Field(default_factory=lambda: ErrorHandling(timeout_seconds=600))
    is_files_allowed: bool = True
    input_schema: ClassVar[type[SpeechToTextInputSchema]] = SpeechToTextInputSchema

    _adapter: BaseSTTAdapter | None = PrivateAttr(default=None)

    def __init__(self, **kwargs):
        """Initialize the node. A new OpenAI connection is created when neither client nor connection is given."""
        if kwargs.get("client") is None and kwargs.get("connection") is None:
            kwargs["connection"] = OpenAIConnection()
        super().__init__(**kwargs)

    @model_validator(mode="after")
    def validate_provider_config(self):
        """Pick the default model for the provider and reject options it cannot honour."""
        adapter_cls = self.adapter_class
        if self.model is None:
            self.model = adapter_cls.default_model
        adapter_cls.check_config(
            diarize=self.diarize,
            timestamps=self.timestamps,
            speakers=self.speakers,
            prompt=self.prompt,
            language=self.language,
        )
        return self

    @property
    def adapter_class(self) -> type[BaseSTTAdapter]:
        return resolve_stt_adapter(self.connection, self.client)

    def init_components(self, connection_manager: ConnectionManager | None = None):
        super().init_components(connection_manager)
        if self._adapter is None:
            self._adapter = self.adapter_class(self.connection, self.client)

    def execute(self, input_data: SpeechToTextInputSchema, config: RunnableConfig = None, **kwargs) -> dict[str, Any]:
        """Transcribe the audio and return the normalized transcript as a dictionary.

        Args:
            input_data (SpeechToTextInputSchema): The audio (bytes or file object) or its URL.
            config (RunnableConfig, optional): Configuration for the execution.
            **kwargs: Additional keyword arguments.

        Returns:
            dict: ``dynamiq.types.audio.Transcript`` fields; ``raw`` only when ``include_raw_response`` is set.
        """
        config = ensure_config(config)
        check_cancellation(config)
        self.run_on_node_execute_run(config.callbacks, **kwargs)

        if self._adapter is None:
            self._adapter = self.adapter_class(self.connection, self.client)

        audio = None
        # An explicit URL wins over the file: agents inject whatever their store holds into `audio`.
        if input_data.audio is not None and not input_data.audio_url:
            audio = prepare_audio_file(input_data.audio, DEFAULT_FILE_NAME, DEFAULT_CONTENT_TYPE)

        request = TranscriptionRequest(
            audio=audio,
            audio_url=input_data.audio_url,
            model=self.model,
            language=self.language,
            diarize=self.diarize,
            timestamps=self.timestamps,
            speakers=self.speakers,
            prompt=self.prompt,
            provider_options=dict(self.provider_options),
        )
        self._adapter.check_request(request)
        transcript = self._adapter.transcribe(request)
        return transcript.model_dump(exclude=None if self.include_raw_response else {"raw"})
