import io
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, Field, PrivateAttr, model_validator

from dynamiq.components.audio.registry import resolve_tts_adapter
from dynamiq.components.audio.tts import BaseTTSAdapter, SpeechRequest
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
from dynamiq.types.audio import AudioFormat
from dynamiq.types.cancellation import check_cancellation


class TextToSpeechInputSchema(BaseModel):
    text: str = Field(..., description="Text to synthesize into speech.")
    output_file_name: str | None = Field(default=None, description="Optional file name for the generated audio.")


class TextToSpeech(ConnectionNode):
    """
    Synthesizes speech with any supported speech provider.

    The connection type selects the provider: OpenAI, Groq, an OpenAI-compatible server (HttpApiKey or
    Whisper connection), ElevenLabs, Mistral Voxtral or Deepgram Aura. The output matches the legacy
    ElevenLabsTTS node (``content`` bytes plus a named file), so existing wiring keeps working.

    Attributes:
        group (Literal[NodeGroup.AUDIO]): The group the node belongs to.
        name (str): The name of the node.
        connection: The provider connection. Defaults to OpenAI when neither client nor connection is given.
        model (str | None): Provider model id. Defaults to the provider's recommended model.
        voice (str | None): Provider voice id or name. Defaults to a provider voice where one exists.
        language (str | None): Language code, for providers whose voices are multilingual.
        speed (float | None): Speaking rate multiplier, where supported.
        instructions (str | None): Free-text delivery instructions, for providers that accept them.
        output_format (AudioFormat): Requested audio encoding; adapters map it to provider names.
        sample_rate (int | None): Requested sample rate, where supported.
        output_file_name (str | None): Default file name for the generated audio.
        provider_options (dict): Extra request fields passed to the provider verbatim.
        error_handling (ErrorHandling): Error handling configuration.
    """

    group: Literal[NodeGroup.AUDIO] = NodeGroup.AUDIO
    name: str = "text-to-speech"
    description: str = "Convert text into speech audio with the configured provider and voice."
    connection: (
        OpenAIConnection
        | GroqConnection
        | HttpApiKeyConnection
        | WhisperConnection
        | ElevenLabsConnection
        | MistralConnection
        | DeepgramConnection
        | None
    ) = None
    model: str | None = Field(default=None, description="Provider model id. Defaults per provider.")
    voice: str | None = Field(default=None, description="Provider voice id or name.")
    language: str | None = Field(default=None, description="Language code for multilingual voices.")
    speed: float | None = Field(default=None, gt=0, description="Speaking rate multiplier.")
    instructions: str | None = Field(default=None, description="Delivery instructions (tone, pace, accent).")
    output_format: AudioFormat = Field(default=AudioFormat.MP3, description="Requested audio encoding.")
    sample_rate: int | None = Field(default=None, gt=0, description="Requested sample rate in Hz.")
    output_file_name: str | None = None
    provider_options: dict[str, Any] = Field(
        default_factory=dict, description="Extra request fields passed to the provider verbatim."
    )
    error_handling: ErrorHandling = Field(default_factory=lambda: ErrorHandling(timeout_seconds=600))
    input_schema: ClassVar[type[TextToSpeechInputSchema]] = TextToSpeechInputSchema

    _adapter: BaseTTSAdapter | None = PrivateAttr(default=None)

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
            output_format=self.output_format,
            speed=self.speed,
            instructions=self.instructions,
            language=self.language,
            sample_rate=self.sample_rate,
        )
        return self

    @property
    def adapter_class(self) -> type[BaseTTSAdapter]:
        return resolve_tts_adapter(self.connection, self.client)

    def init_components(self, connection_manager: ConnectionManager | None = None):
        super().init_components(connection_manager)
        if self._adapter is None:
            self._adapter = self.adapter_class(self.connection, self.client)

    def execute(
        self, input_data: TextToSpeechInputSchema, config: RunnableConfig = None, **kwargs
    ) -> dict[str, bytes | str | list[io.BytesIO]]:
        """Synthesize the text and return the audio.

        Args:
            input_data (TextToSpeechInputSchema): The text and an optional output file name.
            config (RunnableConfig, optional): Configuration for the execution.
            **kwargs: Additional keyword arguments.

        Returns:
            dict: ``content`` (raw audio bytes), ``files`` (one named BytesIO) and ``mime_type``.
        """
        config = ensure_config(config)
        check_cancellation(config)
        self.run_on_node_execute_run(config.callbacks, **kwargs)

        if self._adapter is None:
            self._adapter = self.adapter_class(self.connection, self.client)

        request = SpeechRequest(
            text=input_data.text,
            model=self.model,
            voice=self.voice,
            language=self.language,
            speed=self.speed,
            instructions=self.instructions,
            output_format=self.output_format,
            sample_rate=self.sample_rate,
            provider_options=dict(self.provider_options),
        )
        self._adapter.check_request(request)
        result = self._adapter.synthesize(request)

        output_file_name = input_data.output_file_name or self.output_file_name or f"audio.{result.extension}"
        audio_file = io.BytesIO(result.audio)
        audio_file.name = output_file_name
        audio_file.content_type = result.mime_type

        return {"content": result.audio, "files": [audio_file], "mime_type": result.mime_type}
