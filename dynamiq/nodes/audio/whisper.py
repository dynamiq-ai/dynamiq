import io
from typing import ClassVar, Literal
from urllib.parse import urljoin

from pydantic import BaseModel, ConfigDict, Field

from dynamiq.connections import OpenAI as OpenAIConnection
from dynamiq.connections import Whisper as WhisperConnection
from dynamiq.nodes import ErrorHandling
from dynamiq.nodes.node import ConnectionNode, NodeGroup, ensure_config
from dynamiq.runnables import RunnableConfig

DEFAULT_FILE_NAME = "temp.wav"
DEFAULT_CONTENT_TYPE = "audio/wav"


class WhisperSTTInputSchema(BaseModel):
    audio: io.BytesIO | bytes = Field(..., description="Parameter to provide audio for transcribing.")
    model_config = ConfigDict(arbitrary_types_allowed=True)


class WhisperSTT(ConnectionNode):
    """
    A component for transcribing audio files using the Whisper speech recognition system.

    Attributes:
        group (Literal[NodeGroup.AUDIO]): The group the node belongs to.
        name (str): The name of the node.
        connection (WhisperConnection | None): The connection to the Whisper API.A new connection
            is created if none is provided.
        client (OpenAIClient | None): The OpenAI client instance.
        model (str): The model name to use for transcribing.
        error_handling (ErrorHandling): Error handling configuration.
    """

    group: Literal[NodeGroup.AUDIO] = NodeGroup.AUDIO
    name: str = "Whisper"
    model: str
    connection: WhisperConnection | OpenAIConnection | None = None
    error_handling: ErrorHandling = Field(default_factory=lambda: ErrorHandling(timeout_seconds=600))
    default_file_name: str = DEFAULT_FILE_NAME
    default_content_type: str = DEFAULT_CONTENT_TYPE
    input_schema: ClassVar[type[WhisperSTTInputSchema]] = WhisperSTTInputSchema

    def __init__(self, **kwargs):
        """Initialize the Whisper transcriber.

        If neither client nor connection is provided in kwargs, a new Whisper connection is created.

        Args:
            **kwargs: Keyword arguments to initialize the node.
        """
        if kwargs.get("client") is None and kwargs.get("connection") is None:
            kwargs["connection"] = WhisperConnection()
        super().__init__(**kwargs)

    def execute(self, input_data: WhisperSTTInputSchema, config: RunnableConfig = None, **kwargs):
        """Execute the audio transcribing process.

        This method takes input data, modifies it(if necessary), and returns the result.

        Args:
            input_data (WhisperSTTInputSchema): The input data containing the audio.
            config (RunnableConfig, optional): Configuration for the execution. Defaults to None.
            **kwargs: Additional keyword arguments.

        Returns:
            str: A string containing the transcribe result.
        """
        config = ensure_config(config)
        self.run_on_node_execute_run(config.callbacks, **kwargs)

        audio = input_data.audio
        if isinstance(audio, bytes):
            audio = io.BytesIO(audio)

        if not isinstance(audio, io.BytesIO):
            raise ValueError("Audio must be a BytesIO object or bytes.")

        audio.name = getattr(audio, "name", self.default_file_name)
        audio.content_type = getattr(audio, "content_type", self.default_content_type)

        if isinstance(self.connection, WhisperConnection):
            transcription = self.get_transcription_with_http_request(model=self.model, audio=audio)
        elif isinstance(self.connection, OpenAIConnection):
            transcription = self.get_transcription_with_openai_client(model=self.model, audio=audio)
        else:
            raise ValueError(f"Connection type {type(self.connection)} does not fit required ones.")

        return {"content": transcription.get("text", "")}

    def get_transcription_with_http_request(self, model: str, audio: io.BytesIO):
        """Get the audio transcription by request.

        This method takes whisper model and audio file, sends request with defined params, and returns the
        transcription.

        Args:
            model(str): The model used for transcribing.
            audio(io.BytesIO): The audio file in BytesIO that should be transcribed
        Returns:
            dict: transcription result.
        """
        connection_url = urljoin(self.connection.url, "audio/transcriptions")
        response = self.client.request(
            method=self.connection.method,
            url=connection_url,
            headers=self.connection.headers,
            params=self.connection.params,
            data=self.connection.data | {"model": model},
            files={"file": (audio.name, audio, audio.content_type)},
        )
        if response.status_code != 200:
            response.raise_for_status()

        return response.json()

    def get_transcription_with_openai_client(self, model: str, audio: io.BytesIO):
        """Get the audio transcription by request.

        This method takes whisper model and audio file, sends request with defined params, and returns the
        transcription.

        Args:
            model(str): The model used for transcribing.
            audio(io.BytesIO): The audio file in BytesIO that should be transcribed
        Returns:
            dict: transcription result.
        """
        response = self.client.audio.transcriptions.create(model=model, file=audio)

        return {"text": response.text}
