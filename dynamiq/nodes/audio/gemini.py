import base64
import io
import mimetypes
from typing import Callable, ClassVar, Literal

from filetype import filetype
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, field_validator

from dynamiq.connections import Gemini as GeminiConnection
from dynamiq.nodes import NodeGroup
from dynamiq.nodes.audio.whisper import DEFAULT_CONTENT_TYPE
from dynamiq.nodes.node import ConnectionNode, ErrorHandling, ensure_config
from dynamiq.runnables import RunnableConfig

DEFAULT_AUDIO_TRANSCRIPTION_PROMPT = "Generate a transcript of the speech."


class GeminiSTTInputSchema(BaseModel):
    audio: io.BytesIO | bytes = Field(..., description="Parameter to provide audio for transcribing.")
    model_config = ConfigDict(arbitrary_types_allowed=True)


class GeminiSTT(ConnectionNode):
    """
    A component for transcribing audio files using the Gemini speech recognition API.

    Attributes:
        group (Literal[NodeGroup.AUDIO]): The group the node belongs to.
        name (str): The name of the node.
        connection (GeminiConnection | None): The connection to the Gemini API.A new connection
            is created if none is provided.
        model (str): The model name to use for transcribing.
        prompt (str): The prompt to use for audio processing.
        error_handling (ErrorHandling): Error handling configuration.
    """

    group: Literal[NodeGroup.AUDIO] = NodeGroup.AUDIO
    name: str = "GeminiSTT"
    model: str
    connection: GeminiConnection | None = None
    error_handling: ErrorHandling = Field(default_factory=lambda: ErrorHandling(timeout_seconds=600))
    default_content_type: str = DEFAULT_CONTENT_TYPE
    prompt: str = DEFAULT_AUDIO_TRANSCRIPTION_PROMPT
    input_schema: ClassVar[type[GeminiSTTInputSchema]] = GeminiSTTInputSchema
    MODEL_PREFIX: ClassVar[str] = "gemini/"

    _completion: Callable = PrivateAttr()

    def __init__(self, **kwargs):
        """Initialize the Gemini transcriber.

        If neither client nor connection is provided in kwargs, a new Gemini connection is created.

        Args:
            **kwargs: Keyword arguments to initialize the node.
        """
        from litellm import completion

        if kwargs.get("client") is None and kwargs.get("connection") is None:
            kwargs["connection"] = GeminiConnection()
        super().__init__(**kwargs)
        self._completion = completion

    @field_validator("model")
    @classmethod
    def set_model(cls, value: str | None) -> str:
        """Set the model with the appropriate prefix.

        Args:
            value (str | None): The model value.

        Returns:
            str: The model value with the prefix.
        """
        if cls.MODEL_PREFIX is not None and not value.startswith(cls.MODEL_PREFIX):
            value = f"{cls.MODEL_PREFIX}{value}"
        return value

    def execute(self, input_data: GeminiSTTInputSchema, config: RunnableConfig = None, **kwargs):
        """Execute the audio transcribing process.

        This method takes input data, modifies it(if necessary), and returns the result.

        Args:
            input_data (GeminiSTTInputSchema): The input data containing the audio.
            config (RunnableConfig, optional): Configuration for the execution. Defaults to None.
            **kwargs: Additional keyword arguments.

        Returns:
            content: A string containing the transcribe result.
        """
        config = ensure_config(config)
        self.run_on_node_execute_run(config.callbacks, **kwargs)

        audio = input_data.audio
        if isinstance(audio, io.BytesIO):
            audio.seek(0)
            audio = audio.read()
        if not isinstance(audio, bytes):
            raise ValueError("Audio must be a BytesIO object or bytes.")

        encoded_data = base64.b64encode(audio).decode("utf-8")
        extension = filetype.guess_extension(audio)
        if not extension:
            extension = "wav"

        mime_type, _ = mimetypes.guess_type(f"file.{extension}")

        if mime_type is None:
            mime_type = self.default_content_type

        response = self._completion(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": self.prompt},
                        {
                            "type": "file",
                            "file": {
                                "file_data": f"data:{mime_type};base64,{encoded_data}",
                            },
                        },
                    ],
                }
            ],
            **self.connection.conn_params,
        )

        return {"content": response.choices[0].message.content}
