import enum
import io
import json
from typing import ClassVar, Literal
from urllib.parse import urljoin

from pydantic import BaseModel, ConfigDict, Field

from dynamiq.connections import ElevenLabs as ElevenLabsConnection
from dynamiq.nodes import ErrorHandling
from dynamiq.nodes.node import ConnectionNode, NodeGroup, ensure_config
from dynamiq.runnables import RunnableConfig


class Voices(str, enum.Enum):
    Rachel = "21m00Tcm4TlvDq8ikWAM"
    Drew = "29vD33N1CtxCmqQRPOHJ"
    Clyde = "2EiwWnXFnvU5JabPnv8n"
    Paul = "5Q0t7uMcjvnagumLfvZi"
    Domi = "AZnzlk1XvdvUeBnXmlld"
    Dave = "CYw3kZ02Hs0563khs1Fj"
    Fin = "D38z5RcWu1voky8WS1ja"
    Sarah = "EXAVITQu4vr4xnSDxMaL"
    Antoni = "ErXwobaYiN019PkySvjV"
    Thomas = "GBv7mTt0atIp3Br8iCZE"
    Charlie = "IKne3meq5aSn9XLyUdCD"
    Emily = "LcfcDJNUP1GQjkzn1xUU"


def format_url(method: str, url: str, voice_id: str) -> str:
    """Formats the given URL by including the `voice_id` if necessary.

    Args:
        method: type of request for vocalizer
        voice_id: voice id for vocalizer.
        url (str): The URL to format.

    Returns:
        str: The modified URL.
    """
    if url.rstrip("/").endswith("v1"):
        url = urljoin(url, method)
    if "{voice_id}" in url:
        url = url.format(voice_id=voice_id)
    elif url.rstrip("/").endswith("text-to-speech") or url.rstrip("/").endswith(
        "speech-to-speech"
    ):
        url = urljoin(url, voice_id)
    return url


class ElevenLabsTTSInputSchema(BaseModel):
    text: str = Field(..., description="Parameter to provide text for vocalization.")


class ElevenLabsTTS(ConnectionNode):
    """
    A component for vocalizing text using the ElevenLabs API.

    Attributes:
        group (Literal[NodeGroup.AUDIO]): The group the node belongs to.
        name (str): The name of the node.
        connection (ElevenLabsConnection | None): The connection to the ElevenLabs API. A new connection
            is created if none is provided.
        voice_id: The voice identifier, that should be used for vocalizing.
        error_handling (ErrorHandling): Error handling configuration.
        model(str): The model for vocalizing, defaults to "eleven_monolingual_v1"
        stability(float): The slider determines how stable the voice is and the randomness
        between each generation.
        similarity_boost(float): The slider dictates how closely the AI should adhere to the original voice when
        attempting to replicate it.
        style(float): The setting that attempts to amplify the style of the original speaker.
        use_speaker_boost(bool):The setting for boosting the similarity to the original speaker
    """

    group: Literal[NodeGroup.AUDIO] = NodeGroup.AUDIO
    name: str = "ElevenLabsTTS"
    voice_id: Voices | str | None = Voices.Rachel
    connection: ElevenLabsConnection | None = None
    error_handling: ErrorHandling = Field(default_factory=lambda: ErrorHandling(timeout_seconds=600))
    model: str = "eleven_monolingual_v1"
    stability: float = 0.5
    similarity_boost: float = 0.5
    style: float = 0
    use_speaker_boost: bool = True
    input_schema: ClassVar[type[ElevenLabsTTSInputSchema]] = ElevenLabsTTSInputSchema

    def __init__(self, **kwargs):
        """Initialize the ElevenLabs audio generation.

        If neither client nor connection is provided in kwargs, a new ElevenLabs connection is created.

        Args:
            **kwargs: Keyword arguments to initialize the node.
        """
        if kwargs.get("client") is None and kwargs.get("connection") is None:
            kwargs["connection"] = ElevenLabsConnection()
        super().__init__(**kwargs)

    def execute(
        self, input_data: ElevenLabsTTSInputSchema, config: RunnableConfig = None, **kwargs
    ) -> dict[str, bytes]:
        """Execute the audio generation process.

        This method takes input data and returns the result.

        Args:
            input_data (ElevenLabsTTSInputSchema): The input data containing the text.
            config (RunnableConfig, optional): Configuration for the execution. Defaults to None.
            **kwargs: Additional keyword arguments.

        Returns:
             dict: A dictionary with the following key:
                - "content" (bytes): Bytes containing the audio generation result.
        """
        input_dict = {
            "model_id": self.model,
            "text": input_data.text,
            "voice_settings": {
                "stability": self.stability,
                "similarity_boost": self.similarity_boost,
                "style": self.style,
                "use_speaker_boost": self.use_speaker_boost,
            },
        }
        config = ensure_config(config)
        self.run_on_node_execute_run(config.callbacks, **kwargs)
        response = self.client.request(
            method=self.connection.method,
            url=format_url("text-to-speech/", self.connection.url, self.voice_id),
            json={**input_dict, **self.connection.data},
            headers=self.connection.headers,
        )
        if response.status_code != 200:
            response.raise_for_status()
        return {
            "content": response.content,
        }


class ElevenLabsSTSInputSchema(BaseModel):
    audio: io.BytesIO | bytes = Field(..., description="Parameter to provide input audio for audio generation.")
    model_config = ConfigDict(arbitrary_types_allowed=True)


class ElevenLabsSTS(ConnectionNode):
    """
    A component for vocalizing text using the ElevenLabs API.

    Attributes:
        group (Literal[NodeGroup.AUDIO]): The group the node belongs to. name (str): The name of the node.
        connection (ElevenLabsConnection | None): The connection to the ElevenLabs API. A new connection is created if
            none is provided.
        voice_id: The voice identifier, that should be used for vocalizing.
        error_handling (ErrorHandling): Error handling configuration.
        model(str): The model for vocalizing, defaults to "eleven_english_sts_v2".
        stability(float): The slider determines how stable the voice is and the randomness
            between each generation.
        similarity_boost(float): The slider dictates how closely the AI should adhere to the original voice when
            attempting to replicate it.
        style(float): The setting that attempts to amplify the style of the original speaker.
        use_speaker_boost(bool):The setting for boosting the similarity to the original speaker
    """

    group: Literal[NodeGroup.AUDIO] = NodeGroup.AUDIO
    name: str = "ElevenLabsSTS"
    voice_id: Voices | str | None = Voices.Rachel
    connection: ElevenLabsConnection | None = None
    error_handling: ErrorHandling = Field(default_factory=lambda: ErrorHandling(timeout_seconds=600))
    model: str = "eleven_english_sts_v2"
    stability: float = 0.5
    similarity_boost: float = 0.5
    style: float = 0
    use_speaker_boost: bool = True
    input_schema: ClassVar[type[ElevenLabsSTSInputSchema]] = ElevenLabsSTSInputSchema

    def __init__(self, **kwargs):
        """Initialize the ElevenLabs audio generation.

        If neither client nor connection is provided in kwargs, a new ElevenLabs connection is created.

        Args:
            **kwargs: Keyword arguments to initialize the node.
        """
        if kwargs.get("client") is None and kwargs.get("connection") is None:
            kwargs["connection"] = ElevenLabsConnection()
        super().__init__(**kwargs)

    def execute(
        self, input_data: ElevenLabsSTSInputSchema, config: RunnableConfig = None, **kwargs
    ) -> dict[str, bytes]:
        """Execute the audio generation process.

        This method takes input data and returns the result.

        Args:
            input_data (dict[str, Any]): The input data containing audio that should be vocalized. Audio
                can be BytesIO or bytes format only.
            config (RunnableConfig, optional): Configuration for the execution. Defaults to None.
            **kwargs: Additional keyword arguments.

        Returns:
             dict: A dictionary with the following key:
                - "content" (bytes): Bytes containing the audio generation result.
        """
        input_dict = {
            "model_id": self.model,
            "voice_settings": json.dumps(
                {
                    "stability": self.stability,
                    "similarity_boost": self.similarity_boost,
                    "style": self.style,
                    "use_speaker_boost": self.use_speaker_boost,
                }
            ),
        }
        audio = input_data.audio
        config = ensure_config(config)
        self.run_on_node_execute_run(config.callbacks, **kwargs)
        response = self.client.request(
            method=self.connection.method,
            url=format_url("speech-to-speech/", self.connection.url, self.voice_id),
            headers=self.connection.headers,
            data=input_dict | self.connection.data,
            files={"audio": audio},
        )
        if response.status_code != 200:
            response.raise_for_status()
        return {
            "content": response.content,
        }
