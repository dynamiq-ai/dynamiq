import enum
from abc import ABC, abstractmethod
from functools import partial
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator
from pydantic_core.core_schema import ValidationInfo

from dynamiq.utils import generate_uuid
from dynamiq.utils.env import get_env_var
from dynamiq.utils.logger import logger

if TYPE_CHECKING:
    from chromadb import ClientAPI as ChromaClient
    from openai import OpenAI as OpenAIClient
    from pinecone import Pinecone as PineconeClient
    from qdrant_client import QdrantClient
    from weaviate import WeaviateClient


class ConnectionType(str, enum.Enum):
    """
    This enum defines various connection types for different services and databases.
    """
    Anthropic = "Anthropic"
    AWS = "AWS"
    Chroma = "Chroma"
    Cohere = "Cohere"
    Gemini = "Gemini"
    GeminiVertexAI = "GeminiVertexAI"
    HttpApiKey = "HttpApiKey"
    Http = "Http"
    Mistral = "Mistral"
    MySQL = "MySQL"
    OpenAI = "OpenAI"
    Pinecone = "Pinecone"
    Unstructured = "Unstructured"
    Weaviate = "Weaviate"
    Whisper = "Whisper"
    ElevenLabs = "ElevenLabs"
    Tavily = "Tavily"
    ScaleSerp = "ScaleSerp"
    ZenRows = "ZenRows"
    Groq = "Groq"
    TogetherAI = "TogetherAI"
    Anyscale = "Anyscale"
    HuggingFace = "HuggingFace"
    WatsonX = "WatsonX"
    AzureAI = "AzureAI"
    Firecrawl = "Firecrawl"
    E2B = "E2B"
    DeepInfra = "DeepInfra"
    Cerebras = "Cerebras"
    Replicate = "Replicate"
    AI21 = "AI21"
    Qdrant = "Qdrant"
    SambaNova = "SambaNova"
    Milvus = "Milvus"
    Perplexity = "Perplexity"
    DeepSeek = "DeepSeek"
    PostgreSQL = "PostgreSQL"
    Exa = "Exa"
    Ollama = "Ollama"


class HTTPMethod(str, enum.Enum):
    """
    This enum defines various method types for different HTTP requests.
    """

    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    DELETE = "DELETE"
    PATCH = "PATCH"


class SearchType(enum.Enum):
    WEB = None
    NEWS = "news"
    IMAGES = "images"
    VIDEOS = "videos"


class BaseConnection(BaseModel, ABC):
    """Represents a base connection class.

    This class should be subclassed to provide specific implementations for different types of
    connections.

    Attributes:
        id (str): A unique identifier for the connection, generated using `generate_uuid`.
        type (ConnectionType): The type of connection.
    """
    id: str = Field(default_factory=generate_uuid)
    type: ConnectionType

    @property
    def conn_params(self) -> dict:
        """
        Returns the parameters required for connection.

        Returns:
            dict: An empty dictionary.
        """
        return {}

    def to_dict(self, **kwargs) -> dict:
        """Converts the connection instance to a dictionary.

        Returns:
            dict: A dictionary representation of the connection instance.
        """
        return self.model_dump(**kwargs)

    @abstractmethod
    def connect(self):
        """Connects to the service.

        This method should be implemented by subclasses to establish a connection to the service.

        Raises:
            NotImplementedError: If the method is not implemented by a subclass.
        """
        raise NotImplementedError


class BaseApiKeyConnection(BaseConnection):
    """
    Represents a base connection class that uses an API key for authentication.

    Attributes:
        api_key (str): The API key used for authentication.
    """
    api_key: str

    @abstractmethod
    def connect(self):
        """
        Connects to the service.

        This method should be implemented by subclasses to establish a connection to the service using
        the provided API key.

        Raises:
            NotImplementedError: If the method is not implemented by a subclass.
        """
        raise NotImplementedError

    @property
    def conn_params(self) -> dict:
        """
        Returns the parameters required for connection.

        Returns:
            dict: A dictionary containing the API key with the key 'api_key'.
        """
        return {"api_key": self.api_key}


class HttpApiKey(BaseApiKeyConnection):
    """
    Represents a connection to an API that uses an HTTP API key for authentication.

    Attributes:
        type (Literal[ConnectionType.HttpApiKey]): The type of connection, always 'HttpApiKey'.
        url (str): The URL of the API.
    """

    type: Literal[ConnectionType.HttpApiKey] = ConnectionType.HttpApiKey
    url: str

    def connect(self):
        """
        Connects to the API.

        This method establishes a connection to the API using the provided URL and returns a requests
        session.

        Returns:
            requests: A requests module for making HTTP requests to the API.
        """
        import requests

        return requests

    @property
    def conn_params(self) -> dict:
        """
        Returns the parameters required for connection.

        Returns:
            dict: A dictionary containing the API key with the key 'api_key' and base url with the key 'api_base'.
        """
        return {
            "api_base": self.url,
            "api_key": self.api_key,
        }


class Http(BaseConnection):
    """
    Represents a connection to an API.

    Attributes:
        type (Literal[ConnectionType.HttpConnection]): The type of connection, always 'HttpConnection'.
        url (str): The URL of the API.
        method (str): HTTP method used for the request, defaults to HTTPMethod.POST.
        headers (dict[str, Any]): Additional headers to include in the request, defaults to an empty dictionary.
        params (Optional[dict[str, Any]]): Parameters to include in the request, defaults to an empty dictionary.
        data (Optional[dict[str, Any]]): Data to include in the request, defaults to an empty dictionary.
    """

    type: Literal[ConnectionType.Http] = ConnectionType.Http
    url: str = ""
    method: HTTPMethod
    headers: dict[str, Any] = Field(default_factory=dict)
    params: dict[str, Any] | None = Field(default_factory=dict)
    data: dict[str, Any] | None = Field(default_factory=dict)

    def connect(self):
        """
        Connects to the API.

        This method establishes a connection to the API using the provided URL and returns a requests
        session.

        Returns:
            requests: A requests module for making HTTP requests to the API.
        """
        import requests

        return requests


class OpenAI(BaseApiKeyConnection):
    """
    Represents a connection to the OpenAI service.

    Attributes:
        type (Literal[ConnectionType.OpenAI]): The type of connection, which is always 'OpenAI'.
        api_key (str): The API key for the OpenAI service, fetched from the environment variable 'OPENAI_API_KEY'.
    """
    type: Literal[ConnectionType.OpenAI] = ConnectionType.OpenAI
    api_key: str = Field(default_factory=partial(get_env_var, "OPENAI_API_KEY"))

    def connect(self) -> "OpenAIClient":
        """
        Connects to the OpenAI service.

        This method establishes a connection to the OpenAI service using the provided API key.

        Returns:
            OpenAIClient: An instance of the OpenAIClient connected with the specified API key.
        """
        # Import in runtime to save memory
        from openai import OpenAI as OpenAIClient
        openai_client = OpenAIClient(api_key=self.api_key)
        logger.debug("Connected to OpenAI")
        return openai_client


class Anthropic(BaseApiKeyConnection):
    type: Literal[ConnectionType.Anthropic] = ConnectionType.Anthropic
    api_key: str = Field(default_factory=partial(get_env_var, "ANTHROPIC_API_KEY"))

    def connect(self):
        pass


class AWS(BaseConnection):
    type: Literal[ConnectionType.AWS] = ConnectionType.AWS
    access_key_id: str | None = Field(
        default_factory=partial(get_env_var, "AWS_ACCESS_KEY_ID")
    )
    secret_access_key: str | None = Field(
        default_factory=partial(get_env_var, "AWS_SECRET_ACCESS_KEY")
    )
    region: str = Field(default_factory=partial(get_env_var, "AWS_DEFAULT_REGION"))
    profile: str | None = Field(default_factory=partial(get_env_var, "AWS_DEFAULT_PROFILE"))

    def connect(self):
        pass

    @property
    def conn_params(self):
        if self.profile:
            return {
                "aws_profile_name": self.profile,
                "aws_region_name": self.region,
            }
        else:
            return {
                "aws_access_key_id": self.access_key_id,
                "aws_secret_access_key": self.secret_access_key,
                "aws_region_name": self.region,
            }


class Gemini(BaseApiKeyConnection):
    type: Literal[ConnectionType.Gemini] = ConnectionType.Gemini
    api_key: str = Field(default_factory=partial(get_env_var, "GEMINI_API_KEY"))

    def connect(self):
        pass


class GeminiVertexAI(BaseConnection):
    """
    Represents a connection to the Gemini Vertex AI service.

    This connection requires additional GCP application credentials. The credentials should be set in the
    `application_default_credentials.json` file. The path to this credentials file should be defined in the
    `GOOGLE_APPLICATION_CREDENTIALS` environment variable.

    Attributes:
        project_id (str): The GCP project ID.
        project_location (str): The location of the GCP project.
        type (Literal[ConnectionType.GeminiVertexAI]): The type of connection, which is always 'GeminiVertexAI'.
    """

    project_id: str
    project_location: str
    type: Literal[ConnectionType.GeminiVertexAI] = ConnectionType.GeminiVertexAI

    def connect(self):
        pass

    @property
    def conn_params(self):
        """
        Returns the parameters required for the connection.

        This property returns a dictionary containing the project ID and project location.

        Returns:
            dict: A dictionary with the keys 'vertex_project' and 'vertex_location'.
        """
        return {
            "vertex_project": self.project_id,
            "vertex_location": self.project_location,
        }


class Cohere(BaseApiKeyConnection):
    type: Literal[ConnectionType.Cohere] = ConnectionType.Cohere
    api_key: str = Field(default_factory=partial(get_env_var, "COHERE_API_KEY"))

    def connect(self):
        pass


class Mistral(BaseApiKeyConnection):
    type: Literal[ConnectionType.Mistral] = ConnectionType.Mistral
    api_key: str = Field(default_factory=partial(get_env_var, "MISTRAL_API_KEY"))

    def connect(self):
        pass


class Whisper(Http):
    """
    Represents a connection to the Whisper API using an HTTP request.

    Attributes:
        type (Literal[ConnectionType.Whisper]): Type of the connection, which is always "Whisper".
        url (str): URL of the Whisper API, fetched from the environment variable "WHISPER_URL".
        method (str): HTTP method used for the request, defaults to HTTPMethod.POST.
        api_key (str): API key for authentication, fetched from the environment variable "OPENAI_API_KEY".
    """
    type: Literal[ConnectionType.Whisper] = ConnectionType.Whisper
    url: str = Field(
        default_factory=partial(
            get_env_var, "WHISPER_URL", "https://api.openai.com/v1/"
        )
    )
    method: str = HTTPMethod.POST
    api_key: str = Field(default_factory=partial(get_env_var, "OPENAI_API_KEY"))

    def connect(self):
        """
        Configures the request authorization header with the API key for authentication

        Returns:
            requests: The `requests` module for making HTTP requests.
        """
        self.headers.update({"Authorization": f"Bearer {self.api_key}"})
        return super().connect()


class ElevenLabs(Http):
    """
    Represents a connection to the ElevenLabs API using an HTTP request.

    Attributes:
        type (Literal[ConnectionType.ElevenLabs]): Type of the connection, which is always "ElevenLabs".
        url (str): URL of the ElevenLabs API.
        method (str): HTTP method used for the request, defaults to HTTPMethod.POST.
        api_key (str): API key for authentication, fetched from the environment variable "ELEVENLABS_API_KEY".
    """

    type: Literal[ConnectionType.ElevenLabs] = ConnectionType.ElevenLabs
    url: str = Field(
        default_factory=partial(
            get_env_var,
            "ELEVENLABS_URL",
            "https://api.elevenlabs.io/v1/",
        )
    )
    method: str = HTTPMethod.POST
    api_key: str = Field(default_factory=partial(get_env_var, "ELEVENLABS_API_KEY"))

    def connect(self):
        """
        Connects to the ElevenLabs API.

        Returns:
            requests: The `requests` module for making HTTP requests.
        """
        self.headers.update({"xi-api-key": self.api_key})
        return super().connect()


class Pinecone(BaseApiKeyConnection):
    """
    Represents a connection to the Pinecone service.

    Attributes:
        type (Literal[ConnectionType.Pinecone]): The type of connection, always 'Pinecone'.
        api_key (str): The API key for the service.
            Defaults to the environment variable 'PINECONE_API_KEY'.
    """

    type: Literal[ConnectionType.Pinecone] = ConnectionType.Pinecone
    api_key: str = Field(default_factory=partial(get_env_var, "PINECONE_API_KEY"))

    def connect(self) -> "PineconeClient":
        """
        Connects to the Pinecone service.

        This method establishes a connection to the Pinecone service using the provided API key.

        Returns:
            PineconeClient: An instance of the PineconeClient connected to the service.
        """
        # Import in runtime to save memory
        from pinecone import Pinecone as PineconeClient
        pinecone_client = PineconeClient(self.api_key)
        logger.debug("Connected to Pinecone")
        return pinecone_client


class Qdrant(BaseApiKeyConnection):
    """
    Represents a connection to the Qdrant service.

    Attributes:
        type (Literal[ConnectionType.Qdrant]): The type of connection, always 'Qdrant'.
        url (str): The URL of the Qdrant service.
            Defaults to the environment variable 'QDRANT_URL'.
        api_key (str): The API key for the Qdrant service.
            Defaults to the environment variable 'QDRANT_API_KEY'.
    """

    type: Literal[ConnectionType.Qdrant] = ConnectionType.Qdrant
    url: str = Field(default_factory=partial(get_env_var, "QDRANT_URL"))
    api_key: str = Field(default_factory=partial(get_env_var, "QDRANT_API_KEY"))

    def connect(self) -> "QdrantClient":
        from qdrant_client import QdrantClient

        qdrant_client = QdrantClient(
            url=self.url,
            api_key=self.api_key,
        )

        return qdrant_client


class WeaviateDeploymentType(str, enum.Enum):
    """
    Defines various deployment types for different Weaviate deployments.

    Attributes:
        WEAVIATE_CLOUD (str): Represents a deployment on Weaviate Cloud.
            Value is 'weaviate_cloud'.
        CUSTOM (str): Represents a custom deployment.
            Value is 'custom'.
    """

    WEAVIATE_CLOUD = "weaviate_cloud"
    CUSTOM = "custom"


class Weaviate(BaseApiKeyConnection):
    """
    Represents a connection to the Weaviate service.

    Attributes:
        type (Literal[ConnectionType.Weaviate]): The type of connection, always 'Weaviate'.
        deployment_type (WeaviateDeploymentType): The deployment type of the service.
        api_key (str): The API key for the service.
            Defaults to the environment variable 'WEAVIATE_API_KEY'.
        url (str): The URL of the service.
            Defaults to the environment variable 'WEAVIATE_URL'.
        http_host (str): The HTTP host for the service.
            Defaults to the environment variable 'WEAVIATE_HTTP_HOST'.
        http_port (int): The HTTP port for the service.
            Defaults to the environment variable 'WEAVIATE_HTTP_PORT'.
        grpc_host (str): The gRPC host for the service.
            Defaults to the environment variable 'WEAVIATE_GRPC_HOST'.
        grpc_port (int): The gRPC port for the service.
            Defaults to the environment variable 'WEAVIATE_GRPC_PORT'.
    """

    type: Literal[ConnectionType.Weaviate] = ConnectionType.Weaviate
    deployment_type: WeaviateDeploymentType = WeaviateDeploymentType.WEAVIATE_CLOUD
    api_key: str = Field(default_factory=partial(get_env_var, "WEAVIATE_API_KEY"))
    url: str = Field(default_factory=partial(get_env_var, "WEAVIATE_URL"))
    http_host: str = Field(default_factory=partial(get_env_var, "WEAVIATE_HTTP_HOST"))
    http_port: int = Field(default_factory=partial(get_env_var, "WEAVIATE_HTTP_PORT", 443))
    grpc_host: str = Field(default_factory=partial(get_env_var, "WEAVIATE_GRPC_HOST"))
    grpc_port: int = Field(default_factory=partial(get_env_var, "WEAVIATE_GRPC_PORT", 50051))

    def connect(self) -> "WeaviateClient":
        """
        Connects to the Weaviate service.

        This method establishes a connection to the Weaviate service using the provided URL and API key.

        Returns:
            WeaviateClient: An instance of the WeaviateClient connected to the specified URL.
        """
        # Import in runtime to save memory
        from weaviate import connect_to_custom, connect_to_weaviate_cloud
        from weaviate.classes.init import AdditionalConfig, Auth, Timeout

        if self.deployment_type == WeaviateDeploymentType.WEAVIATE_CLOUD:
            weaviate_client = connect_to_weaviate_cloud(
                cluster_url=self.url,
                auth_credentials=Auth.api_key(self.api_key),
            )
            logger.debug(f"Connected to Weaviate with url={self.url}")
            return weaviate_client

        elif self.deployment_type == WeaviateDeploymentType.CUSTOM:
            weaviate_client = connect_to_custom(
                http_host=self.http_host,
                http_port=self.http_port,
                http_secure=True,
                grpc_host=self.grpc_host,
                grpc_port=self.grpc_port,
                grpc_secure=True,
                auth_credentials=Auth.api_key(self.api_key),
                additional_config=AdditionalConfig(
                    timeout=Timeout(init=30, query=60, insert=120),  # Values in seconds
                ),
                skip_init_checks=False,
            )
            logger.debug(f"Connected to Weaviate with http_host={self.http_host}")
            return weaviate_client
        else:
            raise ValueError("Invalid deployment type")


class Chroma(BaseConnection):
    """
    Represents a connection to the Chroma service.

    Attributes:
        type (Literal[ConnectionType.Chroma]): The type of connection, which is always 'Chroma'.
        host (str): The host address of the Chroma service, fetched from the environment variable 'CHROMA_HOST'.
        port (int): The port number of the Chroma service, fetched from the environment variable 'CHROMA_PORT'.
    """

    type: Literal[ConnectionType.Chroma] = ConnectionType.Chroma
    host: str = Field(default_factory=partial(get_env_var, "CHROMA_HOST"))
    port: int = Field(default_factory=partial(get_env_var, "CHROMA_PORT"))

    @property
    def vector_store_cls(self):
        """
        Returns the ChromaVectorStore class.

        This property dynamically imports and returns the ChromaVectorStore class
        from the 'dynamiq.storages.vector' module.

        Returns:
            type: The ChromaVectorStore class.
        """
        from dynamiq.storages.vector import ChromaVectorStore

        return ChromaVectorStore

    def connect(self) -> "ChromaClient":
        """
        Connects to the Chroma service.

        This method establishes a connection to the Chroma service using the provided host and port.

        Returns:
            ChromaClient: An instance of the ChromaClient connected to the specified host and port.
        """
        # Import in runtime to save memory
        from chromadb import HttpClient

        chroma_client = HttpClient(host=self.host, port=self.port)
        logger.debug(f"Connected to Chroma with host={self.host} and port={str(self.port)}")
        return chroma_client


class Unstructured(HttpApiKey):
    """
    Represents a connection to the Unstructured API.

    Attributes:
        type (Literal[ConnectionType.Unstructured]): The type of connection, which is always 'Unstructured'.
        url (str): The URL of the Unstructured API, fetched from the environment variable 'UNSTRUCTURED_API_URL'.
        api_key (str): The API key for the Unstructured API, fetched from the environment
            variable 'UNSTRUCTURED_API_KEY'.
    """

    type: Literal[ConnectionType.Unstructured] = ConnectionType.Unstructured
    url: str = Field(
        default_factory=partial(
            get_env_var,
            "UNSTRUCTURED_API_URL",
            "https://api.unstructured.io/",
        )
    )
    api_key: str = Field(default_factory=partial(get_env_var, "UNSTRUCTURED_API_KEY"))

    def connect(self):
        """
        Connects to the Unstructured API.
        """
        pass


class Tavily(Http):
    type: Literal[ConnectionType.Tavily] = ConnectionType.Tavily
    url: str = Field(default="https://api.tavily.com")
    api_key: str = Field(default_factory=partial(get_env_var, "TAVILY_API_KEY"))
    method: Literal[HTTPMethod.POST] = HTTPMethod.POST

    def connect(self):
        """
        Returns the requests module for making HTTP requests.
        """
        self.data.update({"api_key": self.api_key})
        return super().connect()


class ScaleSerp(Http):
    """
    Connection class for Scale SERP Search API.
    """

    type: Literal[ConnectionType.ScaleSerp] = ConnectionType.ScaleSerp
    url: str = "https://api.scaleserp.com"
    api_key: str = Field(default_factory=partial(get_env_var, "SERP_API_KEY"))
    method: str = HTTPMethod.GET

    # Common parameters
    search_type: SearchType = SearchType.WEB

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def connect(self):
        """
        Returns the requests module for making HTTP requests.
        """
        self.params.update({"api_key": self.api_key})
        return super().connect()

    def get_params(self, query: str | None = None, url: str | None = None, **kwargs) -> dict[str, Any]:
        """
        Prepare the parameters for the API request.
        """
        params = {"api_key": self.api_key, "search_type": self.search_type, **kwargs}
        if self.search_type == SearchType.WEB:
            params.pop("search_type")

        if query:
            params["q"] = query
        elif url:
            params["url"] = url

        return {k: v for k, v in params.items() if v is not None}


class ZenRows(Http):
    """
    Connection class for ZenRows Scrape API.
    """

    type: Literal[ConnectionType.ZenRows] = ConnectionType.ZenRows
    url: str = "https://api.zenrows.com/v1/"
    api_key: str = Field(default_factory=partial(get_env_var, "ZENROWS_API_KEY"))
    method: str = HTTPMethod.GET

    def connect(self):
        """
        Returns the requests module for making HTTP requests.
        """
        self.params.update({"apikey": self.api_key})
        return super().connect()


class Groq(BaseApiKeyConnection):
    type: Literal[ConnectionType.Groq] = ConnectionType.Groq
    api_key: str = Field(default_factory=partial(get_env_var, "GROQ_API_KEY"))

    def connect(self):
        pass


class TogetherAI(BaseApiKeyConnection):
    type: Literal[ConnectionType.TogetherAI] = ConnectionType.TogetherAI
    api_key: str = Field(default_factory=partial(get_env_var, "TOGETHER_API_KEY"))

    def connect(self):
        pass


class Anyscale(BaseApiKeyConnection):
    type: Literal[ConnectionType.Anyscale] = ConnectionType.Anyscale
    api_key: str = Field(default_factory=partial(get_env_var, "ANYSCALE_API_KEY"))

    def connect(self):
        pass


class Firecrawl(Http):
    type: Literal[ConnectionType.Firecrawl] = ConnectionType.Firecrawl
    url: str = Field(default="https://api.firecrawl.dev/v0/")
    api_key: str = Field(default_factory=lambda: get_env_var("FIRECRAWL_API_KEY"))
    method: Literal[HTTPMethod.POST] = HTTPMethod.POST

    def connect(self):
        """
        Returns the requests module for making HTTP requests.
        """
        self.headers.update({"Authorization": f"Bearer {self.api_key}"})
        return super().connect()


class E2B(BaseApiKeyConnection):
    type: Literal[ConnectionType.E2B] = ConnectionType.E2B
    api_key: str = Field(default_factory=partial(get_env_var, "E2B_API_KEY"))

    def connect(self):
        pass


class HuggingFace(BaseApiKeyConnection):
    type: Literal[ConnectionType.HuggingFace] = ConnectionType.HuggingFace
    api_key: str = Field(default_factory=partial(get_env_var, "HUGGINGFACE_API_KEY"))

    def connect(self):
        pass


class WatsonX(BaseApiKeyConnection):
    type: Literal[ConnectionType.WatsonX] = ConnectionType.WatsonX
    api_key: str = Field(default_factory=partial(get_env_var, "WATSONX_API_KEY"))
    project_id: str = Field(default_factory=partial(get_env_var, "WATSONX_PROJECT_ID"))
    url: str = Field(default_factory=partial(get_env_var, "WATSONX_URL"))

    def connect(self):
        pass

    @property
    def conn_params(self) -> dict:
        """
        Returns the parameters required for connection.

        Returns:
            dict: A dictionary containing

                -the API key with the key 'api_key'.

                -the project ID with the key 'project_id'.

                -the url with the key 'url'.
        """
        return {
            "apikey": self.api_key,
            "project_id": self.project_id,
            "url": self.url,
        }


class AzureAI(BaseApiKeyConnection):
    type: Literal[ConnectionType.AzureAI] = ConnectionType.AzureAI
    api_key: str = Field(default_factory=partial(get_env_var, "AZURE_API_KEY"))
    url: str = Field(default_factory=partial(get_env_var, "AZURE_URL"))
    api_version: str = Field(default_factory=partial(get_env_var, "AZURE_API_VERSION"))

    def connect(self):
        pass

    @property
    def conn_params(self) -> dict:
        """
        Returns the parameters required for connection.

        Returns:
            dict: A dictionary containing

                -the API key with the key 'api_key'.

                -the base url with the key 'api_base'.

                -the API version with the key 'api_version'.
        """
        return {
            "api_base": self.url,
            "api_key": self.api_key,
            "api_version": self.api_version,
        }


class DeepInfra(BaseApiKeyConnection):
    type: Literal[ConnectionType.DeepInfra] = ConnectionType.DeepInfra
    api_key: str = Field(default_factory=partial(get_env_var, "DEEPINFRA_API_KEY"))

    def connect(self):
        pass


class Cerebras(BaseApiKeyConnection):
    type: Literal[ConnectionType.Cerebras] = ConnectionType.Cerebras
    api_key: str = Field(default_factory=partial(get_env_var, "CEREBRAS_API_KEY"))

    def connect(self):
        pass


class Replicate(BaseApiKeyConnection):
    type: Literal[ConnectionType.Replicate] = ConnectionType.Replicate
    api_key: str = Field(default_factory=partial(get_env_var, "REPLICATE_API_KEY"))

    def connect(self):
        pass


class AI21(BaseApiKeyConnection):
    type: Literal[ConnectionType.AI21] = ConnectionType.AI21
    api_key: str = Field(default_factory=partial(get_env_var, "AI21_API_KEY"))

    def connect(self):
        pass


class SambaNova(BaseApiKeyConnection):
    type: Literal[ConnectionType.SambaNova] = ConnectionType.SambaNova
    api_key: str = Field(default_factory=partial(get_env_var, "SAMBANOVA_API_KEY"))

    def connect(self):
        pass


class MilvusDeploymentType(str, enum.Enum):
    """
    Defines general deployment types for Milvus deployments.
    Attributes:
        FILE (str): Represents a file-based deployment, validated with a .db suffix.
        HOST (str): Represents a host-based deployment, which could be a cloud, cluster,
                    or single machine with or without authentication.
    """

    FILE = "file"
    HOST = "host"


class Milvus(BaseConnection):
    """
    Represents a connection to the Milvus service.

    Attributes:
        type (Literal[ConnectionType.Milvus]): The type of connection, always 'Milvus'.
        deployment_type (MilvusDeploymentType): The deployment type of the Milvus service
        api_key (Optional[str]): The API key for Milvus
        uri (str): The URI for the Milvus instance (file path or host URL).
    """

    type: Literal[ConnectionType.Milvus] = ConnectionType.Milvus
    deployment_type: MilvusDeploymentType = MilvusDeploymentType.HOST
    uri: str = Field(default_factory=partial(get_env_var, "MILVUS_URI", "http://localhost:19530"))
    api_key: str | None = Field(default_factory=partial(get_env_var, "MILVUS_API_TOKEN", None))

    @field_validator("uri")
    @classmethod
    def validate_uri(cls, uri: str, values: ValidationInfo) -> str:
        deployment_type = values.data.get("deployment_type")

        if deployment_type == MilvusDeploymentType.FILE and not uri.endswith(".db"):
            raise ValueError("For FILE deployment, URI should point to a file ending with '.db'.")

        return uri

    def connect(self):
        from pymilvus import MilvusClient

        if self.deployment_type == MilvusDeploymentType.FILE:
            milvus_client = MilvusClient(uri=self.uri)

        elif self.deployment_type == MilvusDeploymentType.HOST:
            if self.api_key:
                milvus_client = MilvusClient(uri=self.uri, token=self.api_key)
            else:
                milvus_client = MilvusClient(uri=self.uri)

        else:
            raise ValueError("Invalid deployment type for Milvus connection.")

        return milvus_client


class Perplexity(BaseApiKeyConnection):
    type: Literal[ConnectionType.Perplexity] = ConnectionType.Perplexity
    api_key: str = Field(default_factory=partial(get_env_var, "PERPLEXITYAI_API_KEY"))

    def connect(self):
        pass


class DeepSeek(BaseApiKeyConnection):
    type: Literal[ConnectionType.DeepSeek] = ConnectionType.DeepSeek
    api_key: str = Field(default_factory=partial(get_env_var, "DEEPSEEK_API_KEY"))

    def connect(self):
        pass


class PostgreSQL(BaseConnection):
    type: Literal[ConnectionType.PostgreSQL] = ConnectionType.PostgreSQL
    host: str = Field(default_factory=partial(get_env_var, "POSTGRESQL_HOST", "localhost"))
    port: int = Field(default_factory=partial(get_env_var, "POSTGRESQL_PORT", 5432))
    database: str = Field(default_factory=partial(get_env_var, "POSTGRESQL_DATABASE", "db"))
    user: str = Field(default_factory=partial(get_env_var, "POSTGRESQL_USER", "postgres"))
    password: str = Field(default_factory=partial(get_env_var, "POSTGRESQL_PASSWORD", "password"))

    def connect(self):
        try:
            import psycopg

            conn = psycopg.connect(
                host=self.host, port=self.port, dbname=self.database, user=self.user, password=self.password
            )
            logger.debug(
                f"Connected to PGVector with host={self.host}, "
                f"port={str(self.port)}, user={self.user}, "
                f"database={self.database}."
            )
            return conn
        except ImportError:
            raise ImportError("Please install psycopg to use PGVector connection")
        except Exception as e:
            raise ConnectionError(f"Failed to connect to PGVector: {str(e)}")

    @property
    def conn_params(self) -> str:
        """
        Returns the parameters required for connection.

        Returns:
            dict: A string containing the host, the port, the database,
            the user, and the password for the connection.
        """
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"


class Exa(Http):
    """
    Represents a connection to the Exa AI Search API.

    Attributes:
        type (Literal[ConnectionType.Exa]): The type of connection, which is always 'Exa'.
        url (str): The URL of the Exa API.
        method (Literal[HTTPMethod.POST]): HTTP method used for the request, defaults to POST.
        api_key (str): The API key for authentication, fetched from the environment variable 'EXA_API_KEY'.
    """

    type: Literal[ConnectionType.Exa] = ConnectionType.Exa
    url: Literal["https://api.exa.ai"] = Field(default="https://api.exa.ai")
    method: Literal[HTTPMethod.POST] = HTTPMethod.POST
    api_key: str = Field(default_factory=partial(get_env_var, "EXA_API_KEY"))

    def connect(self):
        """
        Configures the request headers with the API key for authentication.

        Returns:
            requests: The requests module for making HTTP requests.
        """
        self.headers.update({"x-api-key": self.api_key, "Content-Type": "application/json"})
        return super().connect()


class Ollama(BaseConnection):
    """Represents a connection to Ollama API.

    Attributes:
        type (Literal[ConnectionType.HttpApiKey]): The type of connection, always 'HttpApiKey'.
        url (str): The URL of the Ollama API, defaults to "http://localhost:11434".
    """

    type: Literal[ConnectionType.Ollama] = ConnectionType.Ollama
    url: str = Field(default="http://localhost:11434")

    def connect(self):
        """Connects to the Ollama API.

        Returns:
            requests: A requests module for making HTTP requests to the API.
        """
        import requests

        return requests

    @property
    def conn_params(self) -> dict:
        """Returns the parameters required for connection.

        Returns:
            dict: A dictionary containing the base url with the key 'api_base'.
        """
        return {
            "api_base": self.url,
        }
