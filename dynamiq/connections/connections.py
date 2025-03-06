import enum
from abc import ABC, abstractmethod
from functools import cached_property, partial
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, computed_field, field_validator
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


class HTTPMethod(str, enum.Enum):
    """
    This enum defines various method types for different HTTP requests.
    """

    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    DELETE = "DELETE"
    PATCH = "PATCH"


class BaseConnection(BaseModel, ABC):
    """Represents a base connection class.

    This class should be subclassed to provide specific implementations for different types of
    connections.

    Attributes:
        id (str): A unique identifier for the connection, generated using `generate_uuid`.
        type (ConnectionType): The type of connection.
    """
    id: str = Field(default_factory=generate_uuid)

    @computed_field
    @cached_property
    def type(self) -> str:
        return f"{self.__module__.rsplit('.', 1)[0]}.{self.__class__.__name__}"

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
        url (str): The URL of the API.
    """

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
        url (str): The URL of the API.
        method (str): HTTP method used for the request, defaults to HTTPMethod.POST.
        headers (dict[str, Any]): Additional headers to include in the request, defaults to an empty dictionary.
        params (Optional[dict[str, Any]]): Parameters to include in the request, defaults to an empty dictionary.
        data (Optional[dict[str, Any]]): Data to include in the request, defaults to an empty dictionary.
    """

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
        api_key (str): The API key for the OpenAI service, fetched from the environment variable 'OPENAI_API_KEY'.
        url (str): The endpoint url for the OpenAI service, fetched from the environment variable 'OPENAI_URL'.
    """
    api_key: str = Field(default_factory=partial(get_env_var, "OPENAI_API_KEY"))
    url: str = Field(default_factory=partial(get_env_var, "OPENAI_URL", "https://api.openai.com/v1"))

    def connect(self) -> "OpenAIClient":
        """
        Connects to the OpenAI service.

        This method establishes a connection to the OpenAI service using the provided API key.

        Returns:
            OpenAIClient: An instance of the OpenAIClient connected with the specified API key.
        """
        # Import in runtime to save memory
        from openai import OpenAI as OpenAIClient

        openai_client = OpenAIClient(api_key=self.api_key, base_url=self.url)
        logger.debug("Connected to OpenAI")
        return openai_client

    @property
    def conn_params(self) -> dict:
        """
        Returns the parameters required for connection.
        """
        return {
            "api_base": self.url,
            "api_key": self.api_key,
        }


class Anthropic(BaseApiKeyConnection):
    api_key: str = Field(default_factory=partial(get_env_var, "ANTHROPIC_API_KEY"))

    def connect(self):
        pass


class AWS(BaseConnection):
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
    """

    project_id: str
    project_location: str

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
    api_key: str = Field(default_factory=partial(get_env_var, "COHERE_API_KEY"))

    def connect(self):
        pass


class Mistral(BaseApiKeyConnection):
    api_key: str = Field(default_factory=partial(get_env_var, "MISTRAL_API_KEY"))

    def connect(self):
        pass


class Whisper(Http):
    """
    Represents a connection to the Whisper API using an HTTP request.

    Attributes:
        url (str): URL of the Whisper API, fetched from the environment variable "WHISPER_URL".
        method (str): HTTP method used for the request, defaults to HTTPMethod.POST.
        api_key (str): API key for authentication, fetched from the environment variable "OPENAI_API_KEY".
    """
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
        url (str): URL of the ElevenLabs API.
        method (str): HTTP method used for the request, defaults to HTTPMethod.POST.
        api_key (str): API key for authentication, fetched from the environment variable "ELEVENLABS_API_KEY".
    """

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
        api_key (str): The API key for the service.
            Defaults to the environment variable 'PINECONE_API_KEY'.
    """

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
        url (str): The URL of the Qdrant service.
            Defaults to the environment variable 'QDRANT_URL'.
        api_key (str): The API key for the Qdrant service.
            Defaults to the environment variable 'QDRANT_API_KEY'.
    """

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
        host (str): The host address of the Chroma service, fetched from the environment variable 'CHROMA_HOST'.
        port (int): The port number of the Chroma service, fetched from the environment variable 'CHROMA_PORT'.
    """

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
        url (str): The URL of the Unstructured API, fetched from the environment variable 'UNSTRUCTURED_API_URL'.
        api_key (str): The API key for the Unstructured API, fetched from the environment
            variable 'UNSTRUCTURED_API_KEY'.
    """

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

    url: str = "https://api.scaleserp.com"
    api_key: str = Field(default_factory=partial(get_env_var, "SERP_API_KEY"))
    method: str = HTTPMethod.GET

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def connect(self):
        """
        Returns the requests module for making HTTP requests.
        """
        self.params.update({"api_key": self.api_key})
        return super().connect()


class ZenRows(Http):
    """
    Connection class for ZenRows Scrape API.
    """

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
    api_key: str = Field(default_factory=partial(get_env_var, "GROQ_API_KEY"))

    def connect(self):
        pass


class TogetherAI(BaseApiKeyConnection):
    api_key: str = Field(default_factory=partial(get_env_var, "TOGETHER_API_KEY"))

    def connect(self):
        pass


class Anyscale(BaseApiKeyConnection):
    api_key: str = Field(default_factory=partial(get_env_var, "ANYSCALE_API_KEY"))

    def connect(self):
        pass


class Firecrawl(Http):
    url: str = Field(default="https://api.firecrawl.dev/v1/")
    api_key: str = Field(default_factory=lambda: get_env_var("FIRECRAWL_API_KEY"))
    method: Literal[HTTPMethod.POST] = HTTPMethod.POST

    def connect(self):
        """
        Returns the requests module for making HTTP requests.
        """
        self.headers.update({"Authorization": f"Bearer {self.api_key}"})
        return super().connect()


class E2B(BaseApiKeyConnection):
    api_key: str = Field(default_factory=partial(get_env_var, "E2B_API_KEY"))

    def connect(self):
        pass


class HuggingFace(BaseApiKeyConnection):
    api_key: str = Field(default_factory=partial(get_env_var, "HUGGINGFACE_API_KEY"))

    def connect(self):
        pass


class WatsonX(BaseApiKeyConnection):
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
    api_key: str = Field(default_factory=partial(get_env_var, "DEEPINFRA_API_KEY"))

    def connect(self):
        pass


class Cerebras(BaseApiKeyConnection):
    api_key: str = Field(default_factory=partial(get_env_var, "CEREBRAS_API_KEY"))

    def connect(self):
        pass


class Replicate(BaseApiKeyConnection):
    api_key: str = Field(default_factory=partial(get_env_var, "REPLICATE_API_KEY"))

    def connect(self):
        pass


class AI21(BaseApiKeyConnection):
    api_key: str = Field(default_factory=partial(get_env_var, "AI21_API_KEY"))

    def connect(self):
        pass


class SambaNova(BaseApiKeyConnection):
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
        deployment_type (MilvusDeploymentType): The deployment type of the Milvus service
        api_key (Optional[str]): The API key for Milvus
        uri (str): The URI for the Milvus instance (file path or host URL).
    """

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
    api_key: str = Field(default_factory=partial(get_env_var, "PERPLEXITYAI_API_KEY"))

    def connect(self):
        pass


class DeepSeek(BaseApiKeyConnection):
    api_key: str = Field(default_factory=partial(get_env_var, "DEEPSEEK_API_KEY"))

    def connect(self):
        pass


class PostgreSQL(BaseConnection):
    host: str = Field(default_factory=partial(get_env_var, "POSTGRESQL_HOST", "localhost"))
    port: int = Field(default_factory=partial(get_env_var, "POSTGRESQL_PORT", 5432))
    database: str = Field(default_factory=partial(get_env_var, "POSTGRESQL_DATABASE", "db"))
    user: str = Field(default_factory=partial(get_env_var, "POSTGRESQL_USER", "postgres"))
    password: str = Field(default_factory=partial(get_env_var, "POSTGRESQL_PASSWORD", "password"))

    def connect(self):
        try:
            import psycopg

            conn = psycopg.connect(
                host=self.host,
                port=self.port,
                dbname=self.database,
                user=self.user,
                password=self.password,
                row_factory=psycopg.rows.dict_row,
            )
            conn.autocommit = True
            logger.debug(
                f"Connected to PostgreSQL with host={self.host}, "
                f"port={str(self.port)}, user={self.user}, "
                f"database={self.database}."
            )
            return conn
        except Exception as e:
            raise ConnectionError(f"Failed to connect to PostgreSQL: {str(e)}")

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
        url (str): The URL of the Exa API.
        method (Literal[HTTPMethod.POST]): HTTP method used for the request, defaults to POST.
        api_key (str): The API key for authentication, fetched from the environment variable 'EXA_API_KEY'.
    """

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
        url (str): The URL of the Ollama API, defaults to "http://localhost:11434".
    """

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


class Jina(Http):
    """
    Connection class for Jina Scrape API.
    """

    api_key: str = Field(default_factory=partial(get_env_var, "JINA_API_KEY"))
    method: Literal[HTTPMethod.GET] = HTTPMethod.GET

    def connect(self):
        """
        Returns the requests module for making HTTP requests.
        """
        self.headers.update({"Authorization": f"Bearer {self.api_key}"})
        return super().connect()


class MySQL(BaseConnection):
    host: str = Field(default_factory=partial(get_env_var, "MYSQL_HOST", "localhost"))
    port: int = Field(default_factory=partial(get_env_var, "MYSQL_PORT", 3306))
    database: str = Field(default_factory=partial(get_env_var, "MYSQL_DATABASE", "db"))
    user: str = Field(default_factory=partial(get_env_var, "MYSQL_USER", "mysql"))
    password: str = Field(default_factory=partial(get_env_var, "MYSQL_PASSWORD", "password"))

    def connect(self):
        import mysql.connector

        try:
            conn = mysql.connector.connect(
                host=self.host, port=self.port, database=self.database, user=self.user, passwd=self.password
            )
            conn.autocommit = True
            logger.debug(
                f"Connected to MySQL with host={self.host}, " f"user={self.user}, " f"database={self.database}."
            )
            return conn
        except mysql.connector.Error as e:
            raise ConnectionError(f"Failed to connect to MySQL: {str(e)}")

    @property
    def cursor_params(self) -> dict:
        return {"dictionary": True}


class Snowflake(BaseConnection):
    user: str = Field(default_factory=partial(get_env_var, "SNOWFLAKE_USER", "snowflake"))
    password: str = Field(default_factory=partial(get_env_var, "SNOWFLAKE_PASSWORD", "password"))
    account: str = Field(default_factory=partial(get_env_var, "SNOWFLAKE_ACCOUNT", "account"))
    warehouse: str = Field(default_factory=partial(get_env_var, "SNOWFLAKE_WAREHOUSE", "warehouse"))
    database: str = Field(default_factory=partial(get_env_var, "SNOWFLAKE_DATABASE", "db"))
    schema: str = Field(default_factory=partial(get_env_var, "SNOWFLAKE_SCHEMA", "schema"))

    def connect(self):
        try:
            import snowflake.connector

            conn = snowflake.connector.connect(
                user=self.user,
                password=self.password,
                account=self.account,
                warehouse=self.warehouse,
                database=self.database,
                schema=self.schema,
            )
            logger.debug(
                f"Connected to Snowflake using account={self.account}, "
                f"warehouse={str(self.warehouse)}, user={self.user}, "
                f"database={self.database}, schema={self.schema}."
            )
            return conn
        except Exception as e:
            raise ConnectionError(f"Failed to connect to Snowflake: {str(e)}")

    @property
    def cursor_params(self) -> dict:
        from snowflake.connector import DictCursor

        return {"cursor_class": DictCursor}


class AWSRedshift(BaseConnection):
    host: str = Field(default_factory=partial(get_env_var, "AWS_REDSHIFT_HOST"))
    port: int = Field(default_factory=partial(get_env_var, "AWS_REDSHIFT_PORT", 5439))
    database: str = Field(default_factory=partial(get_env_var, "AWS_REDSHIFT_DATABASE", "db"))
    user: str = Field(default_factory=partial(get_env_var, "AWS_REDSHIFT_USER", "awsuser"))
    password: str = Field(default_factory=partial(get_env_var, "AWS_REDSHIFT_PASSWORD", "password"))

    def connect(self):
        try:
            import psycopg

            conn = psycopg.connect(
                host=self.host,
                port=self.port,
                dbname=self.database,
                user=self.user,
                password=self.password,
                client_encoding="utf-8",
                row_factory=psycopg.rows.dict_row,
            )
            conn.autocommit = True
            logger.debug(
                f"Connected to Amazon Redshift with host={self.host}, "
                f"port={str(self.port)}, user={self.user}, "
                f"database={self.database}."
            )
            return conn
        except Exception as e:
            raise ConnectionError(f"Failed to connect to Amazon Redshift : {str(e)}")


class Elasticsearch(BaseConnection):
    """
    Represents a connection to the Elasticsearch service.

    Attributes:
        url (str): The URL of the Elasticsearch service
        api_key (str): API key for authentication
        username (str): Username for basic authentication
        password (str): Password for basic authentication
        cloud_id (str): Cloud ID for Elastic Cloud deployment
        ca_path (str): Path to CA certificate for SSL verification
        verify_certs (bool): Whether to verify SSL certificates
        use_ssl (bool): Whether to use SSL for connection
    """

    url: str = Field(default_factory=partial(get_env_var, "ELASTICSEARCH_URL", None))
    api_key_id: str | None = Field(default_factory=partial(get_env_var, "ELASTICSEARCH_API_KEY_ID", None))
    api_key: str | None = Field(default_factory=partial(get_env_var, "ELASTICSEARCH_API_KEY", None))
    username: str | None = Field(default_factory=partial(get_env_var, "ELASTICSEARCH_USERNAME", None))
    password: str | None = Field(default_factory=partial(get_env_var, "ELASTICSEARCH_PASSWORD", None))
    cloud_id: str | None = Field(default_factory=partial(get_env_var, "ELASTICSEARCH_CLOUD_ID", None))
    ca_path: str | None = Field(default_factory=partial(get_env_var, "ELASTICSEARCH_CA_PATH", None))
    verify_certs: bool = Field(default_factory=partial(get_env_var, "ELASTICSEARCH_VERIFY_CERTS", False))
    use_ssl: bool = Field(default_factory=partial(get_env_var, "ELASTICSEARCH_USE_SSL", True))

    def connect(self):
        """
        Connects to the Elasticsearch service.

        Returns:
            elasticsearch.Elasticsearch: An instance of the Elasticsearch client.

        Raises:
            ImportError: If elasticsearch package is not installed
            ConnectionError: If connection fails
            ValueError: If neither API key nor basic auth credentials are provided
        """

        from elasticsearch import Elasticsearch
        from elasticsearch.exceptions import AuthenticationException

        # Build connection params
        conn_params = {}

        # Handle authentication
        if self.api_key is not None:
            if self.api_key_id is not None:
                conn_params["api_key"] = (self.api_key_id, self.api_key)
            else:
                conn_params["api_key"] = self.api_key
        elif self.username is not None and self.password is not None:
            conn_params["basic_auth"] = (self.username, self.password)
        elif self.cloud_id is None:  # Only require auth for non-cloud deployments
            raise ValueError("Either API key or username/password must be provided")

        # Handle SSL/TLS
        if self.use_ssl:
            if self.ca_path is not None:
                conn_params["ca_certs"] = self.ca_path
            conn_params["verify_certs"] = self.verify_certs

        # Handle cloud deployment
        if self.cloud_id is not None:
            conn_params["cloud_id"] = self.cloud_id
        else:
            conn_params["hosts"] = [self.url]

        # Create client
        try:
            es_client = Elasticsearch(**conn_params)
        except Exception as e:
            raise ConnectionError(f"Failed to connect to Elasticsearch: {str(e)}")

        if not es_client.ping():
            try:
                info = es_client.info()
            except AuthenticationException as e:
                info = f"Authentication error: {e}"
            raise ConnectionError(f"Failed to connect to Elasticsearch. {info}")

        logger.debug(f"Connected to Elasticsearch at {self.cloud_id or self.url}")
        return es_client


class xAI(BaseApiKeyConnection):
    api_key: str = Field(default_factory=partial(get_env_var, "XAI_API_KEY"))

    def connect(self):
        pass
