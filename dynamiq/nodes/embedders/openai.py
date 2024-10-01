from typing import Any, Literal

from dynamiq.components.embedders.openai import (
    OpenAIEmbedder as OpenAIEmbedderComponent,
)
from dynamiq.connections import OpenAI as OpenAIConnection
from dynamiq.connections.managers import ConnectionManager
from dynamiq.nodes.node import ConnectionNode, NodeGroup, ensure_config
from dynamiq.runnables import RunnableConfig
from dynamiq.utils.logger import logger


class OpenAIDocumentEmbedder(ConnectionNode):
    """
    Provides functionality to compute embeddings for documents using OpenAI's models.

    This class extends ConnectionNode to create embeddings for documents using OpenAI's API.

    Attributes:
        group (Literal[NodeGroup.EMBEDDERS]): The group the node belongs to.
        name (str): The name of the node.
        connection (OpenAIConnection | None): The connection to the OpenAI API.
        client (OpenAIClient | None): The OpenAI client instance.
        model (str): The model name to use for embedding.
        dimensions (int | None): The number of dimensions for the output embeddings.
        document_embedder (OpenAIDocumentEmbedderComponent): The component for document embedding.

    Args:
        connection (Optional[OpenAIConnection]): The connection to the OpenAI API. A new connection
            is created if none is provided.
        model (str): The model name to use for embedding. Defaults to 'text-embedding-3-small'.
        dimensions (Optional[int]): The number of dimensions for the output embeddings. Supported
            only by 'text-embedding-3' and later models. Defaults to None.
    """

    group: Literal[NodeGroup.EMBEDDERS] = NodeGroup.EMBEDDERS
    name: str = "OpenAIDocumentEmbedder"
    connection: OpenAIConnection | None = None
    model: str = "text-embedding-3-small"
    dimensions: int | None = None
    document_embedder: OpenAIEmbedderComponent = None

    def __init__(self, **kwargs):
        """
        Initializes the OpenAIDocumentEmbedder.

        If neither client nor connection is provided in kwargs, a new OpenAIConnection is created.

        Args:
            **kwargs: Arbitrary keyword arguments.
        """
        if kwargs.get("client") is None and kwargs.get("connection") is None:
            kwargs["connection"] = OpenAIConnection()
        super().__init__(**kwargs)

    @property
    def to_dict_exclude_params(self):
        return super().to_dict_exclude_params | {"document_embedder": True}

    def init_components(
        self, connection_manager: ConnectionManager = ConnectionManager()
    ):
        """
        Initializes the components of the OpenAIDocumentEmbedder.

        This method sets up the document_embedder component if it hasn't been initialized yet.

        Args:
            connection_manager (ConnectionManager): The connection manager to use. Defaults to a new
                ConnectionManager instance.
        """
        super().init_components(connection_manager)
        if self.document_embedder is None:
            self.document_embedder = OpenAIEmbedderComponent(
                connection=self.connection,
                model=self.model,
                dimensions=self.dimensions,
                client=self.client,
            )

    def execute(
        self, input_data: dict[str, Any], config: RunnableConfig = None, **kwargs
    ):
        """
        Executes the document embedding process.

        This method takes input documents, computes their embeddings using the OpenAI API, and
        returns the result.

        Args:
            input_data (dict[str, Any]): A dictionary containing the input data. Expected to have a
                'documents' key with the documents to embed.
            config (RunnableConfig, optional): Configuration for the execution. Defaults to None.
            **kwargs: Additional keyword arguments.

        Returns:
            The output from the document_embedder component, typically the computed embeddings.
        """
        config = ensure_config(config)
        self.run_on_node_execute_run(config.callbacks, **kwargs)

        output = self.document_embedder.embed_documents(input_data["documents"])
        logger.debug("OpenAIDocumentEmbedder executed successfully.")

        return output


class OpenAITextEmbedder(ConnectionNode):
    """
    A component designed to embed strings using specified OpenAI models.

    This class extends ConnectionNode to provide text embedding functionality using OpenAI's API.

    Args:
        connection (Optional[OpenAIConnection]): An existing connection to OpenAI's API. If not
            provided, a new connection will be established using environment variables.
        model (str): The identifier of the OpenAI model for text embeddings. Defaults to
            'text-embedding-3-small'.
        dimensions (Optional[int]): Desired dimensionality of output embeddings. Defaults to None,
            using the model's default output dimensionality.

    Attributes:
        group (Literal[NodeGroup.EMBEDDERS]): The group the node belongs to.
        name (str): The name of the node.
        connection (OpenAIConnection | None): The connection to OpenAI's API.
        client (OpenAIClient | None): The OpenAI client instance.
        model (str): The OpenAI model identifier for text embeddings.
        dimensions (int | None): The desired dimensionality of output embeddings.
        text_embedder (OpenAITextEmbedderComponent): The component for text embedding.

    Notes:
        The `dimensions` parameter is model-dependent and may not be supported by all models.
    """

    group: Literal[NodeGroup.EMBEDDERS] = NodeGroup.EMBEDDERS
    name: str = "OpenAITextEmbedder"
    connection: OpenAIConnection | None = None
    model: str = "text-embedding-3-small"
    dimensions: int | None = None
    text_embedder: OpenAIEmbedderComponent = None

    def __init__(self, **kwargs):
        """Initialize the OpenAITextEmbedder.

        If neither client nor connection is provided in kwargs, a new OpenAIConnection is created.

        Args:
            **kwargs: Keyword arguments to initialize the node.
        """
        if kwargs.get("client") is None and kwargs.get("connection") is None:
            kwargs["connection"] = OpenAIConnection()
        super().__init__(**kwargs)

    @property
    def to_dict_exclude_params(self):
        return super().to_dict_exclude_params | {"text_embedder": True}

    def init_components(
        self, connection_manager: ConnectionManager = ConnectionManager()
    ):
        """
        Initialize the components of the OpenAITextEmbedder.

        This method sets up the text_embedder component if it hasn't been initialized yet.

        Args:
            connection_manager (ConnectionManager): The connection manager to use. Defaults to a new
                ConnectionManager instance.
        """
        super().init_components(connection_manager)
        if self.text_embedder is None:
            self.text_embedder = OpenAIEmbedderComponent(
                connection=self.connection,
                model=self.model,
                dimensions=self.dimensions,
                client=self.client,
            )

    def execute(
        self, input_data: dict[str, Any], config: RunnableConfig = None, **kwargs
    ):
        """
        Execute the text embedding process.

        This method takes input data, runs the text embedding, and returns the result.

        Args:
            input_data (dict[str, Any]): The input data containing the query to embed.
            config (RunnableConfig, optional): Configuration for the execution. Defaults to None.
            **kwargs: Additional keyword arguments.

        Returns:
            dict: A dictionary containing the embedding and the original query.
        """
        config = ensure_config(config)
        self.run_on_node_execute_run(config.callbacks, **kwargs)

        output = self.text_embedder.embed_text(input_data["query"])
        logger.debug(f"OpenAITextEmbedder: {output['meta']}")
        return {
            "embedding": output["embedding"],
            "query": input_data["query"],
        }
