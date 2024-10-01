from typing import Any, Literal

from dynamiq.components.embedders.bedrock import (
    BedrockEmbedder as BedrockEmbedderComponent,
)
from dynamiq.connections import AWS as BedrockConnection
from dynamiq.connections.managers import ConnectionManager
from dynamiq.nodes.node import ConnectionNode, NodeGroup, ensure_config
from dynamiq.runnables import RunnableConfig
from dynamiq.utils.logger import logger


class BedrockDocumentEmbedder(ConnectionNode):
    """
    Provides functionality to compute embeddings for documents using Bedrock models.

    This class extends ConnectionNode to create embeddings for documents using Bedrock API.

    Attributes:
        group (Literal[NodeGroup.EMBEDDERS]): The group the node belongs to.
        name (str): The name of the node.
        connection (BedrockConnection | None): The connection to the Bedrock API.
        model (str): The model name to use for embedding.
        document_embedder (BedrockDocumentEmbedderComponent): The component for document embedding.

    Args:
        connection (Optional[BedrockConnection]): The connection to the Bedrock API. A new connection
            is created if none is provided.
        model (str): The model name to use for embedding. Defaults to 'amazon.titan-embed-text-v1'.
    """

    group: Literal[NodeGroup.EMBEDDERS] = NodeGroup.EMBEDDERS
    name: str = "AmazonBedrockDocumentEmbedder"
    connection: BedrockConnection | None = None
    model: str = "amazon.titan-embed-text-v1"
    document_embedder: BedrockEmbedderComponent = None

    def __init__(self, **kwargs):
        """
        Initializes the BedrockDocumentEmbedder.

        If neither client nor connection is provided in kwargs, a new BedrockConnection is created.

        Args:
            **kwargs: Arbitrary keyword arguments.
        """
        if kwargs.get("client") is None and kwargs.get("connection") is None:
            kwargs["connection"] = BedrockConnection()
        super().__init__(**kwargs)

    @property
    def to_dict_exclude_params(self):
        return super().to_dict_exclude_params | {"document_embedder": True}

    def init_components(
        self, connection_manager: ConnectionManager = ConnectionManager()
    ):
        """
        Initializes the components of the BedrockDocumentEmbedder.

        This method sets up the document_embedder component if it hasn't been initialized yet.

        Args:
            connection_manager (ConnectionManager): The connection manager to use. Defaults to a new
                ConnectionManager instance.
        """
        super().init_components(connection_manager)
        if self.document_embedder is None:
            self.document_embedder = BedrockEmbedderComponent(
                connection=self.connection, model=self.model, client=self.client
            )

    def execute(
        self, input_data: dict[str, Any], config: RunnableConfig = None, **kwargs
    ):
        """
        Executes the document embedding process.

        This method takes input documents, computes their embeddings using the Bedrock API, and returns the result.

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
        logger.debug("BedrockDocumentEmbedder executed successfully.")

        return output


class BedrockTextEmbedder(ConnectionNode):
    """
    A component designed to embed strings using specified Cohere models.

    This class extends ConnectionNode to provide text embedding functionality using Bedrock API.

    Args:
        connection (Optional[BedrockConnection]): An existing connection to Bedrock API. If not
            provided, a new connection will be established using environment variables.
        model (str): The identifier of the Bedrock model for text embeddings. Defaults to
            'amazon.titan-embed-text-v1'.

    Attributes:
        group (Literal[NodeGroup.EMBEDDERS]): The group the node belongs to.
        name (str): The name of the node.
        connection (BedrockConnection | None): The connection to Bedrock API.
        model (str): The Bedrock model identifier for text embeddings.
        text_embedder (BedrockTextEmbedderComponent): The component for text embedding.

    """

    group: Literal[NodeGroup.EMBEDDERS] = NodeGroup.EMBEDDERS
    name: str = "BedrockTextEmbedder"
    connection: BedrockConnection | None = None
    model: str = "amazon.titan-embed-text-v1"
    text_embedder: BedrockEmbedderComponent = None

    def __init__(self, **kwargs):
        """Initialize the BedrockTextEmbedder.

        If neither client nor connection is provided in kwargs, a new BedrockConnection is created.

        Args:
            **kwargs: Keyword arguments to initialize the node.
        """
        if kwargs.get("client") is None and kwargs.get("connection") is None:
            kwargs["connection"] = BedrockConnection()
        super().__init__(**kwargs)

    @property
    def to_dict_exclude_params(self):
        return super().to_dict_exclude_params | {"text_embedder": True}

    def init_components(
        self, connection_manager: ConnectionManager = ConnectionManager()
    ):
        """
        Initialize the components of the BedrockTextEmbedder.

        This method sets up the text_embedder component if it hasn't been initialized yet.

        Args:
            connection_manager (ConnectionManager): The connection manager to use. Defaults to a new
                ConnectionManager instance.
        """
        super().init_components(connection_manager)
        if self.text_embedder is None:
            self.text_embedder = BedrockEmbedderComponent(
                connection=self.connection, model=self.model, client=self.client
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
        logger.debug(f"BedrockTextEmbedder: {output['meta']}")
        return {
            "embedding": output["embedding"],
            "query": input_data["query"],
        }
