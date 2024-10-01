from typing import Any, Literal

from dynamiq.components.embedders.cohere import (
    CohereEmbedder as CohereEmbedderComponent,
)
from dynamiq.connections import Cohere as CohereConnection
from dynamiq.connections.managers import ConnectionManager
from dynamiq.nodes.node import ConnectionNode, NodeGroup, ensure_config
from dynamiq.runnables import RunnableConfig
from dynamiq.utils.logger import logger


class CohereDocumentEmbedder(ConnectionNode):
    """
    Provides functionality to compute embeddings for documents using Cohere models.

    This class extends ConnectionNode to create embeddings for documents using Cohere API.

    Attributes:
        group (Literal[NodeGroup.EMBEDDERS]): The group the node belongs to.
        name (str): The name of the node.
        connection (CohereConnection | None): The connection to the Cohere API.
        model (str): The model name to use for embedding.
        document_embedder (CohereDocumentEmbedderComponent): The component for document embedding.

    Args:
        connection (Optional[CohereConnection]): The connection to the Cohere API. A new connection
            is created if none is provided.
        model (str): The model name to use for embedding. Defaults to 'cohere/embed-english-v2.0'.
    """

    group: Literal[NodeGroup.EMBEDDERS] = NodeGroup.EMBEDDERS
    name: str = "CohereDocumentEmbedder"
    connection: CohereConnection | None = None
    model: str = "cohere/embed-english-v2.0"
    document_embedder: CohereEmbedderComponent = None

    def __init__(self, **kwargs):
        """
        Initializes the CohereDocumentEmbedder.

        If neither client nor connection is provided in kwargs, a new CohereConnection is created.

        Args:
            **kwargs: Arbitrary keyword arguments.
        """
        if kwargs.get("client") is None and kwargs.get("connection") is None:
            kwargs["connection"] = CohereConnection()
        super().__init__(**kwargs)

    @property
    def to_dict_exclude_params(self):
        return super().to_dict_exclude_params | {"document_embedder": True}

    def init_components(
        self, connection_manager: ConnectionManager = ConnectionManager()
    ):
        """
        Initializes the components of the CohereDocumentEmbedder.

        This method sets up the document_embedder component if it hasn't been initialized yet.

        Args:
            connection_manager (ConnectionManager): The connection manager to use. Defaults to a new
                ConnectionManager instance.
        """
        super().init_components(connection_manager)
        if self.document_embedder is None:
            self.document_embedder = CohereEmbedderComponent(
                connection=self.connection, model=self.model, client=self.client
            )

    def execute(
        self, input_data: dict[str, Any], config: RunnableConfig = None, **kwargs
    ):
        """
        Executes the document embedding process.

        This method takes input documents, computes their embeddings using the Cohere API, and returns the result.

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
        logger.debug("CohereDocumentEmbedder executed successfully.")

        return output


class CohereTextEmbedder(ConnectionNode):
    """
    A component designed to embed strings using specified Cohere models.

    This class extends ConnectionNode to provide text embedding functionality using litellm embedding.

    Args:
        connection (Optional[CohereConnection]): An existing connection to Cohere API. If not
            provided, a new connection will be established using environment variables.
        model (str): The identifier of the Cohere model for text embeddings. Defaults to
            'cohere/embed-english-v2.0'.

    Attributes:
        group (Literal[NodeGroup.EMBEDDERS]): The group the node belongs to.
        name (str): The name of the node.
        connection (CohereConnection | None): The connection to Cohere API.
        model (str): The Cohere model identifier for text embeddings.
        text_embedder (CohereTextEmbedderComponent): The component for text embedding.

    """

    group: Literal[NodeGroup.EMBEDDERS] = NodeGroup.EMBEDDERS
    name: str = "CohereTextEmbedder"
    connection: CohereConnection | None = None
    model: str = "cohere/embed-english-v2.0"
    text_embedder: CohereEmbedderComponent = None

    def __init__(self, **kwargs):
        """
        Initialize the CohereTextEmbedder.

        If neither client nor connection is provided in kwargs, a new CohereConnection is created.

        Args:
            **kwargs: Keyword arguments to initialize the node.
        """
        if kwargs.get("client") is None and kwargs.get("connection") is None:
            kwargs["connection"] = CohereConnection()
        super().__init__(**kwargs)

    @property
    def to_dict_exclude_params(self):
        return super().to_dict_exclude_params | {"text_embedder": True}

    def init_components(
        self, connection_manager: ConnectionManager = ConnectionManager()
    ):
        """
        Initialize the components of the CohereTextEmbedder.

        This method sets up the text_embedder component if it hasn't been initialized yet.

        Args:
            connection_manager (ConnectionManager): The connection manager to use. Defaults to a new
                ConnectionManager instance.
        """
        super().init_components(connection_manager)
        if self.text_embedder is None:
            self.text_embedder = CohereEmbedderComponent(
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
        logger.debug(f"CohereTextEmbedder: {output['meta']}")
        return {
            "embedding": output["embedding"],
            "query": input_data["query"],
        }
