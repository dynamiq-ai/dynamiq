from typing import Any, Literal

from dynamiq.components.embedders.huggingface import (
    HuggingFaceEmbedder as HuggingFaceEmbedderComponent,
)
from dynamiq.connections import HuggingFace as HuggingFaceConnection
from dynamiq.connections.managers import ConnectionManager
from dynamiq.nodes.node import ConnectionNode, NodeGroup, ensure_config
from dynamiq.runnables import RunnableConfig
from dynamiq.utils.logger import logger


class HuggingFaceDocumentEmbedder(ConnectionNode):
    """
    Provides functionality to compute embeddings for documents using HuggingFace models.

    This class extends ConnectionNode to create embeddings for documents using litellm embedding.

    Attributes:
        group (Literal[NodeGroup.EMBEDDERS]): The group the node belongs to.
        name (str): The name of the node.
        connection (HuggingFaceConnection | None): The connection to the HuggingFace API.
        model (str): The model name to use for embedding.
        document_embedder (HuggingFaceDocumentEmbedderComponent): The component for document embedding.

    Args:
        connection (Optional[HuggingFaceConnection]): The connection to the HuggingFace API. A new connection
            is created if none is provided.
        model (str): The model name to use for embedding. Defaults to 'huggingface/microsoft/codebert-base'.
    """

    group: Literal[NodeGroup.EMBEDDERS] = NodeGroup.EMBEDDERS
    name: str = "HuggingFaceDocumentEmbedder"
    connection: HuggingFaceConnection | None = None
    model: str = "huggingface/BAAI/bge-large-zh"
    document_embedder: HuggingFaceEmbedderComponent = None

    def __init__(self, **kwargs):
        """
        Initializes the HuggingFaceDocumentEmbedder.

        If neither client nor connection is provided in kwargs, a new HuggingFaceConnection is created.

        Args:
            **kwargs: Arbitrary keyword arguments.
        """
        if kwargs.get("client") is None and kwargs.get("connection") is None:
            kwargs["connection"] = HuggingFaceConnection()
        super().__init__(**kwargs)

    @property
    def to_dict_exclude_params(self):
        return super().to_dict_exclude_params | {"document_embedder": True}

    def init_components(
        self, connection_manager: ConnectionManager = ConnectionManager()
    ):
        """
        Initializes the components of the HuggingFaceDocumentEmbedder.

        This method sets up the document_embedder component if it hasn't been initialized yet.

        Args:
            connection_manager (ConnectionManager): The connection manager to use. Defaults to a new
                ConnectionManager instance.
        """
        super().init_components(connection_manager)
        if self.document_embedder is None:
            self.document_embedder = HuggingFaceEmbedderComponent(
                connection=self.connection, model=self.model, client=self.client
            )

    def execute(
        self, input_data: dict[str, Any], config: RunnableConfig = None, **kwargs
    ):
        """
        Executes the document embedding process.

        This method takes input documents, computes their embeddings using the litellm embedding HuggingFace , and
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
        logger.debug("HuggingFaceDocumentEmbedder executed successfully.")

        return output


class HuggingFaceTextEmbedder(ConnectionNode):
    """
    A component designed to embed strings using specified HuggingFace models.

    This class extends ConnectionNode to provide text embedding functionality using litellm embedding.

    Args:
        connection (Optional[HuggingFaceConnection]): An existing connection to HuggingFace's API. If not
            provided, a new connection will be established using environment variables.
        model (str): The identifier of the HuggingFace model for text embeddings. Defaults to
            'huggingface/microsoft/codebert-base'.

    Attributes:
        group (Literal[NodeGroup.EMBEDDERS]): The group the node belongs to.
        name (str): The name of the node.
        connection (HuggingFaceConnection | None): The connection to HuggingFace API.
        model (str): The HuggingFace model identifier for text embeddings.
        text_embedder (HuggingFaceTextEmbedderComponent): The component for text embedding.

    """

    group: Literal[NodeGroup.EMBEDDERS] = NodeGroup.EMBEDDERS
    name: str = "HuggingFaceTextEmbedder"
    connection: HuggingFaceConnection | None = None
    model: str = "huggingface/microsoft/codebert-base"
    text_embedder: HuggingFaceEmbedderComponent = None

    def __init__(self, **kwargs):
        """
        Initialize the HuggingFaceTextEmbedder.

        If neither client nor connection is provided in kwargs, a new HuggingFaceConnection is created.

        Args:
            **kwargs: Keyword arguments to initialize the node.
        """
        if kwargs.get("client") is None and kwargs.get("connection") is None:
            kwargs["connection"] = HuggingFaceConnection()
        super().__init__(**kwargs)

    @property
    def to_dict_exclude_params(self):
        return super().to_dict_exclude_params | {"text_embedder": True}

    def init_components(
        self, connection_manager: ConnectionManager = ConnectionManager()
    ):
        """
        Initialize the components of the HuggingFaceTextEmbedder.

        This method sets up the text_embedder component if it hasn't been initialized yet.

        Args:
            connection_manager (ConnectionManager): The connection manager to use. Defaults to a new
                ConnectionManager instance.
        """
        super().init_components(connection_manager)
        if self.text_embedder is None:
            self.text_embedder = HuggingFaceEmbedderComponent(
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
        logger.debug(f"HuggingFaceTextEmbedder: {output['meta']}")
        return {
            "embedding": output["embedding"],
            "query": input_data["query"],
        }
