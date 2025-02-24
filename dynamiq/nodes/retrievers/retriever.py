from typing import Any, ClassVar, Literal

from pydantic import BaseModel, Field

from dynamiq.connections.managers import ConnectionManager
from dynamiq.nodes import ErrorHandling, Node
from dynamiq.nodes.agents.exceptions import ToolExecutionException
from dynamiq.nodes.embedders.base import TextEmbedder
from dynamiq.nodes.node import NodeDependency, NodeGroup, ensure_config
from dynamiq.nodes.retrievers.base import Retriever
from dynamiq.runnables import RunnableConfig
from dynamiq.types import Document
from dynamiq.utils.logger import logger


class VectorStoreRetrieverInputSchema(BaseModel):
    query: str = Field(..., description="Parameter to provide a query to retrieve documents.")
    alpha: float = Field(default=0.0, description="Parameter to provide alpha for hybrid retrieval.")
    filters: dict[str, Any] = Field(
        default_factory=dict, description="Parameter to provide filters to apply for retrieving specific documents."
    )
    top_k: int = Field(default=0, description="Parameter to provided how many documents to retrieve.")


class VectorStoreRetriever(Node):
    """Node for retrieving relevant documents based on a query.

    Attributes:
        group (Literal[NodeGroup.TOOLS]): Group for the node. Defaults to NodeGroup.TOOLS.
        name (str): Name of the tool. Defaults to "Retrieval Tool".
        description (str): Description of the tool.
        error_handling (ErrorHandling): Error handling configuration.
        text_embedder (TextEmbedder): Text embedder node.
        document_retriever (Retriever): Document retriever node.
        filters (dict[str, Any] | None): Filters for document retrieval.
        top_k (int): The maximum number of documents to return.
        alpha (float): The alpha parameter for hybrid retrieval.
    """

    group: Literal[NodeGroup.TOOLS] = NodeGroup.TOOLS
    name: str = "VectorStore Retriever"
    description: str = "A node for retrieving relevant documents based on a query."
    error_handling: ErrorHandling = Field(default_factory=lambda: ErrorHandling(timeout_seconds=600))
    text_embedder: TextEmbedder
    document_retriever: Retriever
    filters: dict[str, Any] = {}
    top_k: int = 0
    alpha: float = 0.0

    input_schema: ClassVar[type[VectorStoreRetrieverInputSchema]] = VectorStoreRetrieverInputSchema

    def __init__(self, **kwargs):
        """
        Initializes the VectorStoreRetriever with the given parameters.

        Args:
            **kwargs: Additional keyword arguments to be passed to the parent class constructor.
        """
        super().__init__(**kwargs)
        self._run_depends = []

    def reset_run_state(self):
        """
        Reset the intermediate steps (run_depends) of the node.
        """
        self._run_depends = []

    def init_components(self, connection_manager: ConnectionManager | None = None) -> None:
        """
        Initialize the components of the tool.

        Args:
            connection_manager (ConnectionManager, optional): connection manager. Defaults to ConnectionManager.
        """
        connection_manager = connection_manager or ConnectionManager()
        super().init_components(connection_manager)
        if self.text_embedder.is_postponed_component_init:
            self.text_embedder.init_components(connection_manager)
        if self.document_retriever.is_postponed_component_init:
            self.document_retriever.init_components(connection_manager)

    @property
    def to_dict_exclude_params(self):
        """
        Property to define which parameters should be excluded when converting the class instance to a dictionary.

        Returns:
            dict: A dictionary defining the parameters to exclude.
        """
        return super().to_dict_exclude_params | {"text_embedder": True, "document_retriever": True}

    def to_dict(self, **kwargs) -> dict:
        """Converts the instance to a dictionary.

        Returns:
            dict: A dictionary representation of the instance.
        """
        data = super().to_dict(**kwargs)
        data["text_embedder"] = self.text_embedder.to_dict(**kwargs)
        data["document_retriever"] = self.document_retriever.to_dict(**kwargs)
        return data

    def format_content(self, documents: list[Document], metadata_fields: list[str] | None = None) -> str:
        """Format the retrieved documents' metadata and content.

        Args:
            documents (list[Document]): List of retrieved documents.
            metadata_fields (list[str]): Metadata fields to include.

        Returns:
            str: Formatted content of the documents.
        """
        metadata_fields = metadata_fields or ["title", "url"]
        formatted_docs = []
        for i, doc in enumerate(documents):
            metadata = doc.metadata
            formatted_doc = f"Source {i + 1}\n"
            for field in metadata_fields:
                if field in metadata:
                    formatted_doc += f"{field.capitalize()}: {metadata[field]}\n"
            formatted_doc += f"Content: {doc.content}\n"
            formatted_docs.append(formatted_doc)
        return "\n\n".join(formatted_docs)

    def execute(
        self, input_data: VectorStoreRetrieverInputSchema, config: RunnableConfig | None = None, **kwargs
    ) -> dict[str, Any]:
        """Execute the retrieval tool.

        Args:
            input_data (dict[str, Any]): Input data for the tool.
            config (RunnableConfig, optional): Configuration for the runnable, including callbacks.
            **kwargs: Additional keyword arguments.

        Returns:
            dict[str, Any]: Result of the retrieval.
        """

        logger.info(f"Tool {self.name} - {self.id}: started with INPUT DATA:\n{input_data.model_dump()}")
        config = ensure_config(config)
        self.reset_run_state()
        self.run_on_node_execute_run(config.callbacks, **kwargs)

        filters = input_data.filters or self.filters
        top_k = input_data.top_k or self.top_k

        alpha = input_data.alpha or self.alpha
        query = input_data.query
        try:
            kwargs = kwargs | {"parent_run_id": kwargs.get("run_id")}
            kwargs.pop("run_depends", None)
            text_embedder_output = self.text_embedder.run(
                input_data={"query": query}, run_depends=self._run_depends, config=config, **kwargs
            )
            self._run_depends = [NodeDependency(node=self.text_embedder).to_dict()]
            embedding = text_embedder_output.output.get("embedding")

            document_retriever_output = self.document_retriever.run(
                input_data={
                    "embedding": embedding,
                    "top_k": top_k,
                    "filters": filters,
                    "alpha": alpha,
                    **({"query": query} if alpha else {}),
                },
                run_depends=self._run_depends,
                config=config,
                **kwargs,
            )
            self._run_depends = [NodeDependency(node=self.document_retriever).to_dict()]
            retrieved_documents = document_retriever_output.output.get("documents", [])
            logger.debug(f"Tool {self.name} - {self.id}: retrieved {len(retrieved_documents)} documents")

            result = self.format_content(retrieved_documents)
            logger.info(f"Tool {self.name} - {self.id}: finished with RESULT:\n{str(result)[:200]}...")

            return {"content": result, "documents": retrieved_documents}
        except Exception as e:
            logger.error(f"Tool {self.name} - {self.id}: execution error: {str(e)}", exc_info=True)
            raise ToolExecutionException(
                f"Tool '{self.name}' failed to retrieve data using the specified action. "
                f"Error: {str(e)}. Please analyze the error and take appropriate action.",
                recoverable=True,
            )
