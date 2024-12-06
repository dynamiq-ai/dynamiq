from typing import Any, ClassVar, Literal

from pydantic import BaseModel, Field

from dynamiq.connections.managers import ConnectionManager
from dynamiq.nodes import ErrorHandling, Node
from dynamiq.nodes.node import ConnectionNode, NodeGroup
from dynamiq.types import Document
from dynamiq.utils.logger import logger


class RetrievalInputSchema(BaseModel):
    query: str = Field(..., description="Parameter to provide a query to retrieve documents.")


class RetrievalTool(Node):
    """Tool for retrieving relevant documents based on a query.

    Attributes:
        group (Literal[NodeGroup.TOOLS]): Group for the node. Defaults to NodeGroup.TOOLS.
        name (str): Name of the tool. Defaults to "Retrieval Tool".
        description (str): Description of the tool.
        error_handling (ErrorHandling): Error handling configuration.
        connection_manager (ConnectionManager | None): Connection manager.
        text_embedder (ConnectionNode | None): Text embedder node.
        document_retriever (ConnectionNode | None): Document retriever node.
    """
    group: Literal[NodeGroup.TOOLS] = NodeGroup.TOOLS
    name: str = "Retrieval Tool"
    description: str = "A tool for retrieving relevant documents based on a query."
    error_handling: ErrorHandling = Field(default_factory=lambda: ErrorHandling(timeout_seconds=600))
    connection_manager: ConnectionManager | None = None
    text_embedder: ConnectionNode | None = None
    document_retriever: ConnectionNode | None = None
    input_schema: ClassVar[type[RetrievalInputSchema]] = RetrievalInputSchema

    def __init__(
        self,
        connection_manager: ConnectionManager | None = None,
        text_embedder: ConnectionNode | None = None,
        document_retriever: Any | None = None,
        **data,
    ):
        """Initialize the RetrievalTool.

        Args:
            connection_manager (ConnectionManager | None): Connection manager.
            text_embedder (ConnectionNode | None): Text embedder node.
            document_retriever (Any | None): Document retriever node.
            **data: Additional keyword arguments.
        """
        super().__init__(**data)
        self.connection_manager = connection_manager
        self.text_embedder = text_embedder
        self.document_retriever = document_retriever

    def init_components(self, connection_manager: ConnectionManager | None = None):
        """Initialize components with the connection manager.

        Args:
            connection_manager (ConnectionManager, optional): connection manager. Defaults to ConnectionManager.
        """
        connection_manager = connection_manager or ConnectionManager()
        super().init_components(connection_manager)
        if (
            hasattr(self.text_embedder, "is_postponed_component_init")
            and self.text_embedder.is_postponed_component_init
        ):
            self.text_embedder.init_components(connection_manager)
        if (
            hasattr(self.document_retriever, "is_postponed_component_init")
            and self.document_retriever.is_postponed_component_init
        ):
            self.document_retriever.init_components(connection_manager)

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

    def execute(self, input_data: RetrievalInputSchema, **_) -> dict[str, Any]:
        """Execute the retrieval tool.

        Args:
            input_data (dict[str, Any]): Input data for the tool.
            **kwargs: Additional keyword arguments.

        Returns:
            dict[str, Any]: Result of the retrieval.
        """

        logger.info(f"Tool {self.name} - {self.id}: started with INPUT DATA:\n{input_data.model_dump()}")

        if not self.text_embedder:
            raise ValueError(f"{self.name}: Text embedder is not initialized.")
        if not self.document_retriever:
            raise ValueError(f"{self.name}: Document retriever is not initialized.")

        try:
            text_embedder_output = self.text_embedder.run(input_data={"query": input_data.query})
            embedding = text_embedder_output.output.get("embedding")

            document_retriever_output = self.document_retriever.run(input_data={"embedding": embedding})
            retrieved_documents = document_retriever_output.output.get("documents", [])
            logger.debug(f"Tool {self.name} - {self.id}: retrieved {len(retrieved_documents)} documents")

            result = self.format_content(retrieved_documents)
            logger.info(f"Tool {self.name} - {self.id}: finished with RESULT:\n{str(result)[:200]}...")

            return {"content": result}
        except Exception as e:
            logger.error(f"Tool {self.name} - {self.id}: execution error: {str(e)}", exc_info=True)
            raise

    def to_dict(self, **kwargs) -> dict:
        """Convert the RetrievalTool object to a dictionary.

        Args:
            **kwargs: Additional keyword arguments.

        Returns:
            dict: Dictionary representation of the object.
        """
        data = super().to_dict(**kwargs)
        data.pop("text_embedder", None)
        data.pop("document_retriever", None)
        return data
