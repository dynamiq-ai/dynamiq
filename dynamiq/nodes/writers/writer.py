from typing import Any, ClassVar, Literal

from pydantic import BaseModel, Field, model_validator

from dynamiq.connections.managers import ConnectionManager
from dynamiq.nodes import ErrorHandling, Node
from dynamiq.nodes.agents.exceptions import ToolExecutionException
from dynamiq.nodes.embedders.base import DocumentEmbedder
from dynamiq.nodes.node import NodeDependency, NodeGroup, ensure_config
from dynamiq.nodes.writers.base import Writer
from dynamiq.runnables import RunnableConfig
from dynamiq.types import Document
from dynamiq.utils.logger import logger

DESCRIPTION_VECTOR_STORE_WRITER = """Writes documents (or text) to a vector store.

Key Capabilities:
- Adding textual content to the vector store as separate database entries
- Adding metadata to the vector store entries

Parameter Guide:
- documents: List of strings to write to the vector store along with their metadata.

Guildelines:
- The vector story entry metadata may consist of the following fields (but is not limited to):
    - url
    - title
    - description
    - author
    - published_date
    - source
    - category
    - tags
    - etc.
- If any metadata field is provided by the user, it should be included in the vector store entry metadata.
- The input documents should be a list of dictionaries with the following structure:
    - { "documents": [{"content": "<content of the vector store entry>","metadata": {"<field name>": "<field value>"}}]}

Examples:
- {
    "documents": [
        {
            "content": "Artificial intelligence is transforming healthcare by improving diagnostics and patient care.",
            "metadata": {
                "title": "AI in Healthcare",
                "author": "Jane Doe",
                "published_date": "2025-09-10",
                "source": "Nature Medicine",
                "url": "https://www.nature.com/articles/ai-healthcare",
                "category": "Healthcare",
                "tags": ["AI", "medicine", "technology"]
            }
        }
    ]
}
- {
    "documents": [
        {
            "content": "OpenAI has announced a new framework for autonomous agents capable of reasoning and planning.",
            "metadata": {
                "title": "Next-Gen AI Agents",
                "author": "OpenAI Research Team",
                "published_date": "2025-07-01",
                "category": "Artificial Intelligence",
                "tags": ["AI", "agents", "research"]
            }
        }
    ]
}
"""


class VectorStoreWriterInputSchema(BaseModel):
    documents: list[Document] | list[dict] = Field(
        ...,
        description="Parameter to provide documents to write to the vector store.",
    )

    @model_validator(mode="after")
    def validate_input_documents(self):
        """
        Validate the input documents by converting list of dictionaries
        to Documents (when using inside an agent) and ensuring metadata is never None.
        """
        if self.documents:
            if isinstance(self.documents[0], dict):
                converted_docs = []
                for doc_dict in self.documents:
                    if not doc_dict.get("content", ""):
                        raise ValueError("Document dict must contain 'content' field")
                    if not doc_dict.get("metadata", {}):
                        doc_dict["metadata"] = {}
                    converted_docs.append(Document(**doc_dict))
                self.documents = converted_docs
            elif isinstance(self.documents[0], Document):
                for doc in self.documents:
                    doc.metadata = doc.metadata or {}
        return self


class VectorStoreWriter(Node):
    """Node for writing documents to a vector store.

    Attributes:
        group (Literal[NodeGroup.TOOLS]): Group for the node. Defaults to NodeGroup.TOOLS.
        name (str): Name of the tool. Defaults to "VectorStore Writer".
        description (str): Description of the tool.
        error_handling (ErrorHandling): Error handling configuration.
        document_embedder (DocumentEmbedder): Document embedder node.
        document_writer (Writer): Document writer node.
    """

    group: Literal[NodeGroup.TOOLS] = NodeGroup.TOOLS
    name: str = "VectorStore Writer"
    description: str = DESCRIPTION_VECTOR_STORE_WRITER
    error_handling: ErrorHandling = Field(default_factory=lambda: ErrorHandling(timeout_seconds=600))
    document_embedder: DocumentEmbedder
    document_writer: Writer

    input_schema: ClassVar[type[VectorStoreWriterInputSchema]] = VectorStoreWriterInputSchema

    def __init__(self, **kwargs):
        """
        Initializes the VectorStoreWriter with the given parameters.

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
        if self.document_embedder.is_postponed_component_init:
            self.document_embedder.init_components(connection_manager)
        if self.document_writer.is_postponed_component_init:
            self.document_writer.init_components(connection_manager)

    @property
    def to_dict_exclude_params(self):
        """
        Property to define which parameters should be excluded when converting the class instance to a dictionary.

        Returns:
            dict: A dictionary defining the parameters to exclude.
        """
        return super().to_dict_exclude_params | {"document_embedder": True, "document_writer": True}

    def to_dict(self, **kwargs) -> dict:
        """Converts the instance to a dictionary.

        Returns:
            dict: A dictionary representation of the instance.
        """
        data = super().to_dict(**kwargs)
        data["document_embedder"] = self.document_embedder.to_dict(**kwargs)
        data["document_writer"] = self.document_writer.to_dict(**kwargs)
        return data

    def execute(
        self, input_data: VectorStoreWriterInputSchema, config: RunnableConfig | None = None, **kwargs
    ) -> dict[str, Any]:
        """Execute the vector store writer tool.

        Args:
            input_data (VectorStoreWriterInputSchema): Input data for the tool.
            config (RunnableConfig, optional): Configuration for the runnable, including callbacks.
            **kwargs: Additional keyword arguments.

        Returns:
            dict[str, Any]: Result of the writing operation.
        """

        logger.info(f"Tool {self.name} - {self.id}: started with INPUT DATA:\n{input_data.model_dump()}")
        config = ensure_config(config)
        self.reset_run_state()
        self.run_on_node_execute_run(config.callbacks, **kwargs)

        documents = input_data.documents

        try:
            kwargs = kwargs | {"parent_run_id": kwargs.get("run_id")}
            kwargs.pop("run_depends", None)

            document_embedder_output = self.document_embedder.run(
                input_data={"documents": documents}, run_depends=self._run_depends, config=config, **kwargs
            )
            self._run_depends = [NodeDependency(node=self.document_embedder).to_dict(for_tracing=True)]
            embedded_documents = document_embedder_output.output.get("documents", [])
            logger.debug(f"Tool {self.name} - {self.id}: embedded {len(embedded_documents)} documents")

            document_writer_output = self.document_writer.run(
                input_data={"documents": embedded_documents},
                run_depends=self._run_depends,
                config=config,
                **kwargs,
            )
            self._run_depends = [NodeDependency(node=self.document_writer).to_dict(for_tracing=True)]
            upserted_count = document_writer_output.output.get("upserted_count", 0)
            logger.debug(f"Tool {self.name} - {self.id}: wrote {upserted_count} documents to vector store")

            result = {"upserted_count": upserted_count}
            logger.info(f"Tool {self.name} - {self.id}: finished with RESULT:\n{str(result)[:200]}...")

            return result
        except Exception as e:
            logger.error(f"Tool {self.name} - {self.id}: execution error: {str(e)}", exc_info=True)
            raise ToolExecutionException(
                f"Tool '{self.name}' failed to write documents to the vector store. "
                f"Error: {str(e)}. Please analyze the error and take appropriate action.",
                recoverable=True,
            )
