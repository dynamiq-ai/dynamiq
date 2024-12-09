from typing import Any, ClassVar, Literal

from pydantic import BaseModel, Field

from dynamiq.components.splitters.document import DocumentSplitBy
from dynamiq.components.splitters.document import DocumentSplitter as DocumentSplitterComponent
from dynamiq.connections.managers import ConnectionManager
from dynamiq.nodes.node import Node, NodeGroup, ensure_config
from dynamiq.runnables import RunnableConfig
from dynamiq.types import Document
from dynamiq.utils.logger import logger


class DocumentSplitterInputSchema(BaseModel):
    documents: list[Document] = Field(..., description="Parameter to provide documents to split.")


class DocumentSplitter(Node):
    """Splits a list of text documents into a list of text documents with shorter texts.

    Splitting documents with long texts is a common preprocessing step during indexing.
    This allows Embedders to create significant semantic representations
    and avoids exceeding the maximum context length of language models.

    Args:
        split_by (Literal["word", "sentence", "page", "passage"], optional): Determines the unit by
            which the document should be split. Defaults to "word".
        split_length (int, optional): Maximum number of units (as defined by `split_by`) to include
            in each split. Defaults to 200.
        split_overlap (int, optional): Number of units that should overlap between consecutive
            splits. Defaults to 0.

    Attributes:
        group (Literal[NodeGroup.SPLITTERS]): The group of the node.
        name (str): The name of the node.
        split_by (DocumentSplitBy): The unit by which the document should be split.
        split_length (int): The maximum number of units to include in each split.
        split_overlap (int): The number of units that should overlap between consecutive splits.
        document_splitter (DocumentSplitterComponent): The component used for document splitting.
    """

    group: Literal[NodeGroup.SPLITTERS] = NodeGroup.SPLITTERS
    name: str = "DocumentSplitter"
    split_by: DocumentSplitBy = DocumentSplitBy.PASSAGE
    split_length: int = 10
    split_overlap: int = 0
    document_splitter: DocumentSplitterComponent = None
    input_schema: ClassVar[type[DocumentSplitterInputSchema]] = DocumentSplitterInputSchema

    @property
    def to_dict_exclude_params(self):
        return super().to_dict_exclude_params | {"document_splitter": True}

    def init_components(self, connection_manager: ConnectionManager | None = None) -> None:
        """Initializes the components of the DocumentSplitter.

        Args:
            connection_manager (ConnectionManager, optional): The connection manager to use.
                Defaults to ConnectionManager().
        """
        connection_manager = connection_manager or ConnectionManager()
        super().init_components(connection_manager)
        if self.document_splitter is None:
            self.document_splitter = DocumentSplitterComponent(
                split_by=self.split_by,
                split_length=self.split_length,
                split_overlap=self.split_overlap,
            )

    def execute(
        self, input_data: DocumentSplitterInputSchema, config: RunnableConfig = None, **kwargs
    ) -> dict[str, Any]:
        """Executes the document splitting process.

        Args:
            input_data (DocumentSplitterInputSchema): The input data containing the documents to split.
            config (RunnableConfig, optional): The configuration for the execution. Defaults to None.
            **kwargs: Additional keyword arguments.

        Returns:
            dict[str, Any]: A dictionary containing the split documents under the key "documents".
        """
        config = ensure_config(config)
        self.run_on_node_execute_run(config.callbacks, **kwargs)

        documents = input_data.documents
        logger.debug(f"Splitting {len(documents)} documents")
        output = self.document_splitter.run(documents=documents)

        split_documents = output["documents"]
        logger.debug(
            f"Split {len(documents)} documents into {len(split_documents)} parts"
        )

        return {
            "documents": split_documents,
        }
