from typing import Any, ClassVar, Literal

from pydantic import BaseModel, Field

from dynamiq.components.splitters.document import DocumentSplitBy
from dynamiq.components.splitters.document import DocumentSplitter as DocumentSplitterComponent
from dynamiq.nodes.node import Node, NodeGroup, ensure_config
from dynamiq.runnables import RunnableConfig
from dynamiq.types import Document
from dynamiq.utils.logger import logger

PREPROCESS_TOOL_DESCRIPTION = """Preprocesses text by splitting it into smaller parts.

Key Capabilities:
- Splitting text into smaller parts based on configurable parameters
- Dynamic parameter selection that allows to choose the optimal splitting strategy

Parameter Guide:
- documents: List of documents to split.
- split_by: The unit by which the document should be split. \
    Possible values are "word", "sentence", "page", "passage", "title", "character". Defaults to "sentence".
- split_length: The maximum number of units to include in each split. Defaults to 10.
- split_overlap: The number of units that should overlap between consecutive splits. Defaults to 0.

Examples:
- {"documents": [{"content": "<content of the document>",\
    "metadata": {"<field name>": "<field value>"}}]}
- {"documents": [{"content": "<content of the first document>",\
    "metadata": {"<field name>": "<field value>"}},{"content": "<content of the second document>",\
        "metadata": {"<field name>": "<field value>"}}]}
"""


class PreprocessToolInputSchema(BaseModel):
    documents: list[Document] = Field(..., description="Parameter to provide documents to split.")
    split_by: DocumentSplitBy = Field(
        default=DocumentSplitBy.SENTENCE,
        description="Parameter to provide the unit by which the document should be split.",
    )
    split_length: int = Field(
        default=10, description="Parameter to provide the maximum number of units to include in each split."
    )
    split_overlap: int = Field(
        default=0,
        description="Parameter to provide the number of units that should overlap between consecutive splits.",
    )


class PreprocessTool(Node):
    """
    A tool for preprocessing text by splitting it into smaller parts.
    """

    group: Literal[NodeGroup.SPLITTERS] = NodeGroup.SPLITTERS
    name: str = "PreprocessTool"
    description: str = PREPROCESS_TOOL_DESCRIPTION
    split_by: DocumentSplitBy = DocumentSplitBy.SENTENCE
    split_length: int = 10
    split_overlap: int = 0
    input_schema: ClassVar[type[PreprocessToolInputSchema]] = PreprocessToolInputSchema

    def execute(self, input_data: PreprocessToolInputSchema, config: RunnableConfig = None, **kwargs) -> dict[str, Any]:
        """Splits the documents into chunks based on the provided parameters.

        Args:
            input_data (PreprocessToolInputSchema): The input data containing the documents to split.
            config (RunnableConfig, optional): The configuration for the execution. Defaults to None.
            **kwargs: Additional keyword arguments.

        Returns:
            dict[str, Any]: A dictionary containing the split documents under the key "documents".
        """
        config = ensure_config(config)
        self.run_on_node_execute_run(config.callbacks, **kwargs)

        documents = input_data.documents

        split_by = input_data.split_by or self.split_by
        split_length = input_data.split_length if input_data.split_length is not None else self.split_length
        split_overlap = input_data.split_overlap if input_data.split_overlap is not None else self.split_overlap

        logger.debug(
            f"Splitting {len(documents)} documents with parameters: split_by={split_by}, "
            f"split_length={split_length}, split_overlap={split_overlap}"
        )

        splitter = DocumentSplitterComponent(
            split_by=split_by,
            split_length=split_length,
            split_overlap=split_overlap,
        )

        output = splitter.run(documents=documents)

        split_documents = output["documents"]
        logger.debug(f"Split {len(documents)} documents into {len(split_documents)} parts")

        return {
            "content": split_documents,
        }
