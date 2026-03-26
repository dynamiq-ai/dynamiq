import copy
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from dynamiq.nodes import Node, NodeGroup
from dynamiq.nodes.node import ensure_config
from dynamiq.runnables import RunnableConfig
from dynamiq.types import Document


class TextToDynamiqDocumentInputSchema(BaseModel):
    texts: list[str] = Field(..., description="Parameter to provide texts to transform")
    metadata: list[dict[str, Any]] | dict[str, Any] | None = Field(
        None, description="Parameter to provide metadata for the texts"
    )
    model_config = ConfigDict(arbitrary_types_allowed=True)

    @model_validator(mode="after")
    def normalize_metadata(self) -> "TextToDynamiqDocumentInputSchema":
        """Normalize metadata input into a list of dicts matching texts length."""
        texts_count = len(self.texts)

        if self.metadata is None:
            self.metadata = [{} for _ in range(texts_count)]
        elif isinstance(self.metadata, dict):
            self.metadata = [copy.deepcopy(self.metadata) for _ in range(texts_count)]
        elif isinstance(self.metadata, list):
            if len(self.metadata) != texts_count:
                raise ValueError(
                    f"Metadata list length {len(self.metadata)} does not match sources count {texts_count}"
                )
        else:
            raise ValueError("metadata must be None, a dict, or a list of dicts.")

        return self


class TextToDynamiqDocument(Node):
    group: Literal[NodeGroup.TRANSFORMERS] = NodeGroup.TRANSFORMERS
    name: str = "Text to Dynamiq Document"
    description: str = "Node that transforms text with optional metadata to Dynamiq Document type."

    model_config = ConfigDict(arbitrary_types_allowed=True)

    input_schema: ClassVar[type[TextToDynamiqDocumentInputSchema]] = TextToDynamiqDocumentInputSchema

    def execute(
        self,
        input_data: TextToDynamiqDocumentInputSchema,
        config: RunnableConfig = None,
        **kwargs,
    ) -> dict[str, Any]:
        """
        Transform texts with optional metadata to list of Dynamiq Document type.

        Args:
            input_data (TextToDynamiqDocumentInputSchema): input data for the tool, which includes
                texts to transform to Dynamiq documents.
            config (RunnableConfig, optional): Configuration for the runnable, including callbacks.
            **kwargs: Additional arguments passed to the execution context.

        Returns:
            dict[str, Any]: A dictionary containing transformed text.
        """
        config = ensure_config(config)
        self.run_on_node_execute_run(config.callbacks, **kwargs)

        texts = input_data.texts
        metadata = input_data.metadata

        documents = []

        if texts is not None:
            for text, metadata in zip(texts, metadata):
                doc = Document(content=text, metadata=metadata if metadata else {})
                documents.append(doc)
        return {"documents": documents}
