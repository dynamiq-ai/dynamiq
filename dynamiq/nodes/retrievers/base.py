from abc import ABC
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, Field

from dynamiq.components.retrievers.base import DocumentRetriever
from dynamiq.nodes.node import NodeGroup, VectorStoreNode, ensure_config
from dynamiq.runnables import RunnableConfig
from dynamiq.types import Document


class RetrieverInputSchema(BaseModel):
    embedding: list[float] = Field(..., description="Parameter to provided embedding for search.")
    filters: dict[str, Any] = Field(
        default={}, description="Parameter to provided filters to apply for retrieving specific documents."
    )
    top_k: int = Field(default=0, description="Parameter to provided how many documents to retrieve.")


class Retriever(VectorStoreNode, ABC):
    group: Literal[NodeGroup.RETRIEVERS] = NodeGroup.RETRIEVERS
    filters: dict[str, Any] | None = None
    top_k: int = 10
    document_retriever: DocumentRetriever
    input_schema: ClassVar[type[RetrieverInputSchema]] = RetrieverInputSchema

    @property
    def to_dict_exclude_params(self):
        return super().to_dict_exclude_params | {"document_retriever": True}

    def execute(self, input_data: RetrieverInputSchema, config: RunnableConfig = None, **kwargs) -> dict[str, Document]:
        """
        Execute the document retrieval process.

        This method retrieves documents based on the input embedding.

        Args:
            input_data (RetrieverInputSchema): The input data containing the query embedding.
            config (RunnableConfig, optional): The configuration for the execution. Defaults to None.
            **kwargs: Additional keyword arguments.

        Returns:
            dict[str, Any]: A dictionary containing the retrieved documents.
        """
        config = ensure_config(config)
        self.run_on_node_execute_run(config.callbacks, **kwargs)

        query_embedding = input_data.embedding
        filters = input_data.filters or self.filters
        top_k = input_data.top_k or self.top_k

        output = self.document_retriever.run(query_embedding, filters=filters, top_k=top_k)

        return {
            "documents": output["documents"],
        }
