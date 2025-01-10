from abc import ABC
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, Field

from dynamiq.nodes.node import NodeGroup, VectorStoreNode


class RetrieverInputSchema(BaseModel):
    embedding: list[float] = Field(..., description="Parameter to provided embedding for search.")
    filters: dict[str, Any] = Field(
        default={}, description="Parameter to provided filters to apply for retrieving specific documents."
    )
    top_k: int = Field(default=0, description="Parameter to provided how many documents to retrieve.")
    content_key: str = Field(default=None, description="Parameter to provide content key.")
    embedding_key: str = Field(default=None, description="Parameter to provide embedding key.")
    query: str = Field(default=None, description="Parameter to provide query for search.")
    alpha: float = Field(default=None, description="Parameter to provide alpha for hybrid retrieval.")


class Retriever(VectorStoreNode, ABC):
    group: Literal[NodeGroup.RETRIEVERS] = NodeGroup.RETRIEVERS
    filters: dict[str, Any] | None = None
    top_k: int = 10
    input_schema: ClassVar[type[RetrieverInputSchema]] = RetrieverInputSchema

    @property
    def to_dict_exclude_params(self):
        return super().to_dict_exclude_params | {"document_retriever": True}
