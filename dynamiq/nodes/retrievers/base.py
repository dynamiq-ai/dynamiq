from abc import ABC
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, Field

from dynamiq.nodes.node import NodeGroup, VectorStoreNode
from dynamiq.nodes.types import ActionType
from dynamiq.types import Document


class RetrieverInputSchema(BaseModel):
    embedding: list[float] = Field(..., description="Parameter to provided embedding for search.")
    filters: dict[str, Any] = Field(
        default_factory=dict,
        description="Parameter to provided filters to apply for retrieving specific documents.",
    )
    top_k: int = Field(default=0, description="Parameter to provided how many documents to retrieve.")
    similarity_threshold: float | None = Field(
        default=None,
        description="Parameter to provide minimal similarity "
        "or maximal distance score accepted for retrieved documents.",
    )
    content_key: str | None = Field(default=None, description="Parameter to provide content key.")
    embedding_key: str | None = Field(default=None, description="Parameter to provide embedding key.")
    query: str | None = Field(default=None, description="Parameter to provide query for search.")
    alpha: float | None = Field(
        default=None,
        ge=0,
        le=1,
        description="Parameter to provide alpha for hybrid retrieval.",
    )


class Retriever(VectorStoreNode, ABC):
    group: Literal[NodeGroup.RETRIEVERS] = NodeGroup.RETRIEVERS
    action_type: ActionType = ActionType.SEMANTIC_SEARCH
    filters: dict[str, Any] | None = None
    top_k: int = 10
    similarity_threshold: float | None = None
    input_schema: ClassVar[type[RetrieverInputSchema]] = RetrieverInputSchema

    @property
    def to_dict_exclude_params(self):
        return super().to_dict_exclude_params | {"document_retriever": True}

    def get_documents_by_id(self, ids: list[str]) -> list[Document]:
        """Fetch documents by their exact ids, forwarding to the underlying vector store.

        Shared by every retriever node; backends whose store can't fetch by id raise NotImplementedError.
        """
        if not ids:
            return []
        if self.vector_store is None:
            self.init_components()
        return self.vector_store.get_documents_by_id(list(ids))
