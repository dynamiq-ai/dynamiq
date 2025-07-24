from abc import ABC
from typing import ClassVar, Literal

from pydantic import BaseModel, Field

from dynamiq.nodes.node import NodeGroup, VectorStoreNode
from dynamiq.types import Document


class WriterInputSchema(BaseModel):
    documents: list[Document] = Field(..., description="Parameter to provide documents to write.")
    content_key: str = Field(default=None, description="Parameter to provide content key.")
    embedding_key: str = Field(default=None, description="Parameter to provide embedding key.")


class Writer(VectorStoreNode, ABC):

    group: Literal[NodeGroup.WRITERS] = NodeGroup.WRITERS
    input_schema: ClassVar[type[WriterInputSchema]] = WriterInputSchema
