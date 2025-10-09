from typing import Any, ClassVar, Literal

from pydantic import BaseModel, Field

from dynamiq.nodes.node import Node
from dynamiq.nodes.types import NodeGroup
from dynamiq.runnables.base import RunnableConfig


class KnowledgeBaseRetrieverInputSchema(BaseModel):
    query: str = Field(..., description="Parameter to provide a query to retrieve documents.")


class KnowledgeBaseRetriever(Node):
    """
    A Mock Node for representing a knowledge base
    """

    name: str = "KnowledgeBaseRetriever"
    group: Literal[NodeGroup.TOOLS] = NodeGroup.TOOLS
    description: str = "A node for retrieving relevant documents from a knowledge base."
    input_schema: ClassVar[type[KnowledgeBaseRetrieverInputSchema]] = KnowledgeBaseRetrieverInputSchema

    def execute(self, input_data: dict[str, Any] | BaseModel, config: RunnableConfig = None, **kwargs) -> Any:
        return
