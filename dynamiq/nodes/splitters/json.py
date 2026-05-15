from typing import Any, ClassVar, Literal

from pydantic import BaseModel, Field

from dynamiq.components.splitters.json import RecursiveJsonSplitterComponent
from dynamiq.connections.managers import ConnectionManager
from dynamiq.nodes.node import Node, NodeGroup, ensure_config
from dynamiq.runnables import RunnableConfig
from dynamiq.types import Document
from dynamiq.utils.logger import logger


class JsonSplitterInputSchema(BaseModel):
    documents: list[Document] = Field(..., description="JSON documents to split.")


class RecursiveJsonSplitter(Node):
    """Splits JSON-serializable text recursively, keeping each chunk under ``max_chunk_size``."""

    group: Literal[NodeGroup.SPLITTERS] = NodeGroup.SPLITTERS
    name: str = "RecursiveJsonSplitter"
    description: str = "Splits JSON documents into smaller JSON chunks."

    max_chunk_size: int = Field(default=2000, gt=0, description="Maximum serialized size per chunk.")
    min_chunk_size: int | None = Field(default=None, description="Minimum serialized size per chunk.")
    convert_lists: bool = Field(default=False, description="Convert lists to indexed dicts before splitting.")

    splitter: Any | None = None
    input_schema: ClassVar[type[JsonSplitterInputSchema]] = JsonSplitterInputSchema

    @property
    def to_dict_exclude_params(self) -> dict[str, Any]:
        return super().to_dict_exclude_params | {"splitter": True}

    def init_components(self, connection_manager: ConnectionManager | None = None) -> None:
        connection_manager = connection_manager or ConnectionManager()
        super().init_components(connection_manager)
        if self.splitter is None:
            self.splitter = RecursiveJsonSplitterComponent(
                max_chunk_size=self.max_chunk_size,
                min_chunk_size=self.min_chunk_size,
                convert_lists=self.convert_lists,
            )

    def execute(self, input_data: JsonSplitterInputSchema, config: RunnableConfig = None, **kwargs) -> dict[str, Any]:
        config = ensure_config(config)
        self.run_on_node_execute_run(config.callbacks, **kwargs)
        documents = input_data.documents
        logger.debug(f"RecursiveJsonSplitter: splitting {len(documents)} documents.")
        output = self.splitter.run(documents=documents)
        return {"documents": output["documents"]}
