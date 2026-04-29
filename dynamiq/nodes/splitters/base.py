from typing import Any, ClassVar, Literal

from pydantic import BaseModel, Field

from dynamiq.components.splitters.base import IdStrategy, LengthUnit
from dynamiq.connections.managers import ConnectionManager
from dynamiq.nodes.node import Node, NodeGroup, ensure_config
from dynamiq.runnables import RunnableConfig
from dynamiq.types import Document
from dynamiq.utils.logger import logger


class SplitterInputSchema(BaseModel):
    documents: list[Document] = Field(..., description="Documents to split.")


class BaseSplitterNode(Node):
    """Base class for splitter nodes.

    Subclasses set ``component_cls`` and implement :meth:`_build_component`. They share
    a common YAML-friendly Pydantic config surface (chunk_size, overlap, metadata
    flags, deterministic IDs, parent-chunk pairing) and a uniform ``execute`` flow.
    """

    component_cls: ClassVar[type] = type(None)

    group: Literal[NodeGroup.SPLITTERS] = NodeGroup.SPLITTERS
    name: str = "BaseSplitter"
    description: str = "Splits documents into smaller chunks."

    chunk_size: int = Field(default=4000, gt=0, description="Maximum size of each chunk.")
    chunk_overlap: int = Field(default=200, ge=0, description="Overlap between consecutive chunks.")
    length_unit: LengthUnit = Field(default=LengthUnit.CHARS, description="Unit used to measure chunk size.")
    keep_separator: bool = Field(default=False, description="Keep the separator string at chunk boundaries.")
    strip_whitespace: bool = Field(default=True, description="Strip whitespace from chunk edges.")
    add_start_index: bool = Field(default=True, description="Record each chunk's start index in source text.")
    add_chunk_index: bool = Field(default=True, description="Record each chunk's positional index in metadata.")
    merge_short_chunks: bool = Field(default=True, description="Merge tail chunks shorter than chunk_size.")
    parent_chunk_size: int | None = Field(default=None, description="Optional larger parent-chunk size.")
    parent_chunk_overlap: int | None = Field(default=None, description="Overlap for parent chunks.")
    id_strategy: IdStrategy = Field(default=IdStrategy.UUID, description="Chunk ID generation strategy.")

    splitter: Any | None = None
    input_schema: ClassVar[type[SplitterInputSchema]] = SplitterInputSchema

    @property
    def to_dict_exclude_params(self) -> dict[str, Any]:
        return super().to_dict_exclude_params | {"splitter": True}

    def init_components(self, connection_manager: ConnectionManager | None = None) -> None:
        connection_manager = connection_manager or ConnectionManager()
        super().init_components(connection_manager)
        if self.splitter is None:
            self.splitter = self._build_component()

    def _component_kwargs(self) -> dict[str, Any]:
        return {
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "length_unit": self.length_unit,
            "keep_separator": self.keep_separator,
            "strip_whitespace": self.strip_whitespace,
            "add_start_index": self.add_start_index,
            "add_chunk_index": self.add_chunk_index,
            "merge_short_chunks": self.merge_short_chunks,
            "parent_chunk_size": self.parent_chunk_size,
            "parent_chunk_overlap": self.parent_chunk_overlap,
            "id_strategy": self.id_strategy,
        }

    def _build_component(self) -> Any:
        return self.component_cls(**self._component_kwargs())

    def execute(self, input_data: SplitterInputSchema, config: RunnableConfig = None, **kwargs) -> dict[str, Any]:
        config = ensure_config(config)
        self.run_on_node_execute_run(config.callbacks, **kwargs)
        documents = input_data.documents
        logger.debug(f"{self.name}: splitting {len(documents)} documents.")
        output = self.splitter.run(documents=documents)
        return {"documents": output["documents"]}
