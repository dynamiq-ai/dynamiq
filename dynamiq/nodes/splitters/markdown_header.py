from typing import Any, ClassVar, Literal

from pydantic import BaseModel, Field

from dynamiq.components.splitters.markdown_header import MarkdownHeaderSplitterComponent
from dynamiq.connections.managers import ConnectionManager
from dynamiq.nodes.node import Node, NodeGroup, ensure_config
from dynamiq.runnables import RunnableConfig
from dynamiq.types import Document
from dynamiq.utils.logger import logger


class MarkdownHeaderSplitterInputSchema(BaseModel):
    documents: list[Document] = Field(..., description="Markdown documents to split.")


class MarkdownHeaderSplitter(Node):
    """Splits Markdown text on header markers, carrying the active header path in metadata.

    Each chunk's metadata is enriched with one entry per active header level
    (e.g. ``{"h1": "Intro", "h2": "Goals"}``).
    """

    group: Literal[NodeGroup.SPLITTERS] = NodeGroup.SPLITTERS
    name: str = "MarkdownHeaderSplitter"
    description: str = "Splits Markdown documents on header tags and propagates header path to metadata."

    headers_to_split_on: list[tuple[str, str]] = Field(
        default_factory=lambda: [("#", "h1"), ("##", "h2"), ("###", "h3"), ("####", "h4")],
        description="Pairs of (markdown-prefix, metadata-key).",
    )
    strip_headers: bool = Field(default=True, description="Drop the header line itself from chunks.")
    return_each_line: bool = Field(default=False, description="Emit one chunk per non-empty line.")

    splitter: Any | None = None
    input_schema: ClassVar[type[MarkdownHeaderSplitterInputSchema]] = MarkdownHeaderSplitterInputSchema

    @property
    def to_dict_exclude_params(self) -> dict[str, Any]:
        return super().to_dict_exclude_params | {"splitter": True}

    def init_components(self, connection_manager: ConnectionManager | None = None) -> None:
        connection_manager = connection_manager or ConnectionManager()
        super().init_components(connection_manager)
        if self.splitter is None:
            self.splitter = MarkdownHeaderSplitterComponent(
                headers_to_split_on=[tuple(pair) for pair in self.headers_to_split_on],
                strip_headers=self.strip_headers,
                return_each_line=self.return_each_line,
            )

    def execute(
        self, input_data: MarkdownHeaderSplitterInputSchema, config: RunnableConfig = None, **kwargs
    ) -> dict[str, Any]:
        config = ensure_config(config)
        self.run_on_node_execute_run(config.callbacks, **kwargs)
        documents = input_data.documents
        logger.debug(f"MarkdownHeaderSplitter: splitting {len(documents)} documents.")
        output = self.splitter.run(documents=documents)
        return {"documents": output["documents"]}
