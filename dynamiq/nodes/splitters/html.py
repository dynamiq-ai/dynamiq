from typing import Any, ClassVar, Literal

from pydantic import BaseModel, Field

from dynamiq.components.splitters.html import HTMLHeaderSplitterComponent, HTMLSectionSplitterComponent
from dynamiq.connections.managers import ConnectionManager
from dynamiq.nodes.node import Node, NodeGroup, ensure_config
from dynamiq.runnables import RunnableConfig
from dynamiq.types import Document
from dynamiq.utils.logger import logger


class HTMLSplitterInputSchema(BaseModel):
    documents: list[Document] = Field(..., description="HTML documents to split.")


class HTMLHeaderSplitter(Node):
    """Splits HTML on header tags (``h1``..``h6``) and carries the header path in metadata.

    Requires the optional ``beautifulsoup4`` package (``lxml`` recommended).
    """

    group: Literal[NodeGroup.SPLITTERS] = NodeGroup.SPLITTERS
    name: str = "HTMLHeaderSplitter"
    description: str = "Splits HTML on header tags and propagates header path to metadata."

    headers_to_split_on: list[tuple[str, str]] = Field(
        default_factory=lambda: [("h1", "h1"), ("h2", "h2"), ("h3", "h3"), ("h4", "h4")],
        description="Pairs of (html-tag, metadata-key).",
    )
    return_each_element: bool = Field(default=False, description="Emit one chunk per element.")

    splitter: Any | None = None
    input_schema: ClassVar[type[HTMLSplitterInputSchema]] = HTMLSplitterInputSchema

    @property
    def to_dict_exclude_params(self) -> dict[str, Any]:
        return super().to_dict_exclude_params | {"splitter": True}

    def init_components(self, connection_manager: ConnectionManager | None = None) -> None:
        connection_manager = connection_manager or ConnectionManager()
        super().init_components(connection_manager)
        if self.splitter is None:
            self.splitter = HTMLHeaderSplitterComponent(
                headers_to_split_on=[tuple(pair) for pair in self.headers_to_split_on],
                return_each_element=self.return_each_element,
            )

    def execute(self, input_data: HTMLSplitterInputSchema, config: RunnableConfig = None, **kwargs) -> dict[str, Any]:
        config = ensure_config(config)
        self.run_on_node_execute_run(config.callbacks, **kwargs)
        documents = input_data.documents
        logger.debug(f"HTMLHeaderSplitter: splitting {len(documents)} documents.")
        output = self.splitter.run(documents=documents)
        return {"documents": output["documents"]}


class HTMLSectionSplitter(HTMLHeaderSplitter):
    """Splits HTML into sections delimited by configured header tags."""

    name: str = "HTMLSectionSplitter"
    description: str = "Splits HTML into sections; each section is one chunk."

    xpath_filter: str | None = Field(default=None, description="Optional XPath used to scope splitting.")

    def init_components(self, connection_manager: ConnectionManager | None = None) -> None:
        connection_manager = connection_manager or ConnectionManager()
        Node.init_components(self, connection_manager)
        if self.splitter is None:
            self.splitter = HTMLSectionSplitterComponent(
                headers_to_split_on=[tuple(pair) for pair in self.headers_to_split_on],
                xpath_filter=self.xpath_filter,
            )
