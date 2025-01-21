from typing import ClassVar, Literal, Union

from pydantic import BaseModel, Field

from dynamiq.connections import Coda
from dynamiq.nodes import NodeGroup
from dynamiq.nodes.node import ConnectionNode


class CanvasPageContent(BaseModel):
    """
    Content for a page (canvas).
    """

    format: Literal["html", "markdown"] = Field(
        ..., description="Supported content types for page (canvas) content (html or markdown)."
    )
    content: str = Field(..., description="The actual textual (or HTML/Markdown) content of the page.")


class CanvasContent(BaseModel):
    """
    Represents a 'canvas' page content type.
    """

    type: Literal["canvas"] = Field(..., description="Indicates a page containing canvas content.")
    canvas_content: CanvasPageContent = Field(
        ..., alias="canvasContent", description="Canvas details (format + content)."
    )

    class Config:
        allow_population_by_field_name = True


class EmbedContent(BaseModel):
    """
    Represents an 'embed' page content type (embedding external content/URL).
    """

    type: Literal["embed"] = Field(..., description="Indicates a page that embeds other content.")
    url: str = Field(..., description="The URL of the content to embed.")
    render_method: Literal["compatibility", "standard"] | None = Field(
        default=None, alias="renderMethod", description="Render mode for an embed page."
    )

    class Config:
        allow_population_by_field_name = True


class SyncPageContent(BaseModel):
    """
    Represents a 'syncPage' type, which embeds another Coda doc/page.
    It can be in 'page' mode or 'document' mode.
    """

    type: Literal["syncPage"] = Field(..., description="Indicates a page that embeds other Coda content.")
    mode: Literal["page", "document"] = Field(..., description="Whether to sync just one page or the full doc.")
    source_doc_id: str = Field(..., alias="sourceDocId", description="The ID of the source doc to be embedded.")

    # For mode="page", we need sourcePageId and includeSubpages
    source_page_id: str | None = Field(
        default=None, alias="sourcePageId", description="The page ID to insert as a sync page (when mode='page')."
    )
    include_subpages: bool | None = Field(
        default=None, alias="includeSubpages", description="Whether to include subpages in the sync (when mode='page')."
    )

    class Config:
        allow_population_by_field_name = True


PageContentUnion = Union[CanvasContent, EmbedContent, SyncPageContent]


class CellEdit(BaseModel):
    """
    Defines an edit for a single cell in a row.
    """

    column: str = Field(..., description="Column ID, URL, or name.")
    value: str | float | bool | list = Field(
        ..., description="Value to set in the cell (can be string, number, bool, or list)."
    )


class RowEdit(BaseModel):
    """
    Defines a set of cell edits (one row).
    """

    cells: list[CellEdit] = Field(..., description="List of cell edits for a single row.")


class BaseCodaNode(ConnectionNode):
    """
    Base class for all Coda nodes with common utilities and methods
    """

    group: ClassVar[NodeGroup] = NodeGroup.TOOLS
    connection: Coda
    is_optimized_for_agents: bool = True

    def to_camel_case(self, snake_str: str) -> str:
        """
        Converts a snake_case string to camelCase.
        """
        components = snake_str.split("_")
        return components[0] + "".join(x.title() for x in components[1:])

    def dict_to_camel_case(self, data: dict) -> dict:
        """
        Converts dictionary keys from snake_case to camelCase.
        """
        return {self.to_camel_case(k): v for k, v in data.items()}

    def format_agent_response(self, data: dict, template: str) -> dict:
        """
        Formats response for agents using a template
        """
        if self.is_optimized_for_agents:
            return {"content": template.format(**data)}
        return data
