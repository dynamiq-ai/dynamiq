from typing import Any, ClassVar

from pydantic import Field

from dynamiq.components.splitters.auto import AutoSplitterComponent, AutoSplitterRule, AutoSplitterStrategy
from dynamiq.components.splitters.code import CodeParser
from dynamiq.components.splitters.language import Language
from dynamiq.nodes.splitters.base import BaseSplitterNode


class AutoSplitter(BaseSplitterNode):
    """Routes each Document to the most suitable splitter based on metadata and content."""

    component_cls: ClassVar[type] = AutoSplitterComponent

    name: str = "AutoSplitter"
    description: str = "Routes documents to structure-aware splitters and falls back to recursive splitting."

    rules: list[AutoSplitterRule] = Field(
        default_factory=lambda: AutoSplitterComponent().rules,
        description="Serializable rules used to select splitter strategies.",
    )
    fallback_strategy: AutoSplitterStrategy = Field(
        default=AutoSplitterStrategy.RECURSIVE_CHARACTER,
        description="Strategy used when no rule or inference matches.",
    )
    fallback_on_error: bool = Field(default=True, description="Fallback to fallback_strategy on inferred route errors.")
    infer_from_content: bool = Field(default=True, description="Infer strategy from lightweight content sniffing.")
    add_splitter_metadata: bool = Field(default=True, description="Stamp selected splitter strategy into metadata.")
    splitter_metadata_key: str = Field(default="splitter_strategy", description="Metadata key for selected strategy.")

    json_max_chunk_size: int = Field(default=2000, gt=0, description="Maximum serialized JSON chunk size.")
    json_min_chunk_size: int | None = Field(default=None, description="Minimum serialized JSON chunk size.")
    json_convert_lists: bool = Field(default=False, description="Convert JSON lists to indexed dicts before splitting.")

    markdown_headers_to_split_on: list[tuple[str, str]] | None = Field(
        default=None,
        description="Pairs of (markdown-prefix, metadata-key) for MarkdownHeaderSplitter.",
    )
    markdown_strip_headers: bool = Field(default=True, description="Drop Markdown header lines from chunks.")
    markdown_return_each_line: bool = Field(default=False, description="Emit one Markdown chunk per non-empty line.")

    html_headers_to_split_on: list[tuple[str, str]] | None = Field(
        default=None,
        description="Pairs of (html-tag, metadata-key) for HTML splitters.",
    )
    html_return_each_element: bool = Field(default=False, description="Emit one HTML chunk per element.")
    html_xpath_filter: str | None = Field(
        default=None, description="Optional XPath used to scope HTML section splitting."
    )

    code_parser: CodeParser = Field(default=CodeParser.REGEX, description="Parser to use for code splitting.")
    code_default_language: Language = Field(
        default=Language.PYTHON, description="Fallback language for code splitting."
    )

    def _component_kwargs(self) -> dict[str, Any]:
        kwargs = super()._component_kwargs()
        kwargs.update(
            rules=self.rules,
            fallback_strategy=self.fallback_strategy,
            fallback_on_error=self.fallback_on_error,
            infer_from_content=self.infer_from_content,
            add_splitter_metadata=self.add_splitter_metadata,
            splitter_metadata_key=self.splitter_metadata_key,
            json_max_chunk_size=self.json_max_chunk_size,
            json_min_chunk_size=self.json_min_chunk_size,
            json_convert_lists=self.json_convert_lists,
            markdown_headers_to_split_on=self.markdown_headers_to_split_on,
            markdown_strip_headers=self.markdown_strip_headers,
            markdown_return_each_line=self.markdown_return_each_line,
            html_headers_to_split_on=self.html_headers_to_split_on,
            html_return_each_element=self.html_return_each_element,
            html_xpath_filter=self.html_xpath_filter,
            code_parser=self.code_parser,
            code_default_language=self.code_default_language,
        )
        return kwargs
