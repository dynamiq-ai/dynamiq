from typing import Any, ClassVar

from pydantic import Field

from dynamiq.components.splitters.code import CodeParser, CodeSplitterComponent
from dynamiq.components.splitters.language import Language
from dynamiq.nodes.splitters.base import BaseSplitterNode


class CodeSplitter(BaseSplitterNode):
    """Code-aware splitter node.

    Uses language-specific separator presets for fast regex-based splitting; an
    optional ``parser="tree_sitter"`` mode falls back to AST-aware boundaries when
    the ``tree-sitter-languages`` package is installed.
    """

    component_cls: ClassVar[type] = CodeSplitterComponent

    name: str = "CodeSplitter"
    description: str = "Splits source code using language-aware separators."

    language: Language = Field(default=Language.PYTHON, description="Source code language preset.")
    parser: CodeParser = Field(default=CodeParser.REGEX, description="Parser to use for splitting.")
    keep_separator: bool = Field(default=True, description="Keep code-aware separators at chunk boundaries.")

    def _component_kwargs(self) -> dict[str, Any]:
        kwargs = super()._component_kwargs()
        kwargs.update(language=self.language, parser=self.parser)
        return kwargs
