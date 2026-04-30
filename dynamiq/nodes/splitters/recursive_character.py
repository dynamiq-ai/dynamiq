from typing import Any, ClassVar

from pydantic import Field

from dynamiq.components.splitters.language import Language, get_separators_for_language
from dynamiq.components.splitters.recursive_character import RecursiveCharacterSplitterComponent
from dynamiq.nodes.splitters.base import BaseSplitterNode


class RecursiveCharacterSplitter(BaseSplitterNode):
    """Recursive character splitter node.

    Walks a hierarchy of separators (largest -> smallest) and recursively splits any
    chunk that exceeds ``chunk_size``. Optional ``language`` preset substitutes a
    code-aware separator list (Python, JS, Markdown, HTML, ...).
    """

    component_cls: ClassVar[type] = RecursiveCharacterSplitterComponent

    name: str = "RecursiveCharacterSplitter"
    description: str = "Splits text by recursively trying a hierarchy of separators."

    separators: list[str] | None = Field(
        default=None,
        description="Ordered separator list (largest -> smallest). Defaults to ['\\n\\n', '\\n', ' ', ''].",
    )
    is_separator_regex: bool = Field(default=False, description="Treat separators as regular expressions.")
    language: Language | None = Field(
        default=None,
        description="Optional language preset; populates separators with the language's recursive hierarchy.",
    )

    def _component_kwargs(self) -> dict[str, Any]:
        kwargs = super()._component_kwargs()
        if self.language is not None:
            kwargs["separators"] = get_separators_for_language(self.language)
            kwargs["is_separator_regex"] = True
        else:
            kwargs["separators"] = self.separators
            kwargs["is_separator_regex"] = self.is_separator_regex
        return kwargs
