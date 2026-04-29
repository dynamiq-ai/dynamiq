from typing import Any, ClassVar

from pydantic import Field

from dynamiq.components.splitters.base import LengthUnit
from dynamiq.components.splitters.token import TokenSplitterComponent
from dynamiq.nodes.splitters.base import BaseSplitterNode


class TokenSplitter(BaseSplitterNode):
    """Token-based splitter node.

    Splits on tokenizer boundaries using ``tiktoken`` encodings (``cl100k_base``,
    ``o200k_base``, ...) or a model name. Token-aware sizes prevent accidental
    overflow of LLM context windows.
    """

    component_cls: ClassVar[type] = TokenSplitterComponent

    name: str = "TokenSplitter"
    description: str = "Splits text into chunks by token count."

    chunk_size: int = Field(default=512, gt=0, description="Maximum tokens per chunk.")
    chunk_overlap: int = Field(default=50, ge=0, description="Tokens that overlap between consecutive chunks.")
    length_unit: LengthUnit = Field(default=LengthUnit.TOKENS, description="Always tokens for TokenSplitter.")
    encoding_name: str = Field(default="cl100k_base", description="Tiktoken encoding name.")
    model_name: str | None = Field(default=None, description="Optional model name used to look up the encoding.")
    allowed_special: str = Field(default="all", description="Forwarded to tiktoken.encode.")
    disallowed_special: str = Field(default="all", description="Forwarded to tiktoken.encode.")

    def _component_kwargs(self) -> dict[str, Any]:
        kwargs = super()._component_kwargs()
        kwargs.update(
            encoding_name=self.encoding_name,
            model_name=self.model_name,
            allowed_special=self.allowed_special,
            disallowed_special=self.disallowed_special,
        )
        return kwargs
