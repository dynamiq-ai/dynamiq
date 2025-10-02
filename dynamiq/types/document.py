import enum
import uuid
from typing import Any, Callable, TypedDict

from pydantic import BaseModel, Field

from dynamiq.utils.utils import TRUNCATE_LIMIT


class Document(BaseModel):
    """Document class for Dynamiq.

    Attributes:
        id (Callable[[], Any] | str | None): Unique identifier. Defaults to UUID4 hex.
        content (str): Main content of the document.
        metadata (dict | None): Additional metadata. Defaults to None.
        embedding (list | None): Vector representation. Defaults to None.
        score (float | None): Relevance or similarity score. Defaults to None.
    """
    id: Callable[[], Any] | str | None = Field(default_factory=lambda: uuid.uuid4().hex)
    content: str
    metadata: dict | None = None
    embedding: list | None = None
    score: float | None = None

    def to_dict(self, for_tracing: bool = False, truncate_limit: int = TRUNCATE_LIMIT, **kwargs) -> dict:
        """Convert the Document object to a dictionary.

        Returns:
            dict: Dictionary representation of the Document.
        """
        data = self.model_dump(**kwargs)

        if for_tracing and self.embedding is not None:
            original_length = len(data["embedding"])
            if original_length > truncate_limit:
                data["embedding"] = data["embedding"][:truncate_limit]

        return data


class DocumentCreationMode(str, enum.Enum):
    """Enumeration for document creation modes."""
    ONE_DOC_PER_FILE = "one-doc-per-file"
    ONE_DOC_PER_PAGE = "one-doc-per-page"
    ONE_DOC_PER_ELEMENT = "one-doc-per-element"


class TextEmbeddingOutputDict(TypedDict):
    embedding: list[float]
    query: str


class TextEmbeddingOutput(dict):
    """Dict-like output for text embedders with tracing-aware to_dict()."""

    query: str
    embedding: list[float]

    def to_dict(self, for_tracing: bool = False, truncate_limit: int = TRUNCATE_LIMIT, **kwargs) -> dict:
        data = dict(self)
        if for_tracing and isinstance(data.get("embedding"), list):
            if len(data["embedding"]) > truncate_limit:
                data["embedding"] = data["embedding"][:truncate_limit]
        return data
