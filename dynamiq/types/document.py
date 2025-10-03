import enum
import uuid
from typing import Any, Callable

from pydantic import BaseModel, Field

from dynamiq.utils.utils import TRUNCATE_EMBEDDINGS_LIMIT


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

    def to_dict(self, for_tracing: bool = False, truncate_limit: int = TRUNCATE_EMBEDDINGS_LIMIT, **kwargs) -> dict:
        """Convert the Document object to a dictionary.

        Returns:
            dict: Dictionary representation of the Document.
        """
        data = self.model_dump(exclude={"embedding"}, **kwargs)

        if for_tracing and self.embedding is not None and len(self.embedding) > truncate_limit:
            data["embedding"] = self.embedding[:truncate_limit]
        else:
            data["embedding"] = self.embedding

        return data


class DocumentCreationMode(str, enum.Enum):
    """Enumeration for document creation modes."""
    ONE_DOC_PER_FILE = "one-doc-per-file"
    ONE_DOC_PER_PAGE = "one-doc-per-page"
    ONE_DOC_PER_ELEMENT = "one-doc-per-element"
