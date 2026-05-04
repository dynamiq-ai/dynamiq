import enum
import hashlib
from copy import deepcopy
from typing import Any, Callable, ClassVar

from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic.json_schema import SkipJsonSchema

from dynamiq.types import Document
from dynamiq.utils.logger import logger


class LengthUnit(str, enum.Enum):
    """Unit used to measure chunk size."""

    CHARS = "chars"
    TOKENS = "tokens"


class IdStrategy(str, enum.Enum):
    """Strategy for assigning chunk IDs."""

    UUID = "uuid"
    DETERMINISTIC = "deterministic"


class SplitterComponentBase(BaseModel):
    """Base text-splitter component.

    Implements the``_merge_splits`` algorithm with overlap handling, separator
    accounting, optional whitespace stripping, optional ``start_index``/``chunk_index``
    metadata and ParentDocumentRetriever-style parent chunking.

    Subclasses must implement :meth:`split_text`.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    PARENT_DOC_KEY: ClassVar[str] = "parent_chunk_id"

    chunk_size: int = Field(default=4000, gt=0)
    chunk_overlap: int = Field(default=200, ge=0)
    length_unit: LengthUnit = LengthUnit.CHARS
    length_function: SkipJsonSchema[Callable[[str], int] | None] = None
    keep_separator: bool | str = False
    strip_whitespace: bool = True
    add_start_index: bool = True
    add_chunk_index: bool = True
    merge_short_chunks: bool = True
    parent_chunk_size: int | None = None
    parent_chunk_overlap: int | None = None
    id_strategy: IdStrategy = IdStrategy.UUID

    @model_validator(mode="after")
    def validate_splitter_config(self) -> "SplitterComponentBase":
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size.")
        if self.parent_chunk_size is not None and self.parent_chunk_size < self.chunk_size:
            raise ValueError("parent_chunk_size must be >= chunk_size when set.")
        if self.parent_chunk_overlap is None:
            self.parent_chunk_overlap = self.chunk_overlap
        if self.length_function is None:
            self.length_function = len
        return self

    def split_text(self, text: str) -> list[str]:
        """Split a single string into chunk strings. Subclasses must implement."""
        raise NotImplementedError

    def run(self, documents: list[Document]) -> dict[str, list[Document]]:
        """Split a list of :class:`Document` objects, propagating metadata."""
        if not isinstance(documents, list):
            raise TypeError(f"{type(self).__name__} expects a list of Documents as input.")
        if documents and not isinstance(documents[0], Document):
            raise TypeError(f"{type(self).__name__} expects a list of Documents as input.")

        split_documents: list[Document] = []
        for doc in documents:
            if doc.content is None:
                raise ValueError(
                    f"{type(self).__name__} only works with text documents but document.content for "
                    f"document ID {doc.id} is None."
                )
            split_documents.extend(self._split_document(doc))

        logger.debug(f"{type(self).__name__}: split {len(documents)} documents into {len(split_documents)} chunks.")
        return {"documents": split_documents}

    def _split_document(self, doc: Document) -> list[Document]:
        chunks = self.split_text(doc.content)
        base_metadata: dict[str, Any] = deepcopy(doc.metadata) if doc.metadata else {}
        base_metadata["source_id"] = doc.id

        results = self._build_documents(chunks, doc.content, base_metadata, doc_id=doc.id)
        if self.parent_chunk_size:
            results = self._attach_parent_chunks(doc, results)
        return results

    def _build_documents(
        self,
        chunks: list[str],
        source_text: str,
        base_metadata: dict[str, Any],
        doc_id: str | None,
    ) -> list[Document]:
        cursor = 0
        built: list[Document] = []
        for index, chunk in enumerate(chunks):
            if not chunk:
                continue
            metadata = deepcopy(base_metadata)
            if self.add_chunk_index:
                metadata["chunk_index"] = index
            if self.add_start_index:
                start_index = source_text.find(chunk, cursor)
                if start_index < 0:
                    start_index = source_text.find(chunk)
                if start_index < 0:
                    start_index = cursor
                metadata["start_index"] = start_index
                cursor = max(cursor, start_index + 1)
            chunk_id = self._build_chunk_id(doc_id, index, chunk)
            if chunk_id is None:
                built.append(Document(content=chunk, metadata=metadata))
            else:
                built.append(Document(id=chunk_id, content=chunk, metadata=metadata))
        return built

    def _build_chunk_id(self, parent_id: str | None, index: int, content: str) -> str | None:
        if self.id_strategy == IdStrategy.DETERMINISTIC:
            payload = f"{parent_id}:{index}:{content}".encode()
            digest = hashlib.sha256(payload).hexdigest()
            return digest[:32]
        return None  # let Document default factory generate UUID

    def _constructor_kwargs(self) -> dict[str, Any]:
        return {
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "length_unit": self.length_unit,
            "length_function": self.length_function,
            "keep_separator": self.keep_separator,
            "strip_whitespace": self.strip_whitespace,
            "add_start_index": self.add_start_index,
            "add_chunk_index": self.add_chunk_index,
            "merge_short_chunks": self.merge_short_chunks,
            "parent_chunk_size": self.parent_chunk_size,
            "parent_chunk_overlap": self.parent_chunk_overlap,
            "id_strategy": self.id_strategy,
        }

    def _parent_splitter_kwargs(self) -> dict[str, Any]:
        kwargs = self._constructor_kwargs()
        kwargs.update(
            chunk_size=self.parent_chunk_size,
            chunk_overlap=self.parent_chunk_overlap,
            add_start_index=True,
            add_chunk_index=False,
            parent_chunk_size=None,
            parent_chunk_overlap=None,
        )
        return kwargs

    def _attach_parent_chunks(self, source: Document, child_chunks: list[Document]) -> list[Document]:
        """Build parent chunks (larger context) and stamp each child with its parent's ID."""
        parent = self.__class__(**self._parent_splitter_kwargs())
        parent_chunks = parent._build_documents(
            parent.split_text(source.content),
            source.content,
            base_metadata={"source_id": source.id, "is_parent": True},
            doc_id=source.id,
        )
        for child in child_chunks:
            if not child.metadata or "start_index" not in child.metadata:
                continue
            start_index = child.metadata["start_index"]
            owner = next(
                (
                    p
                    for p in parent_chunks
                    if "start_index" in (p.metadata or {})
                    and start_index >= p.metadata["start_index"]
                    and start_index < p.metadata["start_index"] + len(p.content)
                ),
                None,
            )
            if owner is not None:
                child.metadata[self.PARENT_DOC_KEY] = owner.id
        return parent_chunks + child_chunks

    def _join_docs(self, docs: list[str], separator: str) -> str | None:
        text = separator.join(docs)
        if self.strip_whitespace:
            text = text.strip()
        return text or None

    def _length(self, text: str) -> int:
        return (self.length_function or len)(text)

    def _merge_splits(self, splits: list[str], separator: str) -> list[str]:
        """Merge splits respecting chunk_size/chunk_overlap. Ported from LangChain."""
        if not self.merge_short_chunks:
            chunks = []
            for split in splits:
                chunk = split.strip() if self.strip_whitespace else split
                if chunk:
                    chunks.append(chunk)
            return chunks

        separator_len = self._length(separator)
        chunks: list[str] = []
        current: list[str] = []
        total = 0
        for split in splits:
            split_len = self._length(split)
            projected = total + split_len + (separator_len if current else 0)
            if projected > self.chunk_size:
                if total > self.chunk_size:
                    logger.warning(
                        f"Created a chunk of size {total}, which is longer than the chunk_size {self.chunk_size}."
                    )
                if current:
                    merged = self._join_docs(current, separator)
                    if merged is not None:
                        chunks.append(merged)
                    while total > self.chunk_overlap or (
                        total + split_len + (separator_len if current else 0) > self.chunk_size and total > 0
                    ):
                        total -= self._length(current[0]) + (separator_len if len(current) > 1 else 0)
                        current.pop(0)
            current.append(split)
            total += split_len + (separator_len if len(current) > 1 else 0)
        merged = self._join_docs(current, separator)
        if merged is not None:
            chunks.append(merged)
        return chunks
