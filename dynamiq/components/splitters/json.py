import json
from copy import deepcopy
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from dynamiq.types import Document
from dynamiq.utils.logger import logger


class RecursiveJsonSplitterComponent(BaseModel):
    """Recursively splits JSON-like structures so each chunk's serialized size <= ``max_chunk_size``."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    max_chunk_size: int = Field(default=2000, gt=0)
    min_chunk_size: int | None = None
    convert_lists: bool = False

    @model_validator(mode="after")
    def set_default_min_chunk_size(self) -> "RecursiveJsonSplitterComponent":
        if self.min_chunk_size is None:
            self.min_chunk_size = max(self.max_chunk_size - 200, 50)
        return self

    def run(self, documents: list[Document]) -> dict[str, list[Document]]:
        if not isinstance(documents, list):
            raise TypeError("RecursiveJsonSplitter expects a list of Documents as input.")
        results: list[Document] = []
        for doc in documents:
            if doc.content is None:
                raise ValueError(f"RecursiveJsonSplitter requires text content; document ID {doc.id} has none.")
            try:
                parsed = json.loads(doc.content)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Document {doc.id} is not valid JSON: {exc}") from exc
            for index, chunk in enumerate(self.split_text(parsed)):
                metadata = deepcopy(doc.metadata) if doc.metadata else {}
                metadata["source_id"] = doc.id
                metadata["chunk_index"] = index
                results.append(Document(content=chunk, metadata=metadata))
        logger.debug(f"RecursiveJsonSplitter: split {len(documents)} documents into {len(results)} chunks.")
        return {"documents": results}

    def split_json(self, data: Any) -> list[dict[str, Any] | list[Any]]:
        if self.convert_lists:
            data = self._list_to_dict_preprocessing(data)
        chunks: list[Any] = []
        self._split(data, chunks)
        return chunks

    def split_text(self, data: Any) -> list[str]:
        return [json.dumps(chunk, ensure_ascii=False) for chunk in self.split_json(data)]

    def _split(self, data: Any, chunks: list[Any], current_path: list[str] | None = None) -> None:
        current_path = current_path or []
        size = self._json_size(data)
        if size <= self.max_chunk_size:
            chunks.append(data)
            return
        if isinstance(data, dict):
            partial: dict[str, Any] = {}
            for key, value in data.items():
                projected = self._json_size({**partial, key: value})
                if projected > self.max_chunk_size and partial:
                    chunks.append(partial)
                    partial = {}
                if self._json_size({key: value}) > self.max_chunk_size:
                    if partial:
                        chunks.append(partial)
                        partial = {}
                    self._split(value, chunks, current_path + [key])
                else:
                    partial[key] = value
            if partial:
                chunks.append(partial)
        elif isinstance(data, list):
            partial: list[Any] = []
            for item in data:
                projected = self._json_size(partial + [item])
                if projected > self.max_chunk_size and partial:
                    chunks.append(partial)
                    partial = []
                if self._json_size([item]) > self.max_chunk_size:
                    if partial:
                        chunks.append(partial)
                        partial = []
                    self._split(item, chunks, current_path)
                else:
                    partial.append(item)
            if partial:
                chunks.append(partial)
        else:
            chunks.append(data)

    @staticmethod
    def _json_size(data: Any) -> int:
        return len(json.dumps(data, ensure_ascii=False))

    def _list_to_dict_preprocessing(self, data: Any) -> Any:
        if isinstance(data, list):
            return {str(index): self._list_to_dict_preprocessing(item) for index, item in enumerate(data)}
        if isinstance(data, dict):
            return {key: self._list_to_dict_preprocessing(value) for key, value in data.items()}
        return data
