from copy import deepcopy
from typing import Any

from dynamiq.components.splitters.base import LengthUnit, SplitterComponentBase
from dynamiq.types import Document


class TokenSplitterComponent(SplitterComponentBase):
    """Token-based splitter.

    Splits text on token boundaries using ``tiktoken`` (default) or a Hugging Face
    tokenizer when ``model_name`` is supplied via the ``hf:`` prefix.
    """

    def __init__(
        self,
        encoding_name: str = "cl100k_base",
        model_name: str | None = None,
        allowed_special: set[str] | str = "all",
        disallowed_special: set[str] | str = "all",
        **kwargs,
    ) -> None:
        kwargs.setdefault("length_unit", LengthUnit.TOKENS)
        kwargs.setdefault("chunk_size", 512)
        kwargs.setdefault("chunk_overlap", 50)
        super().__init__(**kwargs)
        self.encoding_name = encoding_name
        self.model_name = model_name
        self.allowed_special = allowed_special
        self.disallowed_special = disallowed_special
        self._tokenizer: Any | None = None

    def _ensure_tokenizer(self) -> Any:
        if self._tokenizer is not None:
            return self._tokenizer
        try:
            import tiktoken
        except ImportError as exc:
            raise ImportError(
                "TokenSplitter requires the 'tiktoken' package. Install with `pip install tiktoken`."
            ) from exc
        if self.model_name:
            self._tokenizer = tiktoken.encoding_for_model(self.model_name)
        else:
            self._tokenizer = tiktoken.get_encoding(self.encoding_name)
        return self._tokenizer

    def _length(self, text: str) -> int:
        tokenizer = self._ensure_tokenizer()
        return len(
            tokenizer.encode(
                text,
                allowed_special=self.allowed_special,
                disallowed_special=self.disallowed_special,
            )
        )

    def _constructor_kwargs(self) -> dict[str, Any]:
        kwargs = super()._constructor_kwargs()
        kwargs.update(
            encoding_name=self.encoding_name,
            model_name=self.model_name,
            allowed_special=self.allowed_special,
            disallowed_special=self.disallowed_special,
        )
        return kwargs

    def split_text(self, text: str) -> list[str]:
        return [chunk for chunk, _ in self._split_text_with_offsets(text)]

    def _split_document(self, doc: Document) -> list[Document]:
        base_metadata: dict[str, Any] = deepcopy(doc.metadata) if doc.metadata else {}
        base_metadata["source_id"] = doc.id

        results = self._build_documents_with_token_offsets(doc.content, base_metadata, doc_id=doc.id)
        if self.parent_chunk_size:
            results = self._attach_parent_chunks_with_token_offsets(doc, results)
        return results

    def _build_documents_with_token_offsets(
        self,
        source_text: str,
        base_metadata: dict[str, Any],
        doc_id: str | None,
    ) -> list[Document]:
        built: list[Document] = []
        for index, (chunk, start_index) in enumerate(self._split_text_with_offsets(source_text)):
            if not chunk:
                continue
            metadata = deepcopy(base_metadata)
            if self.add_chunk_index:
                metadata["chunk_index"] = index
            if self.add_start_index:
                metadata["start_index"] = start_index
            chunk_id = self._build_chunk_id(doc_id, index, chunk)
            if chunk_id is None:
                built.append(Document(content=chunk, metadata=metadata))
            else:
                built.append(Document(id=chunk_id, content=chunk, metadata=metadata))
        return built

    def _attach_parent_chunks_with_token_offsets(
        self,
        source: Document,
        child_chunks: list[Document],
    ) -> list[Document]:
        parent = self.__class__(**self._parent_splitter_kwargs())
        parent_chunks = parent._build_documents_with_token_offsets(
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

    def _split_text_with_offsets(self, text: str) -> list[tuple[str, int]]:
        tokenizer = self._ensure_tokenizer()
        token_ids = tokenizer.encode(
            text,
            allowed_special=self.allowed_special,
            disallowed_special=self.disallowed_special,
        )
        if not token_ids:
            return []
        token_byte_offsets = self._token_byte_offsets(text, token_ids)
        byte_to_char = self._byte_to_char_offsets(text)
        chunks: list[tuple[str, int]] = []
        step = max(self.chunk_size - self.chunk_overlap, 1)
        for start in range(0, len(token_ids), step):
            window = token_ids[start : start + self.chunk_size]
            if not window:
                break
            start_index = byte_to_char[token_byte_offsets[start]]
            chunks.append((tokenizer.decode(window), start_index))
            if start + self.chunk_size >= len(token_ids):
                break
        return chunks

    def _token_byte_offsets(self, text: str, token_ids: list[int]) -> list[int]:
        tokenizer = self._ensure_tokenizer()
        if not hasattr(tokenizer, "decode_single_token_bytes"):
            return self._fallback_token_byte_offsets(text, token_ids)
        offsets = [0]
        current = 0
        for token_id in token_ids:
            current += len(tokenizer.decode_single_token_bytes(token_id))
            offsets.append(current)
        if offsets[-1] > len(text.encode("utf-8")):
            return self._fallback_token_byte_offsets(text, token_ids)
        return offsets

    def _fallback_token_byte_offsets(self, text: str, token_ids: list[int]) -> list[int]:
        tokenizer = self._ensure_tokenizer()
        offsets = [0]
        cursor = 0
        for index in range(len(token_ids)):
            token_text = tokenizer.decode(token_ids[: index + 1])
            cursor = max(cursor, len(token_text.encode("utf-8")))
            offsets.append(min(cursor, len(text.encode("utf-8"))))
        return offsets

    @staticmethod
    def _byte_to_char_offsets(text: str) -> list[int]:
        source_bytes = text.encode("utf-8")
        byte_to_char = [0] * (len(source_bytes) + 1)
        cursor = 0
        for char_index, character in enumerate(text):
            encoded = character.encode("utf-8")
            for offset in range(len(encoded)):
                byte_to_char[cursor + offset] = char_index
            cursor += len(encoded)
            byte_to_char[cursor] = char_index + 1
        return byte_to_char
