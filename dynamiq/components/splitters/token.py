from typing import Any

from dynamiq.components.splitters.base import LengthUnit, SplitterComponentBase


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
            tokenizer.encode(text, allowed_special=self.allowed_special, disallowed_special=self.disallowed_special)
        )

    def split_text(self, text: str) -> list[str]:
        tokenizer = self._ensure_tokenizer()
        token_ids = tokenizer.encode(
            text,
            allowed_special=self.allowed_special,
            disallowed_special=self.disallowed_special,
        )
        if not token_ids:
            return []
        chunks: list[str] = []
        step = max(self.chunk_size - self.chunk_overlap, 1)
        for start in range(0, len(token_ids), step):
            window = token_ids[start : start + self.chunk_size]
            if not window:
                break
            chunks.append(tokenizer.decode(window))
            if start + self.chunk_size >= len(token_ids):
                break
        return chunks
