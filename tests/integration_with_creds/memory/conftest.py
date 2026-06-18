import hashlib
from typing import ClassVar

import pytest

from dynamiq.connections import BaseConnection
from dynamiq.nodes.embedders.base import TextEmbedder, TextEmbedderInputSchema


class _StubConnection(BaseConnection):
    """No-op connection to satisfy ConnectionNode's connection/client validator."""

    def connect(self) -> None:
        return None


class FakeTextEmbedder(TextEmbedder):
    """Deterministic 16-dim embedder for integration tests against real backends."""

    name: str = "fake-text-embedder"
    connection: BaseConnection = _StubConnection()
    DIM: ClassVar[int] = 16

    def execute(self, input_data: TextEmbedderInputSchema, config=None, **kwargs) -> dict:
        text = input_data.query if hasattr(input_data, "query") else input_data["query"]
        return {"query": text, "embedding": self._embed(text)}

    def embed(self, text: str) -> list[float]:
        return self._embed(text)

    @classmethod
    def _embed(cls, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        raw = [(b / 127.5) - 1.0 for b in digest[: cls.DIM]]
        norm = sum(x * x for x in raw) ** 0.5 or 1.0
        return [x / norm for x in raw]


@pytest.fixture
def fake_embedder() -> FakeTextEmbedder:
    return FakeTextEmbedder()
