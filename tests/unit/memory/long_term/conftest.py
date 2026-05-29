"""Shared fixtures for long-term memory unit tests."""
import hashlib
from typing import ClassVar

import pytest

from dynamiq.connections import BaseConnection
from dynamiq.nodes.embedders.base import TextEmbedder, TextEmbedderInputSchema


class _StubConnection(BaseConnection):
    """No-op connection used only to satisfy ConnectionNode's connection/client validator."""

    def connect(self) -> None:
        return None


class FakeTextEmbedder(TextEmbedder):
    """Deterministic `TextEmbedder` subclass for tests.

    Maps text to a fixed-length unit vector derived from its sha256 digest.
    Same text → same vector. Different texts → near-orthogonal vectors
    (good enough for cosine ranking in unit tests). Bypasses any real
    `text_embedder` component.
    """

    name: str = "fake-text-embedder"
    connection: BaseConnection = _StubConnection()
    DIM: ClassVar[int] = 16

    def execute(self, input_data: TextEmbedderInputSchema, config=None, **kwargs) -> dict:
        text = input_data.query if hasattr(input_data, "query") else input_data["query"]
        return {"query": text, "embedding": self._embed(text)}

    def embed(self, text: str) -> list[float]:
        """Convenience helper for tests that want a raw vector without an InputSchema."""
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


@pytest.fixture
def backend(fake_embedder):
    """A fresh in-memory backend wired with the deterministic fake embedder."""
    from dynamiq.memory.long_term.backends.in_memory import InMemoryLongTermMemoryBackend

    return InMemoryLongTermMemoryBackend(embedder=fake_embedder)


@pytest.fixture
def user_id() -> str:
    return "user-test-123"


@pytest.fixture
def other_user_id() -> str:
    return "user-other-456"
