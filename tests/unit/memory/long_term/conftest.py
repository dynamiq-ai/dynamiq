"""Shared fixtures for long-term memory unit tests."""
import hashlib
from typing import Any

import pytest


class FakeTextEmbedder:
    """Deterministic text embedder for tests.

    Maps text to a fixed-length unit vector derived from its sha256 digest.
    Same text → same vector. Different texts → near-orthogonal vectors
    (good enough for cosine ranking in unit tests).
    """

    DIM = 16

    def execute(self, input_data: Any, **kwargs) -> dict:
        # Mirror TextEmbedder.execute output shape: {"query": ..., "embedding": ...}
        if hasattr(input_data, "query"):
            text = input_data.query
        elif isinstance(input_data, dict):
            text = input_data["query"]
        else:
            text = str(input_data)
        return {"query": text, "embedding": self._embed(text)}

    def embed(self, text: str) -> list[float]:
        """Convenience helper for tests that want a raw vector."""
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
def user_id() -> str:
    return "user-test-123"


@pytest.fixture
def other_user_id() -> str:
    return "user-other-456"
