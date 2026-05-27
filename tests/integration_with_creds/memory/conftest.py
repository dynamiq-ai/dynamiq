import hashlib

import pytest


class FakeTextEmbedder:
    """Deterministic 16-dim embedder for integration tests against real backends."""

    DIM = 16

    def execute(self, input_data, **kwargs):
        text = input_data["query"] if isinstance(input_data, dict) else input_data.query
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
