from dynamiq.components.splitters.contextual import ContextualChunkerComponent
from dynamiq.types import Document


class _FakeInnerSplitter:
    def run(self, documents):
        chunks = []
        for doc in documents:
            for piece in doc.content.split("|"):
                chunks.append(Document(content=piece, metadata={"source_id": doc.id}))
        return {"documents": chunks}


def test_contextual_chunker_prepends_llm_context():
    calls = []

    def fake_llm(prompt: str) -> str:
        calls.append(prompt)
        return "Doc-level context."

    chunker = ContextualChunkerComponent(
        inner_splitter=_FakeInnerSplitter(),
        llm_fn=fake_llm,
        prepend=True,
    )
    output = chunker.run([Document(content="part-a|part-b")])
    assert len(output["documents"]) == 2
    for chunk in output["documents"]:
        assert chunk.content.startswith("Doc-level context.")
        assert chunk.metadata["context"] == "Doc-level context."
    assert len(calls) == 2


def test_contextual_chunker_caches_repeated_chunks():
    call_count = {"n": 0}

    def fake_llm(prompt: str) -> str:
        call_count["n"] += 1
        return "ctx"

    chunker = ContextualChunkerComponent(
        inner_splitter=_FakeInnerSplitter(),
        llm_fn=fake_llm,
        cache_context=True,
    )
    chunker.run([Document(content="same|same")])
    assert call_count["n"] == 1
