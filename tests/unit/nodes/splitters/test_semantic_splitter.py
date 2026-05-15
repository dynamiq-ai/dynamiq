import numpy as np

from dynamiq.components.splitters.semantic import (
    BreakpointThresholdType,
    SemanticSplitterComponent,
)
from dynamiq.nodes.splitters.semantic import SemanticSplitter


def test_semantic_splitter_splits_on_distance_breakpoints():
    sentences = [
        ("The cat sat on the mat. ", [1.0, 0.0]),
        ("It was a sunny day. ", [1.0, 0.05]),
        ("The economy is volatile. ", [0.0, 1.0]),
        ("Markets reacted to inflation.", [0.05, 1.0]),
    ]
    text = "".join(sentence for sentence, _ in sentences)
    embeddings = {sentence.strip(): vector for sentence, vector in sentences}

    def fake_embed(texts):
        results = []
        for grouped in texts:
            for key, vector in embeddings.items():
                if key in grouped:
                    results.append(vector)
                    break
            else:
                results.append([0.0, 0.0])
        return results

    splitter = SemanticSplitterComponent(
        embed_fn=fake_embed,
        breakpoint_threshold_type=BreakpointThresholdType.PERCENTILE,
        breakpoint_threshold_amount=50.0,
        buffer_size=0,
    )
    chunks = splitter.split_text(text)
    assert len(chunks) >= 2
    assert any("cat" in chunk for chunk in chunks)
    assert any("economy" in chunk or "inflation" in chunk for chunk in chunks)


def test_semantic_splitter_returns_single_split_for_short_text():
    splitter = SemanticSplitterComponent(embed_fn=lambda texts: [[1.0] for _ in texts])
    assert splitter.split_text("only one sentence") == ["only one sentence"]


def test_semantic_splitter_preserves_sentence_spacing_when_no_distances(monkeypatch):
    splitter = SemanticSplitterComponent(embed_fn=lambda texts: [[1.0] for _ in texts])
    monkeypatch.setattr(
        splitter, "_pairwise_cosine_distances", lambda matrix: np.empty(0, dtype=float)
    )

    assert splitter.split_text("First sentence. Second sentence.") == [
        "First sentence. Second sentence."
    ]


def test_semantic_splitter_gradient_threshold_compares_gradients():
    splitter = SemanticSplitterComponent(
        embed_fn=lambda texts: [[1.0, 0.0], [0.0, 1.0]],
        breakpoint_threshold_type=BreakpointThresholdType.GRADIENT,
        buffer_size=0,
    )

    assert splitter.split_text("First sentence. Second sentence.") == [
        "First sentence. Second sentence."
    ]


class _FakeEmbedderComponent:
    model = "fake-model"
    prefix = "prefix:"
    suffix = ":suffix"
    batch_size = 16

    def __init__(self):
        self.embed_text_calls = []
        self.batch_calls = []

    def embed_text(self, text):
        self.embed_text_calls.append(text)
        return {"embedding": [0.0]}

    def _apply_text_truncation(self, text):
        return text

    def _embed_texts_batch(self, texts_to_embed, batch_size):
        self.batch_calls.append((texts_to_embed, batch_size))
        return [[float(index)] for index, _ in enumerate(texts_to_embed)], {"model": self.model}


class _FakeTextEmbedderNode:
    def __init__(self, component):
        self.text_embedder = component


def test_semantic_splitter_node_embeds_sentence_groups_in_batch():
    component = _FakeEmbedderComponent()
    splitter = SemanticSplitter.model_construct(embedder=_FakeTextEmbedderNode(component))

    embeddings = splitter._embed_batch(["first\ntext", "second text"])

    assert embeddings == [[0.0], [1.0]]
    assert component.embed_text_calls == []
    assert component.batch_calls == [(["prefix:first text:suffix", "prefix:second text:suffix"], 16)]
