import numpy as np

from dynamiq.components.splitters.semantic import (
    BreakpointThresholdType,
    SemanticSplitterComponent,
)


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
