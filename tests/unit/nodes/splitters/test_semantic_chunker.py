from dynamiq.components.splitters.semantic import BreakpointThresholdType, SemanticChunkerComponent


def test_semantic_chunker_splits_on_distance_breakpoints():
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

    chunker = SemanticChunkerComponent(
        embed_fn=fake_embed,
        breakpoint_threshold_type=BreakpointThresholdType.PERCENTILE,
        breakpoint_threshold_amount=50.0,
        buffer_size=0,
    )
    chunks = chunker.split_text(text)
    assert len(chunks) >= 2
    assert any("cat" in chunk for chunk in chunks)
    assert any("economy" in chunk or "inflation" in chunk for chunk in chunks)


def test_semantic_chunker_returns_single_chunk_for_short_text():
    chunker = SemanticChunkerComponent(embed_fn=lambda texts: [[1.0] for _ in texts])
    assert chunker.split_text("only one sentence") == ["only one sentence"]
