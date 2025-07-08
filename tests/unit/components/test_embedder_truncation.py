from dynamiq.utils import TruncationMethod, truncate_text_for_embedding


def test_truncation_logic_integration():
    """Test that the truncation logic is properly integrated into embedders."""

    long_text = "A" * 100
    result = truncate_text_for_embedding(long_text, max_tokens=10, truncation_method=TruncationMethod.MIDDLE)

    assert len(result) < len(long_text)
    assert "...[truncated for embedding]..." in result


def test_truncation_with_different_methods_integration():
    """Test different truncation methods work in integration."""
    long_text = "X" * 200

    result_start = truncate_text_for_embedding(long_text, max_tokens=20, truncation_method=TruncationMethod.START)
    assert result_start.startswith("...[truncated for embedding]...")
    assert result_start.endswith("X")

    result_end = truncate_text_for_embedding(long_text, max_tokens=20, truncation_method=TruncationMethod.END)
    assert result_end.startswith("X")
    assert result_end.endswith("...[truncated for embedding]...")

    result_middle = truncate_text_for_embedding(long_text, max_tokens=20, truncation_method=TruncationMethod.MIDDLE)
    assert result_middle.startswith("X")
    assert result_middle.endswith("X")
    assert "...[truncated for embedding]..." in result_middle


def test_embedder_has_truncation_configuration():
    """Test that BaseEmbedder has the expected truncation configuration."""
    from dynamiq.components.embedders.base import BaseEmbedder

    model_fields = BaseEmbedder.model_fields
    assert "truncation_enabled" in model_fields
    assert "max_input_tokens" in model_fields
    assert "truncation_method" in model_fields

    assert model_fields["truncation_enabled"].default is True
    assert model_fields["max_input_tokens"].default == 8192
    assert model_fields["truncation_method"].default == TruncationMethod.MIDDLE
