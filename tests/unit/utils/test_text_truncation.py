from dynamiq.utils import TruncationMethod, truncate_text_for_embedding


def test_no_truncation_needed():
    """Test that short text is not truncated."""
    text = "This is a short message"
    result = truncate_text_for_embedding(text, max_tokens=100)
    assert result == text


def test_empty_text():
    """Test that empty text is handled correctly."""
    assert truncate_text_for_embedding("") == ""
    assert truncate_text_for_embedding(None) is None


def test_middle_truncation():
    """Test middle truncation method."""
    text = "A" * 1000 + "MIDDLE" + "B" * 1000
    result = truncate_text_for_embedding(text, max_tokens=100, truncation_method=TruncationMethod.MIDDLE)

    assert len(result) < len(text)
    assert "...[truncated for embedding]..." in result
    assert result.startswith("A")
    assert result.endswith("B")


def test_start_truncation():
    """Test start truncation method."""
    text = "START" + "A" * 1000
    result = truncate_text_for_embedding(text, max_tokens=100, truncation_method=TruncationMethod.START)

    assert len(result) < len(text)
    assert "...[truncated for embedding]..." in result
    assert result.startswith("...[truncated for embedding]...")
    assert result.endswith("A")


def test_end_truncation():
    """Test end truncation method."""
    text = "A" * 1000 + "END"
    result = truncate_text_for_embedding(text, max_tokens=100, truncation_method=TruncationMethod.END)

    assert len(result) < len(text)
    assert "...[truncated for embedding]..." in result
    assert result.startswith("A")
    assert result.endswith("...[truncated for embedding]...")


def test_custom_truncation_message():
    """Test custom truncation message."""
    text = "A" * 1000
    custom_message = "...CUSTOM TRUNCATION..."
    result = truncate_text_for_embedding(text, max_tokens=100, truncation_message=custom_message)

    assert custom_message in result


def test_token_estimation():
    """Test that token estimation works roughly correctly."""
    text = "A" * 500
    result = truncate_text_for_embedding(text, max_tokens=100)

    assert len(result) < len(text)
    assert len(result) <= 450


def test_very_small_limit():
    """Test handling of very small token limits."""
    text = "This is a longer message that should be truncated"
    result = truncate_text_for_embedding(text, max_tokens=5)

    assert len(result) <= 25
    assert len(result) < len(text)
    assert result == text[:20] or "...[truncated]..." in result


def test_backward_compatibility_with_strings():
    """Test that string values still work for backward compatibility."""
    text = "A" * 200

    result_enum = truncate_text_for_embedding(text, max_tokens=20, truncation_method=TruncationMethod.START)
    result_string = truncate_text_for_embedding(text, max_tokens=20, truncation_method="START")

    assert result_enum == result_string
