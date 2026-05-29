from dynamiq.nodes.retrievers.retriever import VectorStoreRetriever
from dynamiq.types import Document


def _retriever(**kwargs) -> VectorStoreRetriever:
    defaults = {
        "skip_empty_metadata": True,
        "is_optimized_for_agents": False,
    }
    return VectorStoreRetriever.model_construct(**(defaults | kwargs))


def test_format_content_skips_empty_metadata_and_rounds_score():
    retriever = _retriever()
    document = Document(
        content="Relevant chunk text.",
        score=0.756173968315,
        metadata={
            "doi": None,
            "file_path": "kb-uae-adgm-payroll-and-taxation.pdf",
            "title": "Payroll and Taxation",
            "city": "",
            "property": "Payroll",
        },
    )

    content = retriever.format_content([document])

    assert "Score: 0.756" in content
    assert "Doi:" not in content
    assert "City:" not in content
    assert content.index("File Path:") < content.index("Property:")


def test_format_content_metadata_fields_are_an_allowlist():
    retriever = _retriever()
    document = Document(
        content="Relevant chunk text.",
        score=0.9,
        metadata={
            "file_path": "source.pdf",
            "title": "Allowed title",
            "property": "Hidden property",
        },
    )

    content = retriever.format_content([document], metadata_fields=["title"])

    assert "Title: Allowed title" in content
    assert "File Path:" not in content
    assert "Property:" not in content
    assert "Score:" not in content


def test_format_content_empty_metadata_fields_does_not_fallback_to_all_metadata():
    retriever = _retriever()
    document = Document(
        content="Relevant chunk text.",
        score=0.9,
        metadata={
            "title": "Hidden title",
            "property": "Hidden property",
        },
    )

    content = retriever.format_content([document], metadata_fields=["", "missing"])

    assert "No metadata available." in content
    assert "Score:" not in content
    assert "Hidden title" not in content
    assert "Hidden property" not in content


def test_format_content_uses_retrieved_source_delimiters():
    retriever = _retriever(is_optimized_for_agents=True)
    document = Document(
        content="Agent-visible chunk.",
        score=0.91234,
        metadata={
            "file_path": "source.pdf",
            "title": None,
            "property": "Hidden property",
        },
    )

    content = retriever.format_content(
        [document],
        metadata_fields=["score", "file_path"],
    )

    assert content.startswith("--- Retrieved Source 1 ---\n")
    assert "Metadata:\nScore: 0.912" in content
    assert "\n\nContent:\nAgent-visible chunk." in content
    assert content.endswith("--- End Source 1 ---")
    assert "Score: 0.912" in content
    assert "File Path: source.pdf" in content
    assert "Hidden property" not in content
    assert "==" not in content


def test_agent_metadata_defaults_exclude_file_paths_and_ignore_missing_fields():
    retriever = _retriever(is_optimized_for_agents=True)
    document = Document(
        content="Agent-visible chunk.",
        score=0.91,
        metadata={
            "title": "Source Title",
            "file_path": "internal/path/source.pdf",
            "url": "https://example.com/source",
        },
    )

    content = retriever.format_content(
        [document],
        metadata_fields=retriever._resolve_formatted_metadata_fields(),
    )

    assert "Title: Source Title" in content
    assert "Url: https://example.com/source" in content
    assert "File Path:" not in content
    assert "Source Url:" not in content


def test_format_content_strips_outer_content_whitespace_only():
    retriever = _retriever(is_optimized_for_agents=True)
    document = Document(
        content="\n\nFirst paragraph.\n\nSecond paragraph.\n\n",
        score=0.5,
        metadata={"title": "Whitespace Example"},
    )

    content = retriever.format_content(
        [document],
        metadata_fields=retriever._resolve_formatted_metadata_fields(),
    )

    assert "Content:\nFirst paragraph.\n\nSecond paragraph.\n--- End Source 1 ---" in content
