from dynamiq.nodes.splitters.markdown_header import MarkdownHeaderSplitter
from dynamiq.types import Document


def test_markdown_header_splitter_carries_header_path():
    splitter = MarkdownHeaderSplitter()
    splitter.init_components()
    text = "# Title\nIntro line.\n## Subtitle\nBody A.\n## Other\nBody B."
    chunks = splitter.execute(splitter.input_schema(documents=[Document(content=text)]))["documents"]
    assert len(chunks) == 3
    assert chunks[0].metadata["h1"] == "Title"
    assert chunks[1].metadata["h2"] == "Subtitle"
    assert chunks[2].metadata["h2"] == "Other"


def test_markdown_header_splitter_keeps_headers_when_requested():
    splitter = MarkdownHeaderSplitter(strip_headers=False)
    splitter.init_components()
    chunks = splitter.execute(splitter.input_schema(documents=[Document(content="# Title\nbody")]))["documents"]
    assert any("# Title" in chunk.content for chunk in chunks)


def test_markdown_header_splitter_ignores_headers_inside_code_fence():
    splitter = MarkdownHeaderSplitter()
    splitter.init_components()
    text = "# Real\nbefore\n```\n# Fake\n```\nafter"
    chunks = splitter.execute(splitter.input_schema(documents=[Document(content=text)]))["documents"]
    assert len(chunks) == 1
    assert chunks[0].metadata["h1"] == "Real"


def test_markdown_header_splitter_does_not_close_fences_with_info_strings():
    splitter = MarkdownHeaderSplitter()
    splitter.init_components()
    text = "# Real\nbefore\n````markdown\n```python\n# Fake A\n```\n````python\n# Fake B\n````\nafter"

    chunks = splitter.execute(splitter.input_schema(documents=[Document(content=text)]))["documents"]

    assert len(chunks) == 1
    assert chunks[0].metadata["h1"] == "Real"
    assert "# Fake B" in chunks[0].content
