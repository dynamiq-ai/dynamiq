import pytest

from dynamiq.nodes.splitters.html import HTMLHeaderSplitter, HTMLSectionSplitter
from dynamiq.types import Document

pytest.importorskip("bs4")


def test_html_header_splitter_carries_header_path():
    splitter = HTMLHeaderSplitter()
    splitter.init_components()
    text = "<html><body><h1>Title</h1><p>Intro</p><h2>Sub</h2><p>Body</p></body></html>"
    chunks = splitter.execute(splitter.input_schema(documents=[Document(content=text)]))["documents"]
    assert len(chunks) >= 1
    assert any(chunk.metadata.get("h1") == "Title" for chunk in chunks)
    assert any(chunk.metadata.get("h2") == "Sub" for chunk in chunks)


def test_html_header_splitter_clears_custom_metadata_keys_by_header_level():
    splitter = HTMLHeaderSplitter(headers_to_split_on=[("h1", "title"), ("h2", "subtitle")])
    splitter.init_components()
    text = "<html><body><h1>First</h1><h2>Sub</h2><p>A</p><h1>Second</h1><p>B</p></body></html>"

    chunks = splitter.execute(splitter.input_schema(documents=[Document(content=text)]))["documents"]

    second = next(chunk for chunk in chunks if chunk.content == "B")
    assert second.metadata["title"] == "Second"
    assert "subtitle" not in second.metadata


def test_html_section_splitter_applies_xpath_filter():
    pytest.importorskip("lxml")
    splitter = HTMLSectionSplitter(xpath_filter="//main")
    splitter.init_components()
    text = """
    <html>
      <body>
        <aside><h1>Aside</h1><p>Skip me</p></aside>
        <main><h1>Main</h1><p>Keep me</p></main>
      </body>
    </html>
    """

    chunks = splitter.execute(splitter.input_schema(documents=[Document(content=text)]))["documents"]

    assert [chunk.content for chunk in chunks] == ["Keep me"]
    assert chunks[0].metadata["h1"] == "Main"


def test_html_section_splitter_passes_return_each_element_to_component():
    splitter = HTMLSectionSplitter(return_each_element=True)
    splitter.init_components()

    assert splitter.splitter.return_each_element is True
