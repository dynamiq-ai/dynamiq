import pytest

from dynamiq.components.splitters.document import DocumentSplitter
from dynamiq.types import Document


def test_split_by_title_preserves_text():
    content = "intro\n# A\nbody\n# B\nx"
    splitter = DocumentSplitter(split_by="title", split_length=1)
    chunks = splitter.run([Document(id="d", content=content)])["documents"]

    assert [chunk.content for chunk in chunks] == ["intro", "\n# A\nbody", "\n# B\nx"]
    assert "".join(chunk.content for chunk in chunks) == content


def test_split_by_passage_preserves_text():
    content = "one\n\ntwo\n\nthree"
    splitter = DocumentSplitter(split_by="passage", split_length=1)
    chunks = splitter.run([Document(id="d", content=content)])["documents"]

    assert "".join(chunk.content for chunk in chunks) == content


@pytest.mark.parametrize(
    "content,split_length,split_overlap",
    [
        ("\n# A\nx", 1, 0),
        ("# T\nx\n## S\ny", 1, 0),
        ("no headings", 1, 0),
        ("\n# A\nbody\n# B\nx", 2, 0),
    ],
)
def test_split_by_title_lossless_edge_cases(content, split_length, split_overlap):
    splitter = DocumentSplitter(split_by="title", split_length=split_length, split_overlap=split_overlap)
    chunks = splitter.run([Document(id="d", content=content)])["documents"]

    assert "".join(chunk.content for chunk in chunks) == content
