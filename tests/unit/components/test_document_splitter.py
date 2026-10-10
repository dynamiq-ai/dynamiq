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
