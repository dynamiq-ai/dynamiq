import pytest

from dynamiq.nodes.splitters.token import TokenSplitter
from dynamiq.types import Document

tiktoken = pytest.importorskip("tiktoken")


def test_token_splitter_chunks_by_token_budget():
    splitter = TokenSplitter(chunk_size=8, chunk_overlap=2, encoding_name="cl100k_base")
    splitter.init_components()
    text = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron"
    output = splitter.execute(splitter.input_schema(documents=[Document(content=text)]))
    chunks = output["documents"]
    encoding = tiktoken.get_encoding("cl100k_base")
    assert len(chunks) > 1
    for chunk in chunks:
        assert len(encoding.encode(chunk.content)) <= 8


def test_token_splitter_records_metadata():
    splitter = TokenSplitter(chunk_size=4, chunk_overlap=0, encoding_name="cl100k_base")
    splitter.init_components()
    document = Document(content="one two three four five six seven eight", metadata={"foo": "bar"})
    chunks = splitter.execute(splitter.input_schema(documents=[document]))["documents"]
    assert all(chunk.metadata["foo"] == "bar" for chunk in chunks)
    assert chunks[0].metadata["chunk_index"] == 0
