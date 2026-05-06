import pytest

from dynamiq.components.splitters.token import TokenSplitterComponent
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


class _FakeTokenizer:
    def encode(self, text, allowed_special="all", disallowed_special="all"):
        return list(range(len(text)))

    def decode(self, token_ids):
        if token_ids == [0, 1, 2]:
            return "abc"
        if token_ids == [2, 3, 4]:
            return "not-a-source-substring"
        return "".join(chr(ord("a") + token_id) for token_id in token_ids)

    def decode_single_token_bytes(self, token_id):
        return chr(ord("a") + token_id).encode("utf-8")


def test_token_splitter_tracks_start_index_from_token_offsets_when_decoded_chunk_is_not_source_substring():
    splitter = TokenSplitterComponent(chunk_size=3, chunk_overlap=1)
    splitter._tokenizer = _FakeTokenizer()

    chunks = splitter._split_document(Document(content="abcdef"))

    assert [chunk.content for chunk in chunks] == ["abc", "not-a-source-substring", "ef"]
    assert [chunk.metadata["start_index"] for chunk in chunks] == [0, 2, 4]


class _ShortParentDecodedTokenizer:
    def encode(self, text, allowed_special="all", disallowed_special="all"):
        return list(range(len(text)))

    def decode(self, token_ids):
        if token_ids == [0, 1, 2, 3]:
            return "a"
        return "".join(chr(ord("a") + token_id) for token_id in token_ids)

    def decode_single_token_bytes(self, token_id):
        return chr(ord("a") + token_id).encode("utf-8")


class _ShortParentDecodedTokenSplitter(TokenSplitterComponent):
    def _ensure_tokenizer(self):
        return _ShortParentDecodedTokenizer()


def test_token_splitter_parent_matching_uses_token_span_end_offsets():
    splitter = _ShortParentDecodedTokenSplitter(
        chunk_size=2,
        chunk_overlap=0,
        parent_chunk_size=4,
        parent_chunk_overlap=0,
    )

    chunks = splitter._split_document(Document(id="source", content="abcdef"))
    parent_chunks = [chunk for chunk in chunks if chunk.metadata.get("is_parent")]
    child_chunks = [chunk for chunk in chunks if not chunk.metadata.get("is_parent")]

    assert len(parent_chunks) == 2
    assert [chunk.metadata["start_index"] for chunk in child_chunks] == [0, 2, 4]
    assert all("parent_chunk_id" in chunk.metadata for chunk in child_chunks)
    assert child_chunks[1].metadata["parent_chunk_id"] == parent_chunks[0].id


def test_token_splitter_node_accepts_special_token_sets():
    splitter = TokenSplitter(
        allowed_special={"<|endoftext|>"},
        disallowed_special={"<|fim_prefix|>"},
    )

    kwargs = splitter._component_kwargs()

    assert kwargs["allowed_special"] == {"<|endoftext|>"}
    assert kwargs["disallowed_special"] == {"<|fim_prefix|>"}
