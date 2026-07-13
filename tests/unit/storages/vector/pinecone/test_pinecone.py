from unittest.mock import MagicMock

import pytest

from dynamiq.storages.vector.pinecone.pinecone import PineconeVectorStore, validate_pinecone_index_name


@pytest.mark.parametrize(
    "name",
    [
        "valid-name",
        "abc123",
        "index-name",
        "name1",
        "a",
        "long-index-name-with-dashes",
        "n0d3-1nd3x",
        "a" * 30,
    ],
)
def test_valid_index_names(name: str) -> None:
    assert validate_pinecone_index_name(name) == name


@pytest.mark.parametrize(
    "name",
    [
        "",  # empty string
        "invalid.",  # contains a dot
        "123abc",  # starts with a digit
        "_abc",  # starts with underscore
        "-abc",  # starts with hyphen
        "a!b",  # invalid character
        "abc_def",  # underscore in the middle
        "abc.def",  # dot in name
        "Name",  # uppercase
        "名字",  # Chinese characters
        "a--b",  # multiple dashes
        "white space",  # space in the name
    ],
)
def test_invalid_index_names(name: str) -> None:
    with pytest.raises(ValueError, match=r"Index name '.*' is invalid"):
        validate_pinecone_index_name(name)


def _bare_pinecone_store() -> PineconeVectorStore:
    store = PineconeVectorStore.__new__(PineconeVectorStore)
    store.content_key = "content"
    store.namespace = "default"
    store._dummy_vector = [-10.0]
    store._index = MagicMock()
    return store


def test_get_documents_by_id_fetches_and_converts() -> None:
    store = _bare_pinecone_store()
    store._index.fetch.return_value = {
        "vectors": {
            "1": {"id": "1", "metadata": {"content": "ok", "k": "v"}, "values": [-10.0]},
        }
    }

    docs = store.get_documents_by_id(["1", "missing"])

    assert len(docs) == 1
    assert docs[0].id == "1"
    assert docs[0].content == "ok"
    assert docs[0].metadata == {"k": "v"}
    assert docs[0].embedding is None  # include_embeddings defaults to False
    store._index.fetch.assert_called_once_with(ids=["1", "missing"], namespace="default")


def test_get_documents_by_id_includes_embeddings_when_requested() -> None:
    store = _bare_pinecone_store()
    store._index.fetch.return_value = {
        "vectors": {"1": {"id": "1", "metadata": {"content": "ok"}, "values": [0.1, 0.2]}}
    }

    docs = store.get_documents_by_id(["1"], include_embeddings=True)

    assert docs[0].embedding == [0.1, 0.2]


def test_get_documents_by_id_empty_returns_empty_without_query() -> None:
    store = _bare_pinecone_store()
    assert store.get_documents_by_id([]) == []
    store._index.fetch.assert_not_called()
