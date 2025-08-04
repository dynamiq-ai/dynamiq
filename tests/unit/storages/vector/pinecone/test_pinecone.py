import pytest

from dynamiq.storages.vector.pinecone.pinecone import validate_pinecone_index_name


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
