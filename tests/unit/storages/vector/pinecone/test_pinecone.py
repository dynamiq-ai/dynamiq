import pytest

from dynamiq.storages.vector.pinecone import PineconeVectorStore


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
    assert PineconeVectorStore.is_valid_index_name(name) is True


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
    assert PineconeVectorStore.is_valid_index_name(name) is False


@pytest.mark.parametrize(
    "input_name, expected",
    [
        ("valid-name", "valid-name"),
        ("abc123", "abc123"),
        ("index-name", "index-name"),
        ("name1", "name1"),
        ("DEFAULT", "default"),  # upper to lower
        ("index.name", "index-name"),  # dot -> dash
        ("index_name", "index-name"),  # underscore -> dash
        ("index-name...", "index-name"),  # dots -> dash
        (" Index Name ", "index-name"),  # spaces -> dash
        ("MiXeD--Case..Name", "mixed-case-name"),  # normalize and clean
        ("abc.def_ghi", "abc-def-ghi"),  # multiple invalid chars -> cleaned
        ("__init__", "init"),  # strip surrounding underscores
        ("---abc---", "abc"),  # trim excess dashes
    ],
)
def test_fix_and_validate_index_name(input_name: str, expected: str) -> None:
    fixed = PineconeVectorStore._fix_and_validate_index_name(input_name)
    assert fixed == expected


@pytest.mark.parametrize(
    "input_name",
    [
        "123abc",  # starts with a digit
        "名字",  # non-Latin characters
        "!!!",  # cleaned to empty
        "",  # already empty
        "---",  # cleaned to empty
        "💥💥💥",  # emoji-only input
        " ",  # whitespace-only input
    ],
)
def test_fix_and_validate_index_name_invalid(input_name: str) -> None:
    with pytest.raises(ValueError, match=r"Index name '.*' is invalid"):
        PineconeVectorStore._fix_and_validate_index_name(input_name)
