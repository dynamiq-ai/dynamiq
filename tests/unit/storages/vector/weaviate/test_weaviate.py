import pytest
from dynamiq.storages.vector.weaviate.weaviate import WeaviateVectorStore


@pytest.mark.parametrize(
    "name",
    [
        "ValidName",
        "A",
        "ABC_123",
        "Z_MyCollection",
    ],
)
def test_valid_collection_names(name: str) -> None:
    assert WeaviateVectorStore.is_valid_collection_name(name) is True


@pytest.mark.parametrize(
    "name",
    [
        "",  # empty string
        "invalid",  # does not start with uppercase
        "123ABC",  # starts with a digit
        "_ABC",  # starts with underscore
        "A!B",  # invalid character
        "A-Collection",  # contains a hyphen which is not allowed
    ],
)
def test_invalid_collection_names(name: str) -> None:
    assert WeaviateVectorStore.is_valid_collection_name(name) is False


@pytest.mark.parametrize(
    "input_name, expected",
    [
        ("a", "A"),
        ("default", "Default"),
        ("ValidIndex", "ValidIndex"),
    ],
)
def test_fix_and_validate_index_name(input_name: str, expected: str) -> None:
    fixed = WeaviateVectorStore._fix_and_validate_index_name(input_name)
    assert fixed == expected
