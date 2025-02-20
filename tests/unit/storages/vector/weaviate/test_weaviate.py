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


def test_fix_and_validate_index_name_already_valid() -> None:
    # Name already valid; will be returned unchanged.
    input_name = "ValidIndex"
    fixed = WeaviateVectorStore._fix_and_validate_index_name(input_name)
    assert fixed == input_name


def test_fix_and_validate_index_name_lowercase() -> None:
    # Name starts with lowercase; should automatically be fixed.
    input_name = "default"
    fixed = WeaviateVectorStore._fix_and_validate_index_name(input_name)
    assert fixed == "Default"
