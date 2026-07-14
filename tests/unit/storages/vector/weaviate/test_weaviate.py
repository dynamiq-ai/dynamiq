from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from weaviate.classes.config import DataType

from dynamiq.storages.vector.weaviate.weaviate import WeaviateVectorStore
from dynamiq.types import Document


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


@pytest.mark.parametrize(
    "name",
    [
        "validProperty",
        "valid_property",
        "ValidProperty",
        "_property",
        "a",
        "a_1_2_3",
        "_underscore_first",
    ],
)
def test_valid_property_names(name: str) -> None:
    """Test that valid property names are correctly identified."""
    assert WeaviateVectorStore.is_valid_property_name(name) is True


@pytest.mark.parametrize(
    "name",
    [
        "",  # empty string
        "1property",  # starts with a digit
        "property-name",  # contains a hyphen
        "property.name",  # contains a period
        "property name",  # contains a space
        "$property",  # starts with special character
        "property!",  # contains special character
        "property@name",  # contains special character
    ],
)
def test_invalid_property_names(name: str) -> None:
    """Test that invalid property names are correctly identified."""
    assert WeaviateVectorStore.is_valid_property_name(name) is False


def test_get_query_properties_includes_searchable_metadata_fields() -> None:
    store = WeaviateVectorStore.__new__(WeaviateVectorStore)
    store.content_key = "content"
    properties = [
        SimpleNamespace(name="content"),
        SimpleNamespace(name="filename"),
        SimpleNamespace(name="file_name"),
        SimpleNamespace(name="file_path"),
        SimpleNamespace(name="dynamiq_item_source_provider_title"),
        SimpleNamespace(name="file_content_hash"),
    ]

    assert store._get_query_properties(properties) == [
        "content",
        "filename",
        "file_name",
        "file_path",
        "dynamiq_item_source_provider_title",
    ]


def test_get_query_properties_excludes_uuid_and_non_searchable_properties() -> None:
    store = WeaviateVectorStore.__new__(WeaviateVectorStore)
    store.content_key = "content"
    properties = [
        SimpleNamespace(name="content", data_type=DataType.TEXT, index_searchable=True),
        SimpleNamespace(name="source", data_type=DataType.UUID, index_searchable=False),
        SimpleNamespace(name="title", data_type=DataType.TEXT, index_searchable=False),
        SimpleNamespace(name="url", data_type=DataType.TEXT, index_searchable=True),
    ]

    assert store._get_query_properties(properties) == ["content", "url"]


def test_to_document_does_not_mutate_weaviate_properties() -> None:
    store = WeaviateVectorStore.__new__(WeaviateVectorStore)
    store.content_key = "content"
    properties = {"_original_id": "doc-1", "content": "Body", "filename": "guide.md"}
    data = SimpleNamespace(properties=properties, vector=None, metadata=None)

    document = store._to_document(data)

    assert document.content == "Body"
    assert properties == {"_original_id": "doc-1", "content": "Body", "filename": "guide.md"}


def test_hybrid_retrieval_forwards_custom_content_key_to_document_conversion() -> None:
    store = WeaviateVectorStore.__new__(WeaviateVectorStore)
    store.content_key = "content"
    store.alpha = 0.5

    collection_properties = [
        SimpleNamespace(name="custom_content"),
        SimpleNamespace(name="file_name"),
        SimpleNamespace(name="source", data_type=DataType.UUID, index_searchable=False),
    ]
    result_object = SimpleNamespace(objects=[SimpleNamespace(properties={"_original_id": "1", "custom_content": "ok"})])
    store._collection = MagicMock()
    store._collection.config.get.return_value.properties = collection_properties
    store._collection.query.hybrid.return_value = result_object
    store._to_document = MagicMock(return_value=Document(content="ok"))

    docs = store._hybrid_retrieval(
        query_embedding=[0.1, 0.2],
        query="Ticket #5485",
        top_k=1,
        content_key="custom_content",
    )

    assert len(docs) == 1
    assert docs[0].content == "ok"
    store._to_document.assert_called_once_with(result_object.objects[0], content_key="custom_content")
    store._collection.query.hybrid.assert_called_once()
    assert store._collection.query.hybrid.call_args.kwargs["query_properties"] == ["custom_content", "file_name"]


@patch("dynamiq.storages.vector.weaviate.weaviate.Weaviate")
def test_to_data_object_with_valid_properties(mock_weaviate):
    """Test the _to_data_object method with valid property names."""
    # We'll bypass the initialization that connects to Weaviate
    store = WeaviateVectorStore.__new__(WeaviateVectorStore)

    # Set only the required attributes for _to_data_object to work
    store.content_key = "content"

    # Create a document with valid metadata property names
    doc = Document(
        content="Test content",
        document_id="test_id_1",
        metadata={
            "validProperty": "value1",
            "valid_property": "value2",
            "_property": "value3",
            "anotherValid_1_2_3": 123,
        },
        embedding=[0.1, 0.2, 0.3],
    )

    # This should pass without raising any exceptions
    data_obj = store._to_data_object(doc)

    # Verify the properties are correctly transferred
    assert data_obj["validProperty"] == "value1"
    assert data_obj["valid_property"] == "value2"
    assert data_obj["_property"] == "value3"
    assert data_obj["anotherValid_1_2_3"] == 123
    assert "embedding" not in data_obj
    assert "metadata" not in data_obj


@patch("dynamiq.storages.vector.weaviate.weaviate.Weaviate")
def test_to_data_object_with_invalid_properties(mock_weaviate):
    """Test the _to_data_object method with invalid property names."""
    # We'll bypass the initialization that connects to Weaviate
    store = WeaviateVectorStore.__new__(WeaviateVectorStore)

    # Set only the required attributes for _to_data_object to work
    store.content_key = "content"

    # Create a document with invalid metadata property names
    doc = Document(
        content="Test content",
        document_id="test_id_2",
        metadata={
            "valid_property": "value1",
            "1invalid_property": "value2",  # Starts with a number
            "invalid-property": "value3",  # Contains a hyphen
        },
        embedding=[0.1, 0.2, 0.3],
    )

    # This should raise a ValueError
    with pytest.raises(ValueError) as excinfo:
        store._to_data_object(doc)

    # Verify the error message mentions the invalid property name
    assert "Invalid property name" in str(excinfo.value)
    assert "1invalid_property" in str(excinfo.value)
