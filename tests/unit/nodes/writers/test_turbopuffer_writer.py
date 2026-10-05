from unittest.mock import MagicMock, patch

import pytest

from dynamiq.connections import Turbopuffer
from dynamiq.nodes.writers import TurbopufferDocumentWriter
from dynamiq.nodes.writers.base import WriterInputSchema
from dynamiq.runnables import RunnableConfig
from dynamiq.storages.vector.turbopuffer import TurbopufferVectorStore
from dynamiq.types import Document


@pytest.fixture
def vector_store():
    store = MagicMock(spec=TurbopufferVectorStore)
    store.client = MagicMock()
    store.write_documents.return_value = 2
    return store


@patch("dynamiq.connections.Turbopuffer.connect", return_value=MagicMock())
def test_initialization_with_defaults(mock_connect):
    with patch.object(TurbopufferDocumentWriter, "init_components"):
        writer = TurbopufferDocumentWriter()

    assert isinstance(writer.connection, Turbopuffer)


def test_vector_store_params():
    connection = Turbopuffer(api_key="key", region="aws-us-east-1")
    with patch.object(TurbopufferDocumentWriter, "init_components"):
        writer = TurbopufferDocumentWriter(connection=connection, index_name="vs-1", content_key="text")

    params = writer.vector_store_params

    assert writer.vector_store_cls is TurbopufferVectorStore
    assert params["index_name"] == "vs-1"
    assert params["content_key"] == "text"
    assert params["create_if_not_exist"] is False
    assert params["connection"] is connection
    assert "client" in params


def test_execute(vector_store):
    writer = TurbopufferDocumentWriter(vector_store=vector_store)
    documents = [Document(id="1", content="one"), Document(id="2", content="two")]

    result = writer.execute(WriterInputSchema(documents=documents, content_key="text"), RunnableConfig(callbacks=[]))

    vector_store.write_documents.assert_called_once_with(documents, content_key="text")
    assert result == {"upserted_count": 2}
