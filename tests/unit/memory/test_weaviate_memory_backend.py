from unittest.mock import MagicMock

from dynamiq.connections import OpenAI, Weaviate
from dynamiq.memory.backends import Weaviate as WeaviateMemory
from dynamiq.nodes.embedders import OpenAIDocumentEmbedder


def test_messages_are_stored_in_the_configured_collection():
    connection = MagicMock(spec=Weaviate)
    client = connection.connect.return_value
    client.collections.exists.return_value = True

    WeaviateMemory(
        connection=connection,
        embedder=OpenAIDocumentEmbedder(connection=OpenAI(api_key="test"), model="text-embedding-3-small"),
        collection_name="Chats",
    )

    client.collections.get.assert_called_once_with("Chats")
