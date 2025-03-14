from dynamiq.connections import Chroma as ChromaConnection
from dynamiq.connections import Pinecone as PineconeConnection
from dynamiq.connections import Weaviate as WeaviateConnection
from dynamiq.storages.vector import ChromaVectorStore, PineconeVectorStore, WeaviateVectorStore


def delete_pinecone_documents_by_file_id(api_key: str, index_name: str, file_id: str):
    """
    Delete documents from Pinecone index by file_id.

    Args:
        api_key (str): Pinecone API key.
        index_name (str): Name of the Pinecone index.
        file_id (str): The file ID to filter by.
    """
    connection = PineconeConnection(api_key=api_key)
    vector_store = PineconeVectorStore(index_name=index_name, connection=connection)
    vector_store.delete_documents_by_file_id(file_id=file_id)


def delete_weaviate_documents_by_file_id(api_key: str, url: str, index_name: str, file_id: str):
    """
    Delete documents from Weaviate by file_id.

    Args:
        api_key (str): Weaviate API key.
        url (str): Weaviate URL.
        index_name (str): Name of the Weaviate collection.
        file_id (str): The file ID to filter by.
    """
    connection = WeaviateConnection(api_key=api_key, url=url)
    vector_store = WeaviateVectorStore(index_name=index_name, connection=connection)
    vector_store.delete_documents_by_file_id(file_id=file_id)


def delete_chroma_documents_by_file_id(host: str, port: str, index_name: str, file_id: str):
    """
    Delete documents from Chroma by file_id.

    Args:
        host (str): Chroma host.
        port (str): Chroma port.
        index_name (str): Name of the Chroma index.
        file_id (str): The file ID to filter by.
    """
    connection = ChromaConnection(host=host, port=port)
    vector_store = ChromaVectorStore(index_name=index_name, connection=connection)
    vector_store.delete_documents_by_file_id(file_id=file_id)
