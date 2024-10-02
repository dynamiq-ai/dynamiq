import requests

from dynamiq.connections import Chroma as ChromaConnection
from dynamiq.connections import Pinecone as PineconeConnection
from dynamiq.connections import Weaviate as WeaviateConnection
from dynamiq.storages.vector import (
    ChromaVectorStore,
    PineconeVectorStore,
    WeaviateVectorStore,
)


def initialize_connections():
    pinecone_connection = PineconeConnection()
    weaviate_connection = WeaviateConnection()
    chroma_connection = ChromaConnection()
    return pinecone_connection, weaviate_connection, chroma_connection


def initialize_vector_stores(
    pinecone_connection, weaviate_connection, chroma_connection
):
    pinecone_vector_store = PineconeVectorStore(
        index_name="default", connection=pinecone_connection
    )
    chroma_vector_store = ChromaVectorStore(
        index_name="default", connection=chroma_connection
    )
    weaviate_vector_store = WeaviateVectorStore(
        index_name="default", connection=weaviate_connection
    )
    return pinecone_vector_store, chroma_vector_store, weaviate_vector_store


def count_documents(vector_stores):
    for name, store in vector_stores.items():
        print(f"{name} - count documents: ", store.count_documents())


def list_documents(vector_stores):
    documents = {}
    for name, store in vector_stores.items():
        documents[name] = store.list_documents()
    return documents


def gather_file_ids_per_storage(documents):
    file_ids_per_storage = {}
    for storage_name, docs in documents.items():
        file_ids_per_storage[storage_name] = {doc.metadata["file_id"] for doc in docs}
    return file_ids_per_storage


def delete_document(url, file_id, data):
    headers = {
        "Content-Type": "application/json",
    }
    response = requests.delete(url.replace("<file_id>", str(file_id)), json=data, headers=headers)  # nosec
    print(f"Deleting file_id {file_id}: {response.status_code}")
    print(response.json())


def main():
    pinecone_connection, weaviate_connection, chroma_connection = (
        initialize_connections()
    )
    pinecone_vector_store, chroma_vector_store, weaviate_vector_store = (
        initialize_vector_stores(
            pinecone_connection, weaviate_connection, chroma_connection
        )
    )

    vector_stores = {
        "Pinecone": pinecone_vector_store,
        "Chroma": chroma_vector_store,
        "Weaviate": weaviate_vector_store,
    }

    count_documents(vector_stores)
    documents = list_documents(vector_stores)
    file_ids_per_storage = gather_file_ids_per_storage(documents)

    for file_id in file_ids_per_storage["Pinecone"]:
        delete_document(
            url="http://localhost:8000/v1/knowledgebase-files/<file_id>",
            file_id=file_id,
            data={
                "type": "dynamiq.connections.Pinecone",
                "params": {
                    "api_key": pinecone_connection.api_key,
                    "index_name": "default",
                },
            },
        )

    for file_id in file_ids_per_storage["Weaviate"]:
        delete_document(
            url="http://localhost:8000/v1/knowledgebase-files/<file_id>",
            file_id=file_id,
            data={
                "type": "dynamiq.connections.Weaviate",
                "params": {
                    "api_key": weaviate_connection.api_key,
                    "url": weaviate_connection.url,
                    "index_name": "default",
                },
            },
        )

    for file_id in file_ids_per_storage["Chroma"]:
        delete_document(
            url="http://localhost:8000/v1/knowledgebase-files/<file_id>",
            file_id=file_id,
            data={
                "type": "dynamiq.connections.Chroma",
                "params": {
                    "host": chroma_connection.host,
                    "port": chroma_connection.port,
                    "index_name": "default",
                },
            },
        )


if __name__ == "__main__":
    main()
