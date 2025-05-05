"""
This example demonstrates how to use the delete_documents_by_file_ids method
in the Pinecone vector store to delete documents by multiple file IDs.

The example:
1. Creates a test Pinecone index
2. Adds sample documents with different file_ids (some sharing the same file_id)
3. Lists documents before deletion
4. Deletes documents using delete_documents_by_file_ids
5. Lists documents after deletion to verify the operation

Requirements:
- Pinecone API key set as environment variable PINECONE_API_KEY
- Pinecone cloud and region settings or pod type and environment settings

Environment variables needed:
- PINECONE_API_KEY: Your Pinecone API key
- PINECONE_CLOUD: Your Pinecone cloud (e.g., "aws")
- PINECONE_REGION: Your Pinecone region (e.g., "us-west-2")
"""

import os
import time

import numpy as np
from dotenv import load_dotenv

from dynamiq.connections import Pinecone
from dynamiq.storages.vector.pinecone.pinecone import PineconeIndexType, PineconeVectorStore
from dynamiq.types import Document

# Load environment variables
load_dotenv()

# Define test index name with timestamp to ensure uniqueness
TEST_INDEX_NAME = f"delete-file-ids-test-{int(time.time())}"
TEST_NAMESPACE = "default"
EMBEDDING_DIM = 768


def create_random_embedding(dim=EMBEDDING_DIM):
    """Create a random embedding vector with the specified dimension."""
    return np.random.rand(dim).astype(float).tolist()


def setup_pinecone_store():
    """Create a test Pinecone vector store."""
    try:
        # Connect to Pinecone
        connection = Pinecone()

        # Create a test vector store
        store = PineconeVectorStore(
            connection=connection,
            index_name=TEST_INDEX_NAME,
            namespace=TEST_NAMESPACE,
            dimension=EMBEDDING_DIM,
            metric="cosine",
            create_if_not_exist=True,
            index_type=PineconeIndexType.SERVERLESS,
            cloud=os.getenv("PINECONE_CLOUD", "aws"),
            region=os.getenv("PINECONE_REGION", "us-east-1"),
        )

        print(f"Created Pinecone index: {TEST_INDEX_NAME}")
        return store
    except Exception as e:
        print(f"Error setting up Pinecone store: {e}")
        raise


def create_sample_documents():
    """Create sample documents with various file_ids for testing."""
    # Group 1: Documents with file_id = "file1"
    doc1 = Document(
        id="doc1",
        content="This is document 1 from file 1",
        metadata={"file_id": "file1", "type": "text", "category": "test"},
        embedding=create_random_embedding(),
    )

    doc2 = Document(
        id="doc2",
        content="This is document 2 from file 1",
        metadata={"file_id": "file1", "type": "text", "category": "test"},
        embedding=create_random_embedding(),
    )

    # Group 2: Documents with file_id = "file2"
    doc3 = Document(
        id="doc3",
        content="This is document 3 from file 2",
        metadata={"file_id": "file2", "type": "text", "category": "test"},
        embedding=create_random_embedding(),
    )

    # Group 3: Documents with file_id = "file3"
    doc4 = Document(
        id="doc4",
        content="This is document 4 from file 3",
        metadata={"file_id": "file3", "type": "text", "category": "test"},
        embedding=create_random_embedding(),
    )

    doc5 = Document(
        id="doc5",
        content="This is document 5 from file 3",
        metadata={"file_id": "file3", "type": "text", "category": "test"},
        embedding=create_random_embedding(),
    )

    # Group 4: Documents with file_id = "file4" (will not be deleted)
    doc6 = Document(
        id="doc6",
        content="This is document 6 from file 4",
        metadata={"file_id": "file4", "type": "text", "category": "test"},
        embedding=create_random_embedding(),
    )

    return [doc1, doc2, doc3, doc4, doc5, doc6]


def print_document_summary(documents):
    """Print a summary of the documents."""
    if not documents:
        print("No documents found.")
        return

    # Group documents by file_id
    file_id_groups = {}
    for doc in documents:
        file_id = doc.metadata.get("file_id", "unknown")
        if file_id not in file_id_groups:
            file_id_groups[file_id] = []
        file_id_groups[file_id].append(doc.id)

    print(f"Found {len(documents)} documents across {len(file_id_groups)} file IDs:")
    for file_id, doc_ids in file_id_groups.items():
        print(f"  - File ID '{file_id}': {len(doc_ids)} documents - {doc_ids}")


def main():
    """Main function to demonstrate delete_documents_by_file_ids functionality."""
    try:
        # Setup Pinecone store
        store = setup_pinecone_store()

        # Create and insert sample documents
        documents = create_sample_documents()
        num_written = store.write_documents(documents)
        print(f"Inserted {num_written} documents into Pinecone")

        # Wait for Pinecone to process the writes
        print("Waiting for Pinecone to process the writes...")
        time.sleep(10)

        # List documents before deletion
        print("\n--- Documents before deletion ---")
        before_docs = store.list_documents()
        print_document_summary(before_docs)

        # Delete documents by file IDs
        file_ids_to_delete = ["file1", "file3"]
        print(f"\nDeleting documents with file_ids: {file_ids_to_delete}")
        store.delete_documents_by_file_ids(file_ids_to_delete)

        # Wait for Pinecone to process the deletes
        print("Waiting for Pinecone to process the deletes...")
        time.sleep(10)

        # List documents after deletion
        print("\n--- Documents after deletion ---")
        after_docs = store.list_documents()
        print_document_summary(after_docs)

        # Verify the deletion
        expected_remaining = len(documents) - sum(
            1 for doc in documents if doc.metadata.get("file_id") in file_ids_to_delete
        )
        actual_remaining = len(after_docs)

        if actual_remaining == expected_remaining:
            print(f"\nSuccess! Found {actual_remaining} documents after deletion (expected {expected_remaining})")
        else:
            print(
                f"\nUnexpected result: Found {actual_remaining} documents "
                f"after deletion (expected {expected_remaining})"
            )

        # Test batching with a larger list
        print("\n--- Testing batch deletion with many file IDs ---")
        # Generate a large list of file IDs (most won't exist)
        many_file_ids = [f"nonexistent_file_{i}" for i in range(1000)]
        # Add actual file ID to ensure something is deleted
        many_file_ids.append("file2")

        # Before deletion
        before_batch_delete = store.list_documents()
        print(f"Documents before batch deletion: {len(before_batch_delete)}")

        # Delete with batch
        print(f"Deleting documents with {len(many_file_ids)} file IDs in batches...")
        store.delete_documents_by_file_ids(many_file_ids, batch_size=500)

        # Wait for Pinecone to process the deletes
        time.sleep(10)

        # After deletion
        after_batch_delete = store.list_documents()
        print(f"Documents after batch deletion: {len(after_batch_delete)}")
        print_document_summary(after_batch_delete)

        # Finally, clean up by deleting the test index
        print(f"\nCleaning up by deleting index {TEST_INDEX_NAME}...")
        store.delete_index()
        print("Cleanup complete!")

    except Exception as e:
        print(f"Error in the example: {e}")
        # Try to clean up
        try:
            connection = Pinecone()
            available_indexes = connection.connect().list_indexes().index_list["indexes"]
            index_exists = any(index["name"] == TEST_INDEX_NAME for index in available_indexes)
            if index_exists:
                connection.connect().delete_index(TEST_INDEX_NAME)
                print(f"Cleaned up index {TEST_INDEX_NAME}")
        except Exception as cleanup_error:
            print(f"Error during cleanup: {cleanup_error}")


if __name__ == "__main__":
    main()
