"""PGVector storage implementation for Dynamiq."""
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from pgvector.psycopg2 import register_vector
import psycopg2
from psycopg2.extensions import connection
from psycopg2.extras import execute_values

from dynamiq.storages.vector.base import VectorStorage
from dynamiq.storages.vector.exceptions import VectorStorageError
from dynamiq.types import Document


class PGVectorStorage(VectorStorage):
    """PGVector storage implementation."""

    def __init__(
        self,
        connection_string: str,
        collection_name: str = "dynamiq_vectors",
        dimension: int = 1536,
    ):
        """Initialize PGVector storage.

        Args:
            connection_string: PostgreSQL connection string
            collection_name: Name of the collection/table to store vectors
            dimension: Dimension of vectors to store
        """
        super().__init__()
        self.connection_string = connection_string
        self.collection_name = collection_name
        self.dimension = dimension
        self._conn: Optional[connection] = None

    def _get_connection(self) -> connection:
        """Get PostgreSQL connection with pgvector extension."""
        if self._conn is None or self._conn.closed:
            try:
                self._conn = psycopg2.connect(self.connection_string)
                register_vector(self._conn)
                with self._conn.cursor() as cur:
                    # Create extension if not exists
                    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                    # Create table if not exists
                    cur.execute(
                        f"""
                        CREATE TABLE IF NOT EXISTS {self.collection_name} (
                            id TEXT PRIMARY KEY,
                            embedding vector({self.dimension}),
                            metadata JSONB,
                            content TEXT
                        );
                        """
                    )
                    # Create index if not exists
                    cur.execute(
                        f"""
                        CREATE INDEX IF NOT EXISTS {self.collection_name}_embedding_idx 
                        ON {self.collection_name} 
                        USING ivfflat (embedding vector_cosine_ops);
                        """
                    )
                self._conn.commit()
            except Exception as e:
                raise VectorStorageError(f"Failed to connect to PGVector: {str(e)}")
        return self._conn

    def add(
        self,
        documents: List[Document],
        embeddings: List[List[float]],
        **kwargs: Any,
    ) -> List[str]:
        """Add documents with embeddings to storage.

        Args:
            documents: List of documents to add
            embeddings: List of embeddings corresponding to documents
            **kwargs: Additional arguments

        Returns:
            List of document IDs
        """
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                data = [
                    (doc.id, embedding, doc.metadata, doc.content)
                    for doc, embedding in zip(documents, embeddings)
                ]
                execute_values(
                    cur,
                    f"""
                    INSERT INTO {self.collection_name} (id, embedding, metadata, content)
                    VALUES %s
                    ON CONFLICT (id) DO UPDATE SET
                        embedding = EXCLUDED.embedding,
                        metadata = EXCLUDED.metadata,
                        content = EXCLUDED.content;
                    """,
                    data,
                    template="(%s, %s::vector, %s::jsonb, %s)",
                )
            conn.commit()
            return [doc.id for doc in documents]
        except Exception as e:
            conn.rollback()
            raise VectorStorageError(f"Failed to add documents: {str(e)}")

    def search(
        self,
        query_embedding: List[float],
        limit: int = 10,
        **kwargs: Any,
    ) -> List[Tuple[Document, float]]:
        """Search for similar documents.

        Args:
            query_embedding: Query embedding to search for
            limit: Maximum number of results to return
            **kwargs: Additional arguments

        Returns:
            List of (document, score) tuples
        """
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    SELECT id, metadata, content, 1 - (embedding <=> %s::vector) as similarity
                    FROM {self.collection_name}
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s;
                    """,
                    (query_embedding, query_embedding, limit),
                )
                results = []
                for id_, metadata, content, similarity in cur.fetchall():
                    doc = Document(id=id_, metadata=metadata, content=content)
                    results.append((doc, float(similarity)))
                return results
        except Exception as e:
            raise VectorStorageError(f"Failed to search documents: {str(e)}")

    def delete(self, document_ids: List[str], **kwargs: Any) -> None:
        """Delete documents from storage.

        Args:
            document_ids: List of document IDs to delete
            **kwargs: Additional arguments
        """
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    f"DELETE FROM {self.collection_name} WHERE id = ANY(%s);",
                    (document_ids,),
                )
            conn.commit()
        except Exception as e:
            conn.rollback()
            raise VectorStorageError(f"Failed to delete documents: {str(e)}")

    def clear(self) -> None:
        """Clear all documents from storage."""
        conn = self._get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(f"TRUNCATE TABLE {self.collection_name};")
            conn.commit()
        except Exception as e:
            conn.rollback()
            raise VectorStorageError(f"Failed to clear storage: {str(e)}")

    def close(self) -> None:
        """Close connection to storage."""
        if self._conn is not None and not self._conn.closed:
            self._conn.close()
            self._conn = None
