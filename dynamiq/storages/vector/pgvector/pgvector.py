from contextlib import contextmanager
from enum import Enum
from typing import Any, Optional

import psycopg
from psycopg import Cursor
from psycopg.rows import dict_row
from psycopg.sql import SQL, Identifier
from psycopg.sql import Literal as SQLLiteral
from psycopg.types.json import Jsonb

from dynamiq.connections import PGVector
from dynamiq.storages.vector.exceptions import VectorStoreException
from dynamiq.storages.vector.pgvector.filters import _convert_filters_to_query
from dynamiq.types import Document
from dynamiq.utils.logger import logger


class PGVectorEnum(str, Enum):
    def __contains__(cls, item: str) -> bool:
        return item in [e.value for e in cls]

    @classmethod
    def list(cls):
        return list(map(lambda c: c.value, cls))

    @classmethod
    def values(cls):
        return list(cls)


class PGVectorVectorFunction(PGVectorEnum):
    COSINE_SIMILARITY = "cosine_similarity"
    INNER_PRODUCT = "inner_product"
    L2_DISTANCE = "l2_distance"
    L1_DISTANCE = "l1_distance"


class PGVectorIndexMethod(PGVectorEnum):
    EXACT = "exact_nearest_neighbor_search"
    IVFFLAT = "ivfflat"
    HNSW = "hnsw"


VECTOR_FUNCTION_TO_POSTGRESQL_OPS = {
    PGVectorVectorFunction.COSINE_SIMILARITY: "vector_cosine_ops",
    PGVectorVectorFunction.INNER_PRODUCT: "vector_ip_ops",
    PGVectorVectorFunction.L2_DISTANCE: "vector_l2_ops",
    PGVectorVectorFunction.L1_DISTANCE: "vector_l1_ops",
}

VECTOR_FUNCTION_TO_SCORE_DEFINITION = {
    PGVectorVectorFunction.COSINE_SIMILARITY: "1 - (embedding <=> {query_embedding})",
    PGVectorVectorFunction.INNER_PRODUCT: "(embedding <#> {query_embedding}) * -1",
    PGVectorVectorFunction.L2_DISTANCE: "embedding <-> {query_embedding}",
    PGVectorVectorFunction.L1_DISTANCE: "embedding <+> {query_embedding}",
}


class PGVectorStore:
    """Vector store using pgvector."""

    def __init__(
        self,
        connection: PGVector | str | None = None,
        client: Optional["PGVector"] = None,
        table_name: str = "dynamiq",
        schema_name: str = "public",
        dimension: int = 1536,
        vector_function: PGVectorVectorFunction = PGVectorVectorFunction.COSINE_SIMILARITY,
        index_method: PGVectorIndexMethod = PGVectorIndexMethod.EXACT,
        index_name: str | None = None,
        create_if_not_exist: bool = True,
    ):
        if vector_function not in PGVectorVectorFunction:
            raise ValueError(f"vector_function must be one of {PGVectorVectorFunction.list()}")
        if index_method is not None and index_method not in PGVectorIndexMethod:
            raise ValueError(f"index_method must be one of {PGVectorIndexMethod.list()}")

        if isinstance(connection, str):
            self.connection_string = connection
            self._conn = psycopg.connect(self.connection_string)
        elif isinstance(connection, PGVector):
            self._conn = connection.connect()
            self.connection_string = str(connection)
        else:
            raise ValueError("connection must be a string or PGVector object")

        self.client = client if client else self._conn

        self.table_name = table_name
        self.schema_name = schema_name
        self.dimension = dimension
        self.index_method = index_method
        self.vector_function = vector_function

        if create_if_not_exist:
            with self._get_connection() as conn:
                self._create_tables(conn)
                if self.index_method in [PGVectorIndexMethod.IVFFLAT, PGVectorIndexMethod.HNSW]:
                    self.index_name = index_name or f"{self.index_method}_index"
                    self._create_index(conn)

        logger.debug(f"PGVectorStore initialized with table_name: {self.table_name}")

    @contextmanager
    def _get_connection(self):
        """Context manager for handling a single connection"""

        import psycopg

        if self._conn is None or self._conn.closed:
            self._conn = psycopg.connect(self.connection_string)
        try:
            yield self._conn
        except Exception as e:
            self._conn.rollback()
            raise e

    def _execute_sql_query(
        self, sql_query: Any, params: tuple | None = None, cursor: Cursor | None | None = None
    ) -> Any:
        """
        Internal method to execute a SQL query.

        Args:
            sql_query (Any): The SQL query to execute.
            params (tuple | None): The parameters to pass to the query. Defaults to None.
            cursor (Optional[Cursor] | None): The cursor to use for the query. Defaults to None.

        Raises:
            VectorStoreException: If an error occurs while executing the query.

        Returns:
            Any: The result of the query.
        """

        params = params or ()

        sql_query_str = sql_query.as_string(cursor) if not isinstance(sql_query, str) else sql_query

        try:
            result = cursor.execute(sql_query, params)
        except Exception as e:
            self._conn.rollback()
            msg = f"Encountered an error while executing SQL query: {sql_query_str} with params: {params}. \nError: {e}"
            raise VectorStoreException(msg)

        return result

    def _create_tables(self, conn) -> None:
        """
        Internal method to create the tables in the database (if they do not exist).

        Args:
            conn: The connection to the database.
        """

        query = SQL(
            """
            CREATE TABLE IF NOT EXISTS {table_name} (
                id VARCHAR(128) PRIMARY KEY,
                content TEXT,
                metadata JSONB,
                embedding vector({dimension})
            );
            """
        ).format(
            table_name=Identifier(self.table_name),
            dimension=self.dimension,
        )

        with conn.cursor() as cur:
            self._execute_sql_query(query, cursor=cur)
            conn.commit()

    def _drop_tables(self, conn) -> None:
        """
        Internal method to drop the tables in the database (if they exist).

        Args:
            conn: The connection to the database.
        """

        query = SQL(
            """
            DROP TABLE IF EXISTS {table_name};
            """
        ).format(
            table_name=Identifier(self.table_name),
        )

        with conn.cursor() as cur:
            self._execute_sql_query(query, cursor=cur)
            conn.commit()

    def _create_index(self, conn) -> None:
        """
        Internal method to create the index in the database (if it does not exist).
        Should only be called if the index method is either `ivfflat` or `hnsw`.

        Args:
            conn: The connection to the database.

        Raises:
            ValueError: If the index method is not valid.
        """
        if self.index_method not in PGVectorIndexMethod.values():
            msg = f"Invalid index method: {self.index_method}"
            raise ValueError(msg)

        vector_ops = VECTOR_FUNCTION_TO_POSTGRESQL_OPS[self.vector_function]
        query = SQL(
            """
            CREATE INDEX IF NOT EXISTS {index_name}
            ON {table_name} USING {index_method} (embedding {vector_ops});
            """
        ).format(
            index_name=Identifier(f"{self.table_name}_{self.index_method}_index"),
            table_name=Identifier(self.table_name),
            index_method=Identifier(self.index_method),
            vector_ops=Identifier(vector_ops),
        )

        with conn.cursor() as cur:
            self._execute_sql_query(query, cursor=cur)
            conn.commit()

    def _drop_index(self, conn) -> None:
        """
        Internal method to drop the index in the database (if it exists).
        Should only be called if the index method is either `ivfflat` or `hnsw`.

        Args:
            conn: The connection to the database.

        Raises:
            ValueError: If the index method is not valid.
        """
        if self.index_method not in PGVectorIndexMethod.values():
            msg = f"Invalid index method: {self.index_method}"
            raise ValueError(msg)

        query = SQL(
            """
            DROP INDEX IF EXISTS {index_name};
            """
        ).format(
            index_name=Identifier(f"{self.table_name}_{self.index_method}_index"),
        )

        with conn.cursor() as cur:
            self._execute_sql_query(query, cursor=cur)
            conn.commit()

    def count_documents(self) -> int:
        """
        Count the number of documents in the store.

        Returns:
            int: The number of documents in the store.
        """

        with self._get_connection() as conn:
            with conn.cursor() as cur:
                query = SQL("SELECT COUNT(*) FROM {table_name}").format(table_name=Identifier(self.table_name))
                result = self._execute_sql_query(query, cursor=cur)
                return result.fetchone()[0]

    def write_documents(self, documents: list[Document]) -> int:
        """
        Write documents to the pgvector vector store.

        Args:
            documents (list[Document]): List of Document objects to write.

        Returns:
            int: Number of documents successfully written.

        Raises:
            ValueError: If documents are not of type Document.
        """

        if not documents:
            return 0

        if len(documents) > 0 and not isinstance(documents[0], Document):
            msg = "param 'documents' must contain a list of objects of type Document"
            raise ValueError(msg)

        with self._get_connection() as conn:
            with conn.cursor() as cur:
                written = 0
                for doc in documents:
                    query = SQL(
                        """
                        INSERT INTO {table_name} (id, content, metadata, embedding)
                        VALUES (%s, %s, %s, %s)
                        ON CONFLICT (id) DO UPDATE
                        SET content = EXCLUDED.content,
                        metadata = EXCLUDED.metadata,
                        embedding = EXCLUDED.embedding;
                        """
                    ).format(table_name=Identifier(self.table_name))
                    self._execute_sql_query(
                        query, (doc.id, doc.content, Jsonb(doc.metadata), doc.embedding), cursor=cur
                    )
                    written += 1
                conn.commit()
                return written

    def delete_documents_by_filters(self, filters: dict[str, Any], top_k: int = 1000) -> None:
        """
        Delete documents from the pgvector vector store using filters.

        Args:
            filters (dict[str, Any]): Filters to select documents to delete.
        """
        if filters:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    sql_where_clause, params = _convert_filters_to_query(filters)
                    query = SQL("DELETE FROM {table_name}").format(
                        table_name=Identifier(self.table_name),
                        sql_where_clause=sql_where_clause,
                    )
                    query += sql_where_clause
                    self._execute_sql_query(query, params, cursor=cur)
                    conn.commit()
        else:
            logger.warning("No filters provided. No documents will be deleted.")

    def delete_documents(self, document_ids: list[str] | None = None, delete_all: bool = False) -> None:
        """
        Delete documents from the pgvector vector store.

        Args:
            document_ids (list[str]): List of document IDs to delete. Defaults to None.
            delete_all (bool): If True, delete all documents. Defaults to False.
        """
        if delete_all:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    query = SQL("DELETE FROM {table_name}").format(table_name=Identifier(self.table_name))
                    self._execute_sql_query(query, cursor=cur)
                    conn.commit()
        else:
            if not document_ids:
                logger.warning("No document IDs provided. No documents will be deleted.")
            else:
                self.delete_documents_by_file_id(document_ids)

    def delete_documents_by_file_id(self, document_ids: list[str]) -> None:
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                query = SQL("DELETE FROM {table_name} WHERE id = ANY(%s::text[])").format(
                    table_name=Identifier(self.table_name)
                )
                self._execute_sql_query(query, (document_ids,), cursor=cur)
                conn.commit()

    def list_documents(self, include_embeddings: bool = False) -> list[Document]:
        """
        List documents in the pgvector vector store.

        Args:
            include_embeddings (bool): Whether to include embeddings in the results. Defaults to False.

        Returns:
            list[Document]: List of Document objects retrieved.
        """
        select_fields = "id, content, metadata" + (", embedding" if include_embeddings else "")
        with self._get_connection() as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                query = SQL("SELECT {select_fields} FROM {table_name}").format(
                    select_fields=SQL(select_fields),
                    table_name=Identifier(self.table_name),
                )
                result = self._execute_sql_query(query, cursor=cur)
                records = result.fetchall()

                documents = self._convert_query_result_to_documents(records)
                return documents

    def _convert_query_result_to_documents(
        self,
        query_result: dict[str, Any],
    ) -> list[Document]:
        """
        Convert pgvector query results to Document objects.

        Args:
            query_result (dict[str, Any]): The query result from pgvector.

        Returns:
            list[Document]: List of Document objects created from the query result.
        """
        documents = []
        for doc in query_result:
            document = Document(
                id=doc["id"],
                content=doc["content"],
                metadata=doc["metadata"],
            )

            if doc.get("embedding") is not None:
                document.embedding = self._convert_pg_embedding_to_list(doc["embedding"])
            else:
                document.embedding = None

            if doc.get("score") is not None:
                document.score = doc["score"]
            else:
                document.score = None

            documents.append(document)
        return documents

    def _convert_pg_embedding_to_list(self, pg_embedding: Any) -> list[float]:
        """
        Helper method to convert a pgvector embedding type to a list of floats.
        e.g. '[0.1,0.2,0.3]' -> [0.1, 0.2, 0.3]

        Args:
            pg_embedding (Any): The pgvector embedding.

        Returns:
            list[float]: The embedding as a list of floats.
        """

        if isinstance(pg_embedding, str):
            return [float(x) for x in pg_embedding.strip("[]").split(",") if x]
        return pg_embedding

    def _convert_query_embedding_to_pgvector_format(self, query_embedding: list[float]) -> str:
        """
        Helper method to convert query embedding to pgvector format.
        e.g. [0.1, 0.2, 0.3] -> '[0.1,0.2,0.3]'

        Args:
            query_embedding (list[float]): The query embedding vector.

        Returns:
            str: The query embedding in pgvector format (e.g. '[0.1,0.2,0.3]').
        """
        return f"'[{','.join(str(el) for el in query_embedding)}]'"

    def _embedding_retrieval(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        exclude_document_embeddings: bool = True,
        filters: dict[str, Any] | None = None,
    ) -> list[Document]:
        """
        Retrieve documents similar to the given query embedding.

        Args:
            query_embedding (list[float]): The query embedding vector.
            filters (dict[str, Any] | None): Filters for the query. Defaults to None.
            top_k (int): Maximum number of documents to retrieve. Defaults to 10.
            exclude_document_embeddings (bool): Whether to exclude embeddings in results. Defaults to True.

        Returns:
            list[Document]: List of retrieved Document objects.

        Raises:
            ValueError: If query_embedding is empty or filter format is incorrect.
        """
        if not query_embedding:
            msg = "query_embedding must be a non-empty list"
            raise ValueError(msg)

        if len(query_embedding) != self.dimension:
            msg = f"query_embedding must be of dimension {self.dimension}"
            raise ValueError(msg)

        vector_function = self.vector_function

        if vector_function not in PGVectorVectorFunction:
            msg = f"Invalid vector function: {vector_function}"
            raise ValueError(msg)

        query_embedding = self._convert_query_embedding_to_pgvector_format(query_embedding)

        # Generate the score calculation based on the vector function
        score_definition = VECTOR_FUNCTION_TO_SCORE_DEFINITION[vector_function].format(query_embedding=query_embedding)
        score_definition = f"{score_definition} AS score"

        # Do not select the embeddings if exclude_document_embeddings is True
        select_fields = "id, content, metadata" if exclude_document_embeddings else "*"

        # Build the base SELECT query with score
        base_select = SQL("SELECT {fields}, {score} FROM {table_name}").format(
            fields=SQL(select_fields), score=SQL(score_definition), table_name=Identifier(self.table_name)
        )

        # Handle filters if they exist
        where_clause = SQL("")
        params = ()
        if filters:
            where_clause, params = _convert_filters_to_query(filters)

        # Determine sort order based on vector function type
        is_distance_metric = vector_function in ["l2_distance", "l1_distance"]
        
        # Sort by score in ascending order if using a distance metric
        # as the smaller the distance, the more similar the vectors are
        sort_order = "ASC" if is_distance_metric else "DESC"

        # Build the ORDER BY and LIMIT clause
        order_by = SQL(" ORDER BY score {sort_order} LIMIT {limit}").format(
            sort_order=SQL(sort_order), limit=SQLLiteral(top_k)
        )

        # Combine all parts into final query
        sql_query = base_select + where_clause + order_by

        with self._get_connection() as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                result = self._execute_sql_query(sql_query, params, cursor=cur)
                records = result.fetchall()

                documents = self._convert_query_result_to_documents(records)
                return documents

    def __del__(self):
        """Close the connection when the object is deleted."""
        if hasattr(self, "_conn") and self._conn is not None and not self._conn.closed:
            self._conn.close()
