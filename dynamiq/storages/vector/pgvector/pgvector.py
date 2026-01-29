import re
from contextlib import contextmanager
from enum import Enum
from typing import TYPE_CHECKING, Any, Optional

import psycopg
from pgvector.psycopg import register_vector
from psycopg import Cursor
from psycopg import errors as psycopg_errors
from psycopg.rows import dict_row
from psycopg.sql import SQL, Identifier
from psycopg.sql import Literal as SQLLiteral
from psycopg.types.json import Jsonb

from dynamiq.connections import PostgreSQL
from dynamiq.nodes.dry_run import DryRunMixin
from dynamiq.storages.vector.base import BaseVectorStore, BaseVectorStoreParams, BaseWriterVectorStoreParams
from dynamiq.storages.vector.exceptions import VectorStoreException
from dynamiq.storages.vector.pgvector.filters import _convert_filters_to_query
from dynamiq.storages.vector.utils import create_pgvector_file_id_filter, create_pgvector_file_ids_filter
from dynamiq.types import Document
from dynamiq.types.constants import SANITIZED_VALUE_PLACEHOLDER
from dynamiq.types.dry_run import DryRunConfig
from dynamiq.utils.logger import logger

if TYPE_CHECKING:
    from psycopg import Connection as PsycopgConnection


class PGVectorVectorFunction(str, Enum):
    COSINE_SIMILARITY = "cosine_similarity"
    INNER_PRODUCT = "inner_product"
    L2_DISTANCE = "l2_distance"
    L1_DISTANCE = "l1_distance"


class PGVectorIndexMethod(str, Enum):
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
    PGVectorVectorFunction.COSINE_SIMILARITY: "1 - ({embedding_key} <=> {query_embedding})",
    PGVectorVectorFunction.INNER_PRODUCT: "({embedding_key} <#> {query_embedding}) * -1",
    PGVectorVectorFunction.L2_DISTANCE: "{embedding_key} <-> {query_embedding}",
    PGVectorVectorFunction.L1_DISTANCE: "{embedding_key} <+> {query_embedding}",
}

DEFAULT_TABLE_NAME = "default"
DEFAULT_SCHEMA_NAME = "public"
DEFAULT_LANGUAGE = "english"

DEFAULT_KEYWORD_RANK_CONSTANT = 60
DEFAULT_TOP_K_SUBQUERY_MULTIPLIER = 4

DEFAULT_IVFFLAT_LISTS = None  # Auto-calculated
DEFAULT_HNSW_M = 16
DEFAULT_HNSW_EF_CONSTRUCTION = 64


class PGVectorStoreParams(BaseVectorStoreParams):
    table_name: str = DEFAULT_TABLE_NAME
    schema_name: str = DEFAULT_SCHEMA_NAME
    dimension: int = 1536
    vector_function: PGVectorVectorFunction = PGVectorVectorFunction.COSINE_SIMILARITY
    embedding_key: str = "embedding"
    index_name: str | None = None
    keyword_index_name: str | None = None


class PGVectorStoreRetrieverParams(PGVectorStoreParams):
    alpha: float = 0.5


class PGVectorStoreWriterParams(PGVectorStoreParams, BaseWriterVectorStoreParams):
    create_if_not_exist: bool = False


class PGVectorStore(BaseVectorStore, DryRunMixin):
    """Vector store using pgvector."""

    def __init__(
        self,
        table_name: str = DEFAULT_TABLE_NAME,
        connection: PostgreSQL | str | None = None,
        client: Optional["PsycopgConnection"] = None,
        create_extension: bool = True,
        schema_name: str = DEFAULT_SCHEMA_NAME,
        dimension: int = 1536,
        vector_function: PGVectorVectorFunction = PGVectorVectorFunction.COSINE_SIMILARITY,
        index_method: PGVectorIndexMethod = PGVectorIndexMethod.EXACT,
        index_name: str | None = None,
        create_if_not_exist: bool = False,
        content_key: str = "content",
        embedding_key: str = "embedding",
        keyword_index_name: str | None = None,
        language: str = DEFAULT_LANGUAGE,
        ivfflat_lists: int | None = DEFAULT_IVFFLAT_LISTS,
        hnsw_m: int = DEFAULT_HNSW_M,
        hnsw_ef_construction: int = DEFAULT_HNSW_EF_CONSTRUCTION,
        set_runtime_params: bool = True,
        dry_run_config: DryRunConfig | None = None,
    ):
        """
        Initialize a PGVectorStore instance.

        Args:
            connection (PostgreSQL | str): PostgreSQL connection instance. Defaults to None.
            client (Optional[PostgreSQL]): PostgreSQL client instance. Defaults to None.
            create_extension (bool): Whether to create the vector extension (if it does not exist). Defaults to True.
            table_name (str): Name of the table in the database. Defaults to "default".
            schema_name (str): Name of the schema in the database.
            dimension (int): Dimension of the embeddings. Defaults to 1536.
            vector_function (PGVectorVectorFunction): The vector function to use for similarity calculations.
            index_method (PGVectorIndexMethod): The index method to use for the vector store.
            index_name (str | None): Name of the index to create.
            create_if_not_exist (bool): Whether to create the table and index if they do not exist.
            content_key (str): The field used to store content in the storage.
            embedding_key (str): The field used to store embeddings in the storage.
            keyword_index_name (str | None): Name of the keyword index.
            language (str): Language for full-text search.
            ivfflat_lists (int | None): Number of lists for IVFFLAT index. Auto-calculated if None.
            hnsw_m (int): Number of connections per layer for HNSW index.
            hnsw_ef_construction (int): Size of the dynamic candidate list for HNSW index construction.
            set_runtime_params (bool): Whether to automatically set pgvector runtime parameters for optimal performance.
            dry_run_config (Optional[DryRunConfig]): Configuration for dry run mode.
        """
        super().__init__(dry_run_config=dry_run_config)

        if vector_function not in PGVectorVectorFunction:
            raise ValueError(f"vector_function must be one of {list(PGVectorVectorFunction)}")
        if index_method is not None and index_method not in PGVectorIndexMethod:
            raise ValueError(f"index_method must be one of {list(PGVectorIndexMethod)}")

        if client is None or (hasattr(client, "closed") and client.closed):
            if isinstance(connection, str):
                self.connection_string = connection
                self._conn = psycopg.connect(self.connection_string)
            elif isinstance(connection, PostgreSQL):
                self._conn = connection.connect()
                self.connection_string = connection.conn_params
            else:
                raise ValueError("Either 'connection' (str or PostgreSQL) or 'client' must be provided")
            self.client = self._conn
        else:
            self.client = client
            self._conn = client
            self.connection_string = None

        self.create_extension = create_extension
        if self.create_extension:
            self._conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            self._conn.commit()

        register_vector(self._conn)

        self.table_name = table_name
        self.schema_name = schema_name
        self.dimension = dimension
        self.index_method = index_method
        self.vector_function = vector_function
        self.index_name = index_name or f"{self.table_name}_{self.index_method}_index"
        self.keyword_index_name = keyword_index_name or f"{self.table_name}_keyword_index"
        self.language = language
        self.ivfflat_lists = ivfflat_lists
        self.hnsw_m = hnsw_m
        self.hnsw_ef_construction = hnsw_ef_construction
        self.set_runtime_params = set_runtime_params

        self.content_key = content_key
        self.embedding_key = embedding_key

        self.create_if_not_exist = create_if_not_exist

        if (
            self.index_method == PGVectorIndexMethod.IVFFLAT
            and self.vector_function == PGVectorVectorFunction.L1_DISTANCE
        ):
            msg = "IVFFLAT index does not support L1 distance metric"
            raise VectorStoreException(msg)

        if self.create_if_not_exist:
            with self._get_connection() as conn:
                # Check if table exists before creating it
                table_exists = self._check_if_table_exists(conn)

                self._create_schema(conn)
                self._create_tables(conn)
                if self.index_method in [PGVectorIndexMethod.IVFFLAT, PGVectorIndexMethod.HNSW]:
                    self._create_index(conn)
                self._create_keyword_index(conn)

                if not table_exists:
                    self._track_collection(f"{self.schema_name}.{self.table_name}")
        else:
            try:
                if not self._check_if_schema_exists(self._conn):
                    msg = f"Schema '{self.schema_name}' does not exist"
                    raise VectorStoreException(msg)
                if not self._check_if_table_exists(self._conn):
                    msg = f"Table '{self.table_name}' does not exist"
                    raise VectorStoreException(msg)
            except Exception:
                self._safe_rollback()
                raise

        logger.debug(f"PGVectorStore initialized with table_name: {self.table_name}")

    def _sanitize_error_message(self, error_msg: str) -> str:
        """
        Sanitize error messages to prevent exposure of credentials.

        Args:
            error_msg: Raw error message from database operations.

        Returns:
            Sanitized error message safe for logging.
        """
        sanitized = error_msg

        # Remove connection string patterns
        sanitized = re.sub(r'postgresql://[^\'"\s]+', f"postgresql://{SANITIZED_VALUE_PLACEHOLDER}", sanitized)
        sanitized = re.sub(r'password=[^\'"\s&]+', f"password={SANITIZED_VALUE_PLACEHOLDER}", sanitized)
        sanitized = re.sub(r'user=[^\'"\s&]+', f"user={SANITIZED_VALUE_PLACEHOLDER}", sanitized)

        if len(sanitized) > 500:
            sanitized = sanitized[:497] + "..."

        return sanitized

    def _prepare_filters(self, filters: dict[str, Any] | None) -> tuple[SQL, tuple]:
        """
        Prepare filters for SQL queries.

        Args:
            filters: Dictionary of filters to convert to SQL WHERE clause.

        Returns:
            Tuple of (SQL where clause, parameter tuple).
        """
        if not filters:
            return SQL(""), ()
        return _convert_filters_to_query(filters)

    def _set_pgvector_runtime_params(self, conn: "PsycopgConnection", top_k: int) -> None:
        """
        Set pgvector runtime parameters for optimal query performance.

        Args:
            conn: Database connection to set parameters on.
            top_k: Number of results requested, used to tune parameters.
        """
        if not self.set_runtime_params:
            return

        if self.index_method == PGVectorIndexMethod.IVFFLAT:
            # For IVFFLAT, probes should be proportional to top_k but capped
            probes = max(top_k // 10, 1)
            probes = min(probes, 100)
            conn.execute(f"SET ivfflat.probes = {probes}")
        elif self.index_method == PGVectorIndexMethod.HNSW:
            # For HNSW, ef_search should be at least top_k
            ef_search = max(top_k * 2, 40)
            ef_search = min(ef_search, 1000)
            conn.execute(f"SET hnsw.ef_search = {ef_search}")

    @contextmanager
    def _get_connection(self):
        """Context manager for handling a single connection"""
        if self._conn is None or self._conn.closed:
            if not self.connection_string:
                raise VectorStoreException("Connection is closed and no connection string available for reconnection")
            self._conn = psycopg.connect(self.connection_string)
            register_vector(self._conn)
            self.client = self._conn
        try:
            yield self._conn
        except Exception as e:
            self._safe_rollback()
            raise e

    def _check_if_schema_exists(self, conn: psycopg.Connection) -> bool:
        """
        Check if the schema exists in the database.

        Args:
            conn (psycopg.Connection): The connection to the database.

        Returns:
            bool: True if the schema exists, False otherwise.
        """

        query = SQL(
            """
            SELECT EXISTS (
                SELECT 1
                FROM information_schema.schemata
                WHERE schema_name = %s
            );
            """
        )

        with conn.cursor(row_factory=dict_row) as cur:
            result = self._execute_sql_query(query, (self.schema_name,), cursor=cur).fetchone()
            return bool(result["exists"]) if isinstance(result, dict) else bool(result[0])

    def _check_if_table_exists(self, conn: psycopg.Connection) -> bool:
        """
        Check if the table exists in the database.

        Args:
            conn (psycopg.Connection): The connection to the database.

        Returns:
            bool: True if the table exists, False otherwise.
        """

        query = SQL(
            """
            SELECT EXISTS (
                SELECT 1
                FROM information_schema.tables
                WHERE table_schema = %s
                AND table_name = %s
            );
            """
        )

        with conn.cursor(row_factory=dict_row) as cur:
            result = self._execute_sql_query(query, (self.schema_name, self.table_name), cursor=cur).fetchone()
            return bool(result["exists"]) if isinstance(result, dict) else bool(result[0])

    def _safe_rollback(self) -> None:
        """
        Safely attempt to rollback the current transaction.
        """
        if self._conn and not self._conn.closed:
            try:
                self._conn.rollback()
            except Exception as rollback_error:
                logger.warning(f"Failed to rollback transaction: {rollback_error}")

    def _execute_sql_query(self, sql_query: Any, params: tuple | None = None, cursor: Cursor | None = None) -> Cursor:
        """
        Internal method to execute a SQL query.

        Args:
            sql_query (Any): The SQL query to execute.
            params (tuple | None): The parameters to pass to the query. Defaults to None.
            cursor (Cursor | None): The cursor to use for the query. Defaults to None.

        Raises:
            VectorStoreException: If an error occurs while executing the query.
            ValueError: If invalid data is provided.

        Returns:
            Cursor: The cursor with query results.
        """

        params = params or ()

        try:
            result = cursor.execute(sql_query, params)
            return result

        except psycopg_errors.OperationalError as e:
            # Connection issues
            self._safe_rollback()
            sanitized_error = self._sanitize_error_message(str(e))
            logger.warning(f"Database connection error: {sanitized_error}")
            raise VectorStoreException("Database connection error") from e

        except psycopg_errors.UniqueViolation as e:
            # Duplicate key
            self._safe_rollback()
            sanitized_error = self._sanitize_error_message(str(e))
            logger.debug(f"Duplicate key violation: {sanitized_error}")
            raise VectorStoreException("Document already exists") from e

        except psycopg_errors.DataError as e:
            # Invalid data (e.g., wrong vector dimension, invalid JSON)
            self._safe_rollback()
            sanitized_error = self._sanitize_error_message(str(e))
            logger.error(f"Data validation error: {sanitized_error}")
            raise ValueError("Invalid document data provided") from e

        except psycopg_errors.SyntaxError as e:
            # SQL syntax error
            self._safe_rollback()
            sanitized_error = self._sanitize_error_message(str(e))
            logger.error(f"SQL syntax error: {sanitized_error}")
            raise VectorStoreException("Database query syntax error") from e

        except psycopg_errors.InsufficientPrivilege as e:
            # Permission error
            self._safe_rollback()
            sanitized_error = self._sanitize_error_message(str(e))
            logger.error(f"Insufficient privileges: {sanitized_error}")
            raise VectorStoreException("Insufficient database privileges") from e

        except Exception as e:
            self._safe_rollback()
            sanitized_error = self._sanitize_error_message(str(e))
            logger.error(f"Unexpected database error: {sanitized_error}", exc_info=True)
            raise VectorStoreException("Unexpected database error") from e

    def _create_tables(
        self,
        conn: psycopg.Connection,
        content_key: str | None = None,
        embedding_key: str | None = None,
    ) -> None:
        """
        Internal method to create the tables in the database (if they do not exist).

        Args:
            conn (psycopg.Connection): The connection to the database.
            content_key (str | None): The field used to store content in the storage. Defaults to None.
            embedding_key (str | None): The field used to store embeddings in the storage. Defaults to None.
        """

        content_key = content_key or self.content_key
        embedding_key = embedding_key or self.embedding_key

        query = SQL(
            """
            CREATE TABLE IF NOT EXISTS {schema_name}.{table_name} (
                id VARCHAR(128) PRIMARY KEY,
                {content_key} TEXT,
                metadata JSONB,
                {embedding_key} vector({dimension})
            );
            """
        ).format(
            schema_name=Identifier(self.schema_name),
            table_name=Identifier(self.table_name),
            content_key=Identifier(content_key),
            embedding_key=Identifier(embedding_key),
            dimension=self.dimension,
        )

        with conn.cursor(row_factory=dict_row) as cur:
            self._execute_sql_query(query, cursor=cur)
            conn.commit()

    def _drop_tables(self, conn: psycopg.Connection) -> None:
        """
        Internal method to drop the tables in the database (if they exist).

        Args:
            conn (psycopg.Connection): The connection to the database.
        """

        query = SQL(
            """
            DROP TABLE IF EXISTS {schema_name}.{table_name};
            """
        ).format(
            schema_name=Identifier(self.schema_name),
            table_name=Identifier(self.table_name),
        )

        with conn.cursor(row_factory=dict_row) as cur:
            self._execute_sql_query(query, cursor=cur)
            conn.commit()

    def _create_index(
        self,
        conn: psycopg.Connection,
        embedding_key: str | None = None,
    ) -> None:
        """
        Internal method to create the index in the database (if it does not exist).
        Should only be called if the index method is either `ivfflat` or `hnsw`.

        Args:
            conn (psycopg.Connection): The connection to the database.
            embedding_key (str | None): The field used to store embeddings in the storage. Defaults to None.

        Raises:
            ValueError: If the index method is not valid.
        """

        embedding_key = embedding_key or self.embedding_key

        if self.index_method not in PGVectorIndexMethod:
            msg = f"Invalid index method: {self.index_method}"
            raise ValueError(msg)

        vector_ops = VECTOR_FUNCTION_TO_POSTGRESQL_OPS[self.vector_function]

        if self.index_method == PGVectorIndexMethod.IVFFLAT:
            lists = self.ivfflat_lists
            if lists is None:
                try:
                    # Estimate based on the amount of rows
                    row_count = self.count_documents()
                    lists = max(min(row_count // 1000, 1000), 10)
                except Exception:
                    lists = 100

            query = SQL(
                """
                CREATE INDEX IF NOT EXISTS {index_name}
                ON {schema_name}.{table_name} USING ivfflat ({embedding} {vector_ops})
                WITH (lists = {lists})
                """
            ).format(
                index_name=Identifier(self.index_name),
                schema_name=Identifier(self.schema_name),
                table_name=Identifier(self.table_name),
                embedding=Identifier(embedding_key),
                vector_ops=Identifier(vector_ops),
                lists=SQLLiteral(lists),
            )
        elif self.index_method == PGVectorIndexMethod.HNSW:
            query = SQL(
                """
                CREATE INDEX IF NOT EXISTS {index_name}
                ON {schema_name}.{table_name} USING hnsw ({embedding} {vector_ops})
                WITH (m = {m}, ef_construction = {ef_construction})
                """
            ).format(
                index_name=Identifier(self.index_name),
                schema_name=Identifier(self.schema_name),
                table_name=Identifier(self.table_name),
                embedding=Identifier(embedding_key),
                vector_ops=Identifier(vector_ops),
                m=SQLLiteral(self.hnsw_m),
                ef_construction=SQLLiteral(self.hnsw_ef_construction),
            )
        else:
            # EXACT search
            return

        with conn.cursor(row_factory=dict_row) as cur:
            self._execute_sql_query(query, cursor=cur)
            conn.commit()

    def _create_keyword_index(
        self,
        conn: psycopg.Connection,
        content_key: str | None = None,
    ) -> None:
        """
        Internal method to create the keyword index in the database (if it does not exist).

        Args:
            conn (psycopg.Connection): The connection to the database.
            content_key (str | None): The field used to store content in the storage. Defaults to None.
        """

        content_key = content_key or self.content_key

        create_keyword_index_query = SQL(
            """
            CREATE INDEX IF NOT EXISTS {index_name}
            ON {schema_name}.{table_name}
            USING gin(to_tsvector({language}, {content_key}));
            """
        ).format(
            index_name=Identifier(self.keyword_index_name),
            schema_name=Identifier(self.schema_name),
            table_name=Identifier(self.table_name),
            content_key=Identifier(content_key),
            language=SQLLiteral(self.language),
        )

        with conn.cursor(row_factory=dict_row) as cur:
            self._execute_sql_query(create_keyword_index_query, cursor=cur)
            conn.commit()

    def _drop_index(self, conn: psycopg.Connection) -> None:
        """
        Internal method to drop the index in the database (if it exists).
        Should only be called if the index method is either `ivfflat` or `hnsw`.

        Args:
            conn (psycopg.Connection): The connection to the database.

        Raises:
            ValueError: If the index method is not valid.
        """
        if self.index_method not in PGVectorIndexMethod:
            msg = f"Invalid index method: {self.index_method}"
            raise ValueError(msg)

        query = SQL(
            """
            DROP INDEX IF EXISTS {index_name};
            """
        ).format(
            index_name=Identifier(self.index_name),
        )

        with conn.cursor(row_factory=dict_row) as cur:
            self._execute_sql_query(query, cursor=cur)
            conn.commit()

    def _create_schema(self, conn: psycopg.Connection) -> None:
        """
        Internal method to create the schema in the database (if it does not exist).

        Args:
            conn (psycopg.Connection): The connection to the database.
        """

        query = SQL(
            """
            CREATE SCHEMA IF NOT EXISTS {schema_name};
            """
        ).format(
            schema_name=Identifier(self.schema_name),
        )

        with conn.cursor(row_factory=dict_row) as cur:
            self._execute_sql_query(query, cursor=cur)
            conn.commit()

    def _drop_schema(self, conn: psycopg.Connection) -> None:
        """
        Internal method to drop the schema in the database (if it exists).

        Args:
            conn (psycopg.Connection): The connection to the database.
        """

        query = SQL(
            """
            DROP SCHEMA IF EXISTS {schema_name} CASCADE;
            """
        ).format(
            schema_name=Identifier(self.schema_name),
        )

        with conn.cursor(row_factory=dict_row) as cur:
            self._execute_sql_query(query, cursor=cur)
            conn.commit()

    def delete_collection(self, collection_name: str | None = None) -> None:
        """
        Delete the collection in the database.

        Args:
            collection_name (str | None): Name of the collection to delete.
        """
        try:
            with self._get_connection() as conn:
                self._drop_tables(conn)
                if self.schema_name and self.schema_name != DEFAULT_SCHEMA_NAME:
                    self._drop_schema(conn)
        except Exception as e:
            logger.error(f"Failed to delete collection '{self.schema_name}.{self.table_name}': {e}")
            raise

    def count_documents(self) -> int:
        """
        Count the number of documents in the store.

        Returns:
            int: The number of documents in the store.
        """

        with self._get_connection() as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                query = SQL("SELECT COUNT(*) FROM {schema_name}.{table_name}").format(
                    schema_name=Identifier(self.schema_name), table_name=Identifier(self.table_name)
                )
                result = self._execute_sql_query(query, cursor=cur)
                row = result.fetchone()
                if row is None:
                    return 0
                return row[0] if isinstance(row, (tuple, list)) else row.get("count", 0)

    def write_documents(
        self, documents: list[Document], content_key: str | None = None, embedding_key: str | None = None
    ) -> int:
        """
        Write documents to the pgvector vector store.

        Args:
            documents (list[Document]): List of Document objects to write.
            content_key (str | None): The field used to store content in the storage. Defaults to None.
            embedding_key (str | None): The field used to store embeddings in the storage. Defaults to None.

        Returns:
            int: Number of documents successfully written.

        Raises:
            ValueError: If documents are not of type Document or have invalid embeddings.
        """
        if not documents:
            return 0

        if len(documents) > 0 and not isinstance(documents[0], Document):
            msg = "param 'documents' must contain a list of objects of type Document"
            raise ValueError(msg)

        content_key = content_key or self.content_key
        embedding_key = embedding_key or self.embedding_key

        document_ids = []
        for doc in documents:
            if doc.embedding is not None and len(doc.embedding) != self.dimension:
                raise ValueError(
                    f"Document {doc.id} embedding dimension {len(doc.embedding)} "
                    f"does not match configured dimension {self.dimension}"
                )
            document_ids.append(doc.id)

        # Prepare batch data
        batch_data = [(doc.id, doc.content, Jsonb(doc.metadata), doc.embedding) for doc in documents]

        query = SQL(
            """
            INSERT INTO {schema_name}.{table_name} (id, {content_key}, metadata, {embedding_key})
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (id) DO UPDATE SET
                {content_key} = EXCLUDED.{content_key},
                metadata = EXCLUDED.metadata,
                {embedding_key} = EXCLUDED.{embedding_key}
            """
        ).format(
            schema_name=Identifier(self.schema_name),
            table_name=Identifier(self.table_name),
            content_key=Identifier(content_key),
            embedding_key=Identifier(embedding_key),
        )

        with self._get_connection() as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                try:
                    cur.executemany(query, batch_data)
                    conn.commit()
                except psycopg_errors.OperationalError as e:
                    # Connection issues
                    sanitized_error = self._sanitize_error_message(str(e))
                    logger.warning(f"Database connection error: {sanitized_error}")
                    raise VectorStoreException("Database connection error") from e
                except psycopg_errors.UniqueViolation as e:
                    # Duplicate key
                    sanitized_error = self._sanitize_error_message(str(e))
                    logger.debug(f"Duplicate key violation: {sanitized_error}")
                    raise VectorStoreException("Document already exists") from e
                except psycopg_errors.DataError as e:
                    # Invalid data (e.g., wrong vector dimension, invalid JSON)
                    sanitized_error = self._sanitize_error_message(str(e))
                    logger.error(f"Data validation error: {sanitized_error}")
                    raise ValueError("Invalid document data provided") from e
                except psycopg_errors.InsufficientPrivilege as e:
                    # Permission error
                    sanitized_error = self._sanitize_error_message(str(e))
                    logger.error(f"Insufficient privileges: {sanitized_error}")
                    raise VectorStoreException("Insufficient database privileges") from e
                except Exception as e:
                    sanitized_error = self._sanitize_error_message(str(e))
                    logger.error(f"Unexpected database error during batch insert: {sanitized_error}", exc_info=True)
                    raise VectorStoreException("Unexpected database error") from e

        self._track_documents(document_ids)
        return len(documents)

    def delete_documents_by_filters(self, filters: dict[str, Any], top_k: int = 1000) -> None:
        """
        Delete documents from the pgvector vector store using filters.

        Args:
            filters (dict[str, Any]): Filters to select documents to delete.
            top_k (int): Unused parameter, kept for compatibility.
        """
        if filters:
            with self._get_connection() as conn:
                with conn.cursor(row_factory=dict_row) as cur:
                    sql_where_clause, params = self._prepare_filters(filters)
                    query = SQL("DELETE FROM {schema_name}.{table_name}").format(
                        schema_name=Identifier(self.schema_name),
                        table_name=Identifier(self.table_name),
                    )
                    query += sql_where_clause
                    self._execute_sql_query(query, params, cursor=cur)
                    conn.commit()
        else:
            logger.warning("No filters provided. No documents will be deleted.")

    def delete_documents_by_file_id(self, file_id: str) -> None:
        """
        Delete documents from the vector store based on the provided file ID.
        File ID should be located in the metadata of the document.

        Args:
            file_id (str): The file ID to filter by.
        """
        filters = create_pgvector_file_id_filter(file_id)
        self.delete_documents_by_filters(filters)

    def delete_documents_by_file_ids(self, file_ids: list[str], batch_size: int = 500) -> None:
        """
        Delete documents from the vector store based on the provided list of file IDs.
        File IDs should be located in the metadata of the documents.

        Args:
            file_ids (list[str]): The list of file IDs to filter by.
            batch_size (int): Maximum number of file IDs to process in a single batch. Defaults to 500.
        """
        if not file_ids:
            logger.warning("No file IDs provided. No documents will be deleted.")
            return

        if len(file_ids) > batch_size:
            for i in range(0, len(file_ids), batch_size):
                batch = file_ids[i : i + batch_size]
                filters = create_pgvector_file_ids_filter(batch)
                self.delete_documents_by_filters(filters)
                logger.debug(f"Deleted documents batch {i//batch_size + 1} with {len(batch)} file IDs")
        else:
            filters = create_pgvector_file_ids_filter(file_ids)
            self.delete_documents_by_filters(filters)

    def delete_documents(self, document_ids: list[str] | None = None, delete_all: bool = False) -> None:
        """
        Delete documents from the pgvector vector store.

        Args:
            document_ids (list[str]): List of document IDs to delete. Defaults to None.
            delete_all (bool): If True, delete all documents. Defaults to False.
        """
        if delete_all:
            with self._get_connection() as conn:
                with conn.cursor(row_factory=dict_row) as cur:
                    query = SQL("DELETE FROM {schema_name}.{table_name}").format(
                        schema_name=Identifier(self.schema_name), table_name=Identifier(self.table_name)
                    )
                    self._execute_sql_query(query, cursor=cur)
                    conn.commit()
        else:
            if not document_ids:
                logger.warning("No document IDs provided. No documents will be deleted.")
            else:
                with self._get_connection() as conn:
                    with conn.cursor(row_factory=dict_row) as cur:
                        query = SQL("DELETE FROM {schema_name}.{table_name} WHERE id = ANY(%s::text[])").format(
                            schema_name=Identifier(self.schema_name), table_name=Identifier(self.table_name)
                        )
                        self._execute_sql_query(query, (document_ids,), cursor=cur)
                        conn.commit()

    def list_documents(
        self,
        include_embeddings: bool = False,
        content_key: str | None = None,
        embedding_key: str | None = None,
        offset: int = 0,
        limit: int | None = None,
    ) -> list[Document]:
        """
        List documents in the pgvector vector store with optional pagination.

        Args:
            include_embeddings (bool): Whether to include embeddings in the results. Defaults to False.
            content_key (str): The field used to store content in the storage. Defaults to None.
            embedding_key (str): The field used to store embeddings in the storage. Defaults to None.
            offset (int): Number of documents to skip. Defaults to 0.
            limit (int | None): Maximum number of documents to return. If None, returns all documents.

        Returns:
            list[Document]: List of Document objects retrieved.
        """
        if offset < 0:
            msg = f"offset must be non-negative, got {offset}"
            raise ValueError(msg)
        if limit is not None and limit <= 0:
            msg = f"limit must be positive, got {limit}"
            raise ValueError(msg)

        content_key = content_key or self.content_key
        embedding_key = embedding_key or self.embedding_key

        select_columns = [SQL("id"), Identifier(content_key), SQL("metadata")]
        if include_embeddings:
            select_columns.append(Identifier(embedding_key))

        query = SQL("SELECT {} FROM {}.{} ORDER BY id").format(
            SQL(", ").join(select_columns),
            Identifier(self.schema_name),
            Identifier(self.table_name),
        )

        if limit is not None:
            query = query + SQL(" LIMIT {} OFFSET {}").format(SQLLiteral(limit), SQLLiteral(offset))
        elif offset > 0:
            query = query + SQL(" OFFSET {}").format(SQLLiteral(offset))

        with self._get_connection() as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                result = self._execute_sql_query(query, cursor=cur)
                records = result.fetchall()

                documents = self._convert_query_result_to_documents(records)
                return documents

    def _convert_query_result_to_documents(
        self,
        query_result: dict[str, Any],
        content_key: str | None = None,
        embedding_key: str | None = None,
    ) -> list[Document]:
        """
        Convert pgvector query results to Document objects.

        Args:
            query_result (dict[str, Any]): The query result from pgvector.
            content_key (str): The field used to store content in the storage. Defaults to None.
            embedding_key (str): The field used to store embeddings in the storage. Defaults to None.

        Returns:
            list[Document]: List of Document objects created from the query result.
        """
        documents = []

        content_key = content_key or self.content_key
        embedding_key = embedding_key or self.embedding_key

        for doc in query_result:
            document = Document(
                id=doc["id"],
                content=doc[content_key],
                metadata=doc["metadata"],
            )

            if doc.get(embedding_key) is not None:
                document.embedding = self._convert_pg_embedding_to_list(doc[embedding_key])
            else:
                document.embedding = None

            if doc.get("score") is not None:
                document.score = float(doc["score"])
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
        content_key: str | None = None,
        embedding_key: str | None = None,
    ) -> list[Document]:
        """
        Retrieve documents similar to the given query embedding.

        Args:
            query_embedding (list[float]): The query embedding vector.
            top_k (int): Maximum number of documents to retrieve. Defaults to 10.
            exclude_document_embeddings (bool): Whether to exclude embeddings in results. Defaults to True.
            filters (dict[str, Any] | None): Filters for the query. Defaults to None.
            content_key (str): The field used to store content in the storage. Defaults to None.
            embedding_key (str): The field used to store embeddings in the storage. Defaults to None.

        Returns:
            list[Document]: List of retrieved Document objects.

        Raises:
            ValueError: If query_embedding is empty or has incorrect dimension.
        """
        if not query_embedding:
            msg = "query_embedding must be a non-empty list"
            raise ValueError(msg)

        if len(query_embedding) != self.dimension:
            msg = f"query_embedding must be of dimension {self.dimension}"
            raise ValueError(msg)

        if top_k <= 0:
            msg = f"top_k must be positive, got {top_k}"
            raise ValueError(msg)

        vector_function = self.vector_function
        content_key = content_key or self.content_key
        embedding_key = embedding_key or self.embedding_key

        if vector_function not in PGVectorVectorFunction:
            msg = f"Invalid vector function: {vector_function}"
            raise ValueError(msg)

        query_embedding = self._convert_query_embedding_to_pgvector_format(query_embedding)

        # Generate the score calculation based on the vector function
        score_definition = VECTOR_FUNCTION_TO_SCORE_DEFINITION[vector_function].format(
            embedding_key=embedding_key, query_embedding=query_embedding
        )
        score_definition = f"{score_definition} AS score"

        # Do not select the embeddings if exclude_document_embeddings is True
        select_fields = f"id, {content_key}, metadata" if exclude_document_embeddings else "*"

        # Build the base SELECT query with score
        base_select = SQL("SELECT {fields}, {score} FROM {schema_name}.{table_name}").format(
            fields=SQL(select_fields),
            score=SQL(score_definition),
            schema_name=Identifier(self.schema_name),
            table_name=Identifier(self.table_name),
        )

        # Handle filters if they exist
        where_clause, params = self._prepare_filters(filters)

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
            # Set runtime params for the current index method
            self._set_pgvector_runtime_params(conn, top_k)

            with conn.cursor(row_factory=dict_row) as cur:
                result = self._execute_sql_query(sql_query, params, cursor=cur)
                records = result.fetchall()

                documents = self._convert_query_result_to_documents(records)
                return documents

    def _keyword_retrieval(
        self,
        query: str,
        top_k: int = 10,
        exclude_document_embeddings: bool = True,
        filters: dict[str, Any] | None = None,
        content_key: str | None = None,
        embedding_key: str | None = None,
    ) -> list[Document]:
        """
        Retrieve documents similar to the given query using keyword search.

        Args:
            query (str): The query string.
            top_k (int): Maximum number of documents to retrieve. Defaults to 10.
            exclude_document_embeddings (bool): Whether to exclude embeddings in results. Defaults to True.
            filters (dict[str, Any] | None): Filters for the query. Defaults to None.
            content_key (str): The field used to store content in the storage. Defaults to None.
            embedding_key (str): The field used to store embeddings in the storage. Defaults to None.

        Returns:
            list[Document]: List of retrieved Document objects.

        Raises:
            ValueError: If query is empty.
        """
        if not query:
            msg = "query must be provided for keyword retrieval"
            raise ValueError(msg)

        if top_k <= 0:
            msg = f"top_k must be positive, got {top_k}"
            raise ValueError(msg)

        content_key = content_key or self.content_key

        # Do not select the embeddings if exclude_document_embeddings is True
        select_fields = f"id, {content_key}, metadata" if exclude_document_embeddings else "*"

        # Build the base SELECT query with score
        base_select = SQL(
            """
            SELECT {fields}, ts_rank_cd(to_tsvector({language}, {content_key}), query) AS score
            FROM {schema_name}.{table_name}, plainto_tsquery({language}, %s) query
            WHERE to_tsvector({language}, {content_key}) @@ query
            """
        ).format(
            fields=SQL(select_fields),
            schema_name=Identifier(self.schema_name),
            table_name=Identifier(self.table_name),
            language=SQLLiteral(self.language),
            content_key=Identifier(content_key),
        )

        # Handle filters if they exist
        filter_clause, params = self._prepare_filters(filters)
        if filters:
            filter_str = filter_clause.as_string(None)
            if filter_str.strip().startswith("WHERE"):
                # Replace "WHERE" with "AND" for keyword search
                where_clause = SQL(filter_str.replace(" WHERE ", " AND ", 1))
            else:
                where_clause = filter_clause
        else:
            where_clause = SQL("")

        # Build the ORDER BY and LIMIT clause
        order_by = SQL(" ORDER BY score DESC LIMIT {limit}").format(limit=SQLLiteral(top_k))

        # Combine all parts into final query
        sql_query = base_select + where_clause + order_by

        with self._get_connection() as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                result = self._execute_sql_query(sql_query, (query, *params), cursor=cur)
                records = result.fetchall()

                documents = self._convert_query_result_to_documents(records)
                return documents

    def _hybrid_retrieval(
        self,
        query: str,
        query_embedding: list[float],
        top_k: int = 10,
        exclude_document_embeddings: bool = True,
        keyword_rank_constant: int = DEFAULT_KEYWORD_RANK_CONSTANT,
        top_k_subquery_multiplier: int = DEFAULT_TOP_K_SUBQUERY_MULTIPLIER,
        filters: dict[str, Any] | None = None,
        alpha: float = 0.5,
        content_key: str | None = None,
        embedding_key: str | None = None,
    ) -> list[Document]:
        """
        Retrieve documents similar to the given query using a hybrid approach.

        Args:
            query (str): The query string.
            query_embedding (list[float]): The query embedding vector.
            top_k (int): Maximum number of documents to retrieve. Defaults to 10.
            exclude_document_embeddings (bool): Whether to exclude embeddings in results. Defaults to True.
            keyword_rank_constant (int): Constant used in RRF (Reciprocal Rank Fusion) score calculation.
            top_k_subquery_multiplier (int): Multiplier for subquery limits to ensure enough candidates.
            filters (dict[str, Any] | None): Filters for the query. Defaults to None.
            alpha (float): Weight for semantic search (0-1). 0 = pure keyword, 1 = pure semantic, 0.5 = balanced.
            content_key (str): The field used to store content in the storage. Defaults to None.
            embedding_key (str): The field used to store embeddings in the storage. Defaults to None.

        Returns:
            list[Document]: List of retrieved Document objects.

        Raises:
            ValueError: If query is empty, query_embedding is None/empty, or parameters are invalid.
        """

        if not query:
            msg = "query must be provided for hybrid retrieval"
            raise ValueError(msg)

        if not query_embedding:
            msg = "query_embedding must be a non-empty list"
            raise ValueError(msg)

        if len(query_embedding) != self.dimension:
            msg = f"query_embedding must be of dimension {self.dimension}"
            raise ValueError(msg)

        if not 0 <= alpha <= 1:
            msg = f"alpha must be between 0 and 1, got {alpha}"
            raise ValueError(msg)

        if top_k <= 0:
            msg = f"top_k must be positive, got {top_k}"
            raise ValueError(msg)

        vector_function = self.vector_function
        content_key = content_key or self.content_key
        embedding_key = embedding_key or self.embedding_key

        # If alpha is 0, perform purely keyword search
        if alpha == 0:
            return self._keyword_retrieval(
                query,
                top_k=top_k,
                exclude_document_embeddings=exclude_document_embeddings,
                filters=filters,
                content_key=content_key,
                embedding_key=embedding_key,
            )
        # If alpha is 1, perform purely embedding search
        elif alpha == 1:
            return self._embedding_retrieval(
                query_embedding,
                top_k=top_k,
                exclude_document_embeddings=exclude_document_embeddings,
                filters=filters,
                content_key=content_key,
                embedding_key=embedding_key,
            )

        query_embedding = self._convert_query_embedding_to_pgvector_format(query_embedding)

        # Generate the score calculation based on the vector function
        score_definition = VECTOR_FUNCTION_TO_SCORE_DEFINITION[vector_function].format(
            embedding_key=embedding_key, query_embedding=query_embedding
        )

        # Determine sort order based on vector function type
        is_distance_metric = vector_function in ["l2_distance", "l1_distance"]
        sort_order = "ASC" if is_distance_metric else "DESC"

        # Optimize subquery limit based on alpha
        if alpha < 0.1 or alpha > 0.9:
            subquery_limit = max(top_k * 2, 20)
        else:
            subquery_limit = max(top_k * top_k_subquery_multiplier, 50)

        # Apply filters to avoid duplication
        base_where_clause, params = self._prepare_filters(filters)
        # Create separate where clauses for each subquery
        semantic_where_clause = base_where_clause  # semantic search has no WHERE
        keyword_where_clause = base_where_clause  # keyword search has WHERE
        if filters:
            where_str = base_where_clause.as_string(None)
            keyword_where_clause = SQL(where_str.replace(" WHERE ", " AND ", 1))

        embedding_select = SQL("") if exclude_document_embeddings else SQL(", ") + Identifier(embedding_key)
        semantic_search_query = SQL(
            """
            WITH semantic_search AS (
                SELECT id, {content_key}, metadata{embedding_select},
                       RANK() OVER (ORDER BY {score_definition} {sort_order}) AS rank
                FROM {schema_name}.{table_name}
                {where_clause}
                LIMIT {subquery_limit}
            ),
            """
        ).format(
            content_key=Identifier(content_key),
            embedding_select=embedding_select,
            score_definition=SQL(score_definition),
            schema_name=Identifier(self.schema_name),
            table_name=Identifier(self.table_name),
            where_clause=semantic_where_clause,
            subquery_limit=SQLLiteral(subquery_limit),
            sort_order=SQL(sort_order),
        )

        keyword_search_query = SQL(
            """
            keyword_search AS (
                SELECT id, {content_key}, metadata{embedding_select},
                       RANK() OVER (ORDER BY ts_rank_cd(to_tsvector({language}, {content_key}), query) DESC) AS rank
                FROM {schema_name}.{table_name}, plainto_tsquery({language}, {query}) query
                WHERE to_tsvector({language}, {content_key}) @@ query
                {where_clause}
                LIMIT {subquery_limit}
            )
            """
        ).format(
            content_key=Identifier(content_key),
            embedding_select=embedding_select,
            schema_name=Identifier(self.schema_name),
            table_name=Identifier(self.table_name),
            language=SQLLiteral(self.language),
            query=SQLLiteral(query),
            where_clause=keyword_where_clause,
            subquery_limit=SQLLiteral(subquery_limit),
        )

        # Build the final query to merge the results and sort them by score
        if exclude_document_embeddings:
            merge_query = SQL(
                """
                SELECT
                    COALESCE(semantic_search.id, keyword_search.id) AS id,
                    COALESCE(semantic_search.{content_key}, keyword_search.{content_key}) AS {content_key},
                    COALESCE(semantic_search.metadata, keyword_search.metadata) AS metadata,
                    COALESCE({alpha} / ({k} + semantic_search.rank), 0.0) +
                    COALESCE((1 - {alpha}) / ({k} + keyword_search.rank), 0.0) AS score
                FROM semantic_search
                FULL OUTER JOIN keyword_search ON semantic_search.id = keyword_search.id
                ORDER BY score DESC
                LIMIT {top_k}
                """
            ).format(
                content_key=Identifier(content_key),
                top_k=SQLLiteral(top_k),
                alpha=SQLLiteral(alpha),
                k=SQLLiteral(keyword_rank_constant),
            )
        else:
            merge_query = SQL(
                """
                SELECT
                    COALESCE(semantic_search.id, keyword_search.id) AS id,
                    COALESCE(semantic_search.{content_key}, keyword_search.{content_key}) AS {content_key},
                    COALESCE(semantic_search.metadata, keyword_search.metadata) AS metadata,
                    COALESCE(semantic_search.{embedding_key}, keyword_search.{embedding_key}) AS {embedding_key},
                    COALESCE({alpha} / ({k} + semantic_search.rank), 0.0) +
                    COALESCE((1 - {alpha}) / ({k} + keyword_search.rank), 0.0) AS score
                FROM semantic_search
                FULL OUTER JOIN keyword_search ON semantic_search.id = keyword_search.id
                ORDER BY score DESC
                LIMIT {top_k}
                """
            ).format(
                content_key=Identifier(content_key),
                embedding_key=Identifier(embedding_key),
                top_k=SQLLiteral(top_k),
                alpha=SQLLiteral(alpha),
                k=SQLLiteral(keyword_rank_constant),
            )

        sql_query = semantic_search_query + keyword_search_query + merge_query

        params = params + params

        with self._get_connection() as conn:
            # Set runtime params for the current index method
            self._set_pgvector_runtime_params(conn, top_k)

            with conn.cursor(row_factory=dict_row) as cur:
                result = self._execute_sql_query(sql_query, params, cursor=cur)
                records = result.fetchall()

                documents = self._convert_query_result_to_documents(records)
                return documents

    def close(self):
        """Close the connection to the PostgreSQL database."""
        if hasattr(self, "_conn") and self._conn is not None and not self._conn.closed:
            self._conn.close()

    def __del__(self):
        """Close the connection when the object is deleted."""
        self.close()
