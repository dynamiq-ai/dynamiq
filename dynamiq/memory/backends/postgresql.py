import json
import time
import uuid
from typing import Any, ClassVar

import psycopg
from pydantic import ConfigDict, Field, PrivateAttr

from dynamiq.connections import PostgreSQL as PostgreSQLConnection
from dynamiq.memory.backends.base import MemoryBackend
from dynamiq.prompts import Message, MessageRole
from dynamiq.utils.logger import logger


class PostgresMemoryError(Exception):
    """Base exception class for PostgreSQL Memory Backend errors."""

    pass


class PostgreSQL(MemoryBackend):
    """
    PostgreSQL implementation of the memory storage backend.

    Stores messages in a specified PostgreSQL table, using a JSONB column
    for metadata to allow flexible filtering.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str = "PostgreSQL"
    connection: PostgreSQLConnection = Field(default_factory=PostgreSQLConnection)
    table_name: str = Field(default="dynamiq_memory_messages")
    create_table_if_not_exists: bool = Field(default=True)

    message_id_col: str = Field(default="message_id")
    role_col: str = Field(default="role")
    content_col: str = Field(default="content")
    metadata_col: str = Field(default="metadata")
    timestamp_col: str = Field(default="timestamp")

    _conn: psycopg.Connection | None = PrivateAttr(default=None)

    _CREATE_TABLE_SQL: ClassVar[
        str
    ] = """
    CREATE TABLE IF NOT EXISTS "{table_name}" (
        "{message_id_col}" UUID PRIMARY KEY,
        "{role_col}" TEXT NOT NULL,
        "{content_col}" TEXT,
        "{metadata_col}" JSONB,
        "{timestamp_col}" DOUBLE PRECISION NOT NULL
    );
    """
    _CREATE_TIMESTAMP_INDEX_SQL: ClassVar[
        str
    ] = """
    CREATE INDEX IF NOT EXISTS idx_{table_short_name}_timestamp ON "{table_name}" ("{timestamp_col}");
    """
    _CREATE_METADATA_INDEX_SQL: ClassVar[
        str
    ] = """
    CREATE INDEX IF NOT EXISTS idx_{table_short_name}_metadata_gin ON "{table_name}" USING GIN ("{metadata_col}");
    """
    _INSERT_MESSAGE_SQL: ClassVar[
        str
    ] = """
    INSERT INTO "{table_name}" ("{message_id_col}", "{role_col}", "{content_col}", "{metadata_col}", "{timestamp_col}")
    VALUES (%s, %s, %s, %s, %s);
    """
    _SELECT_ALL_SQL: ClassVar[
        str
    ] = """
    SELECT "{message_id_col}", "{role_col}", "{content_col}", "{metadata_col}", "{timestamp_col}"
    FROM "{table_name}"
    ORDER BY "{timestamp_col}" ASC
    """
    _SELECT_SEARCH_SQL: ClassVar[
        str
    ] = """
    SELECT "{message_id_col}", "{role_col}", "{content_col}", "{metadata_col}", "{timestamp_col}"
    FROM "{table_name}"
    """
    _IS_EMPTY_SQL: ClassVar[
        str
    ] = """
    SELECT EXISTS (SELECT 1 FROM "{table_name}" LIMIT 1);
    """
    _CLEAR_SQL: ClassVar[
        str
    ] = """
    TRUNCATE TABLE "{table_name}";
    """

    @property
    def to_dict_exclude_params(self) -> dict[str, bool]:
        """Define parameters to exclude during serialization."""
        return super().to_dict_exclude_params | {"_conn": True}

    def to_dict(self, include_secure_params: bool = False, **kwargs) -> dict[str, Any]:
        """Converts the instance to a dictionary."""
        exclude = kwargs.pop("exclude", self.to_dict_exclude_params.copy())
        data = self.model_dump(exclude=exclude, **kwargs)
        data["connection"] = self.connection.to_dict(include_secure_params=include_secure_params)
        if "type" not in data:
            data["type"] = self.type
        return data

    def model_post_init(self, __context: Any) -> None:
        """Initialize the PostgreSQL connection and ensure table exists."""
        try:
            self._conn = self.connection.connect()
            if self.create_table_if_not_exists:
                self._create_table_and_indices()
            logger.debug(f"PostgreSQL backend connected to table '{self.table_name}'.")
        except psycopg.Error as e:
            logger.error(f"Failed to initialize PostgreSQL connection or table '{self.table_name}': {e}")
            raise PostgresMemoryError(f"Failed to initialize PostgreSQL connection or table: {e}") from e
        except Exception as e:
            logger.error(f"Unexpected error initializing PostgreSQL backend: {e}")
            raise PostgresMemoryError(f"Unexpected error initializing PostgreSQL backend: {e}") from e

    def _execute_sql(self, sql: str, params: tuple | list | None = None, fetch: str | None = None):
        """Helper to execute SQL, handling potential connection issues."""
        if self._conn is None or self._conn.closed:
            logger.warning("PostgreSQL connection lost or not initialized. Attempting to reconnect.")
            try:
                self._conn = self.connection.connect()
            except Exception as e:
                raise PostgresMemoryError(f"Failed to re-establish PostgreSQL connection: {e}") from e

        try:
            with self._conn.cursor() as cur:
                cur.execute(sql, params)
                if fetch == "one":
                    return cur.fetchone()
                elif fetch == "all":
                    return cur.fetchall()
                return None
        except psycopg.Error as e:
            logger.error(f"PostgreSQL error executing SQL: {e}\nSQL: {sql}\nParams: {params}")
            raise PostgresMemoryError(f"PostgreSQL error: {e}") from e

    def _create_table_and_indices(self) -> None:
        """Creates the table and necessary indices if they don't exist."""
        logger.debug(f"Ensuring table '{self.table_name}' and indices exist...")
        table_sql = self._CREATE_TABLE_SQL.format(
            table_name=self.table_name,
            message_id_col=self.message_id_col,
            role_col=self.role_col,
            content_col=self.content_col,
            metadata_col=self.metadata_col,
            timestamp_col=self.timestamp_col,
        )
        self._execute_sql(table_sql)

        table_short_name = "".join(filter(str.isalnum, self.table_name))[:10]

        ts_index_sql = self._CREATE_TIMESTAMP_INDEX_SQL.format(
            table_name=self.table_name, timestamp_col=self.timestamp_col, table_short_name=table_short_name
        )
        self._execute_sql(ts_index_sql)

        meta_index_sql = self._CREATE_METADATA_INDEX_SQL.format(
            table_name=self.table_name, metadata_col=self.metadata_col, table_short_name=table_short_name
        )
        self._execute_sql(meta_index_sql)
        logger.debug(f"Table '{self.table_name}' and indices checked/created.")

    def _row_to_message(self, row: dict) -> Message:
        """Converts a database row (dict) to a Message object."""
        metadata = row.get(self.metadata_col) or {}
        if "timestamp" not in metadata:
            metadata["timestamp"] = row.get(self.timestamp_col)
        if "message_id" not in metadata:
            metadata["message_id"] = row.get(self.message_id_col)

        return Message(
            role=MessageRole(row.get(self.role_col, MessageRole.USER.value)),
            content=row.get(self.content_col, ""),
            metadata=metadata,
        )

    def add(self, message: Message) -> None:
        """Adds a message to the PostgreSQL table."""
        try:
            message_id = message.metadata.get("message_id", uuid.uuid4())
            timestamp = float(message.metadata.get("timestamp", time.time()))
            metadata_to_store = message.metadata or {}

            sql = self._INSERT_MESSAGE_SQL.format(
                table_name=self.table_name,
                message_id_col=self.message_id_col,
                role_col=self.role_col,
                content_col=self.content_col,
                metadata_col=self.metadata_col,
                timestamp_col=self.timestamp_col,
            )
            params = (
                message_id,
                message.role.value,
                message.content,
                json.dumps(metadata_to_store),
                timestamp,
            )
            self._execute_sql(sql, params)
            logger.debug(f"PostgreSQL Memory ({self.table_name}): Added message {message_id}")

        except (TypeError, ValueError) as e:
            logger.error(f"Error preparing message data for PostgreSQL: {e}")
            raise PostgresMemoryError(f"Error preparing message data: {e}") from e

    def get_all(self, limit: int | None = None) -> list[Message]:
        """Retrieves messages from PostgreSQL, sorted chronologically."""
        sql = self._SELECT_ALL_SQL.format(
            table_name=self.table_name,
            message_id_col=self.message_id_col,
            role_col=self.role_col,
            content_col=self.content_col,
            metadata_col=self.metadata_col,
            timestamp_col=self.timestamp_col,
        )
        params = []
        if limit is not None and limit > 0:
            sql += " LIMIT %s"
            params.append(limit)

        rows = self._execute_sql(sql, params, fetch="all")
        messages = [self._row_to_message(row) for row in rows]
        logger.debug(f"PostgreSQL Memory ({self.table_name}): Retrieved {len(messages)} messages.")
        return messages

    def _build_where_clause(self, query: str | None, filters: dict | None) -> tuple[str, list]:
        """Builds the WHERE clause and parameters for search."""
        where_clauses = []
        params = []

        if filters:
            for key, value in filters.items():
                where_clauses.append(f'"{self.metadata_col}"->>%s = %s')
                params.extend([key, str(value)])  # Compare as text
        if query:
            where_clauses.append(f'"{self.content_col}" ILIKE %s')
            params.append(f"%{query}%")
        if not where_clauses:
            return "", []

        where_clause_str = "WHERE " + " AND ".join(where_clauses)
        return where_clause_str, params

    def search(
        self, query: str | None = None, filters: dict[str, Any] | None = None, limit: int | None = None
    ) -> list[Message]:
        """Searches messages using ILIKE for query and JSONB operators for filters."""

        where_clause, params = self._build_where_clause(query, filters)

        sql = self._SELECT_SEARCH_SQL.format(
            table_name=self.table_name,
            message_id_col=self.message_id_col,
            role_col=self.role_col,
            content_col=self.content_col,
            metadata_col=self.metadata_col,
            timestamp_col=self.timestamp_col,
        )

        if where_clause:
            sql += f" {where_clause}"

        sql += f' ORDER BY "{self.timestamp_col}" DESC'

        if limit is not None and limit > 0:
            sql += " LIMIT %s"
            params.append(limit)

        rows = self._execute_sql(sql, params, fetch="all")
        messages = [self._row_to_message(row) for row in rows]

        logger.debug(
            f"PostgreSQL Memory ({self.table_name}): Found {len(messages)} search results "
            f"(Query: {'Yes' if query else 'No'}, Filters: {'Yes' if filters else 'No'}, Limit: {limit})"
        )
        return messages

    def is_empty(self) -> bool:
        """Checks if the PostgreSQL table is empty."""
        sql = self._IS_EMPTY_SQL.format(table_name=self.table_name)
        result = self._execute_sql(sql, fetch="one")
        return not (result and result.get("exists", False))

    def clear(self) -> None:
        """Clears the PostgreSQL table using TRUNCATE."""
        logger.warning(f"Clearing all messages from PostgreSQL table '{self.table_name}' using TRUNCATE.")
        sql = self._CLEAR_SQL.format(table_name=self.table_name)
        self._execute_sql(sql)
        logger.info(f"PostgreSQL Memory ({self.table_name}): Cleared table.")

    def __del__(self):
        """Attempt to close the connection when the object is deleted."""
        if self._conn and not self._conn.closed:
            try:
                self._conn.close()
                logger.debug("PostgreSQL connection closed.")
            except Exception as e:
                logger.error(f"Error closing PostgreSQL connection: {e}")
