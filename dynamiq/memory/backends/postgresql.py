import json
import time
import uuid
from enum import Enum
from typing import Any

import psycopg
from psycopg.sql import SQL, Identifier
from pydantic import ConfigDict, Field, PrivateAttr

from dynamiq.connections import PostgreSQL as PostgreSQLConnection
from dynamiq.memory.backends.base import MemoryBackend
from dynamiq.prompts import Message, MessageRole
from dynamiq.utils.logger import logger


class FetchMode(str, Enum):
    """Enum for SQL fetch modes."""

    ONE = "one"
    ALL = "all"


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
    table_name: str = Field(default="conversations")
    create_if_not_exist: bool = Field(default=True)

    message_id_col: str = Field(default="message_id")
    role_col: str = Field(default="role")
    content_col: str = Field(default="content")
    metadata_col: str = Field(default="metadata")
    timestamp_col: str = Field(default="timestamp")

    _conn: psycopg.Connection | None = PrivateAttr(default=None)
    _is_closed: bool = PrivateAttr(default=False)

    @property
    def to_dict_exclude_params(self) -> dict[str, bool]:
        """Define parameters to exclude during serialization."""
        return super().to_dict_exclude_params | {"_conn": True, "_is_closed": True}

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
            self._is_closed = False
            if self.create_if_not_exist:
                self._create_table_and_indices()
            logger.debug(f"PostgreSQL backend connected to table '{self.table_name}'.")
        except psycopg.Error as e:
            logger.error(f"Failed to initialize PostgreSQL connection or table '{self.table_name}': {e}")
            raise PostgresMemoryError(f"Failed to initialize PostgreSQL connection or table: {e}") from e
        except Exception as e:
            logger.error(f"Unexpected error initializing PostgreSQL backend: {e}")
            raise PostgresMemoryError(f"Unexpected error initializing PostgreSQL backend: {e}") from e

    def close(self) -> None:
        """
        Explicitly close the PostgreSQL connection.

        This is the recommended way to clean up resources when you're done
        with the memory backend. Safe to call multiple times.
        """
        if self._conn and not self._conn.closed:
            try:
                self._conn.close()
                logger.debug("PostgreSQL connection closed explicitly.")
            except Exception as e:
                logger.error(f"Error closing PostgreSQL connection: {e}")
            finally:
                self._is_closed = True
        else:
            self._is_closed = True

    def __enter__(self):
        """Context manager entry point."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit point - automatically close connection."""
        self.close()

    def _check_connection_state(self) -> None:
        """Check if the backend has been explicitly closed."""
        if self._is_closed:
            raise PostgresMemoryError("PostgreSQL backend has been closed. Create a new instance to reconnect.")

    def _execute_sql(self, sql_query: SQL | str, params: tuple | list | None = None, fetch: FetchMode | None = None):
        """Helper to execute SQL, handling potential connection issues."""
        self._check_connection_state()

        if self._conn is None or self._conn.closed:
            logger.warning("PostgreSQL connection lost or not initialized. Attempting to reconnect.")
            try:
                self._conn = self.connection.connect()
            except Exception as e:
                raise PostgresMemoryError(f"Failed to re-establish PostgreSQL connection: {e}") from e

        try:
            with self._conn.cursor() as cur:
                cur.execute(sql_query, params)
                if fetch == FetchMode.ONE:
                    return cur.fetchone()
                elif fetch == FetchMode.ALL:
                    return cur.fetchall()
                return None
        except psycopg.Error as e:
            sql_str = sql_query.as_string(cur) if isinstance(sql_query, SQL) else str(sql_query)
            logger.error(f"PostgreSQL error executing SQL: {e}\nSQL: {sql_str}\nParams: {params}")
            raise PostgresMemoryError(f"PostgreSQL error: {e}") from e

    def _create_table_and_indices(self) -> None:
        """Creates the table and necessary indices if they don't exist."""
        logger.debug(f"Ensuring table '{self.table_name}' and indices exist...")

        table_sql = SQL(
            """
            CREATE TABLE IF NOT EXISTS {table_name} (
                {message_id_col} UUID PRIMARY KEY,
                {role_col} TEXT NOT NULL,
                {content_col} TEXT,
                {metadata_col} JSONB,
                {timestamp_col} DOUBLE PRECISION NOT NULL
            );
        """
        ).format(
            table_name=Identifier(self.table_name),
            message_id_col=Identifier(self.message_id_col),
            role_col=Identifier(self.role_col),
            content_col=Identifier(self.content_col),
            metadata_col=Identifier(self.metadata_col),
            timestamp_col=Identifier(self.timestamp_col),
        )
        self._execute_sql(table_sql)

        table_short_name = "".join(filter(str.isalnum, self.table_name))[:10]

        ts_index_sql = SQL(
            """
            CREATE INDEX IF NOT EXISTS {index_name} ON {table_name} ({timestamp_col});
        """
        ).format(
            index_name=Identifier(f"idx_{table_short_name}_timestamp"),
            table_name=Identifier(self.table_name),
            timestamp_col=Identifier(self.timestamp_col),
        )
        self._execute_sql(ts_index_sql)

        meta_index_sql = SQL(
            """
            CREATE INDEX IF NOT EXISTS {index_name} ON {table_name} USING GIN ({metadata_col});
        """
        ).format(
            index_name=Identifier(f"idx_{table_short_name}_metadata_gin"),
            table_name=Identifier(self.table_name),
            metadata_col=Identifier(self.metadata_col),
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

            sql = SQL(
                """
                INSERT INTO {table_name} ({message_id_col}, {role_col}, {content_col}, {metadata_col}, {timestamp_col})
                VALUES (%s, %s, %s, %s, %s);
            """
            ).format(
                table_name=Identifier(self.table_name),
                message_id_col=Identifier(self.message_id_col),
                role_col=Identifier(self.role_col),
                content_col=Identifier(self.content_col),
                metadata_col=Identifier(self.metadata_col),
                timestamp_col=Identifier(self.timestamp_col),
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
        sql = SQL(
            """
            SELECT {message_id_col}, {role_col}, {content_col}, {metadata_col}, {timestamp_col}
            FROM {table_name}
            ORDER BY {timestamp_col} ASC
        """
        ).format(
            table_name=Identifier(self.table_name),
            message_id_col=Identifier(self.message_id_col),
            role_col=Identifier(self.role_col),
            content_col=Identifier(self.content_col),
            metadata_col=Identifier(self.metadata_col),
            timestamp_col=Identifier(self.timestamp_col),
        )

        params = []
        if limit is not None and limit > 0:
            sql = sql + SQL(" LIMIT %s")
            params.append(limit)

        rows = self._execute_sql(sql, params, fetch=FetchMode.ALL)
        messages = [self._row_to_message(row) for row in rows]
        logger.debug(f"PostgreSQL Memory ({self.table_name}): Retrieved {len(messages)} messages.")
        return messages

    def _build_where_clause(self, query: str | None, filters: dict | None) -> tuple[SQL, list]:
        """Builds the WHERE clause and parameters for search."""
        where_clauses = []
        params = []

        if filters:
            for key, value in filters.items():
                where_clauses.append(SQL("{metadata_col}->>%s = %s").format(metadata_col=Identifier(self.metadata_col)))
                params.extend([key, str(value)])  # Compare as text

        if query:
            where_clauses.append(SQL("{content_col} ILIKE %s").format(content_col=Identifier(self.content_col)))
            params.append(f"%{query}%")

        if not where_clauses:
            return SQL(""), []

        where_clause_sql = SQL("WHERE ") + SQL(" AND ").join(where_clauses)
        return where_clause_sql, params

    def search(
        self, query: str | None = None, filters: dict[str, Any] | None = None, limit: int | None = None
    ) -> list[Message]:
        """Searches messages using ILIKE for query and JSONB operators for filters."""

        where_clause, params = self._build_where_clause(query, filters)

        sql = SQL(
            """
            SELECT {message_id_col}, {role_col}, {content_col}, {metadata_col}, {timestamp_col}
            FROM {table_name}
        """
        ).format(
            table_name=Identifier(self.table_name),
            message_id_col=Identifier(self.message_id_col),
            role_col=Identifier(self.role_col),
            content_col=Identifier(self.content_col),
            metadata_col=Identifier(self.metadata_col),
            timestamp_col=Identifier(self.timestamp_col),
        )

        if where_clause:
            sql = sql + SQL(" ") + where_clause

        sql = sql + SQL(" ORDER BY {timestamp_col} DESC").format(timestamp_col=Identifier(self.timestamp_col))

        if limit is not None and limit > 0:
            sql = sql + SQL(" LIMIT %s")
            params.append(limit)

        rows = self._execute_sql(sql, params, fetch=FetchMode.ALL)
        messages = [self._row_to_message(row) for row in rows]

        logger.debug(
            f"PostgreSQL Memory ({self.table_name}): Found {len(messages)} search results "
            f"(Query: {'Yes' if query else 'No'}, Filters: {'Yes' if filters else 'No'}, Limit: {limit})"
        )
        return messages

    def is_empty(self) -> bool:
        """Checks if the PostgreSQL table is empty."""
        sql = SQL("SELECT EXISTS (SELECT 1 FROM {table_name} LIMIT 1);").format(table_name=Identifier(self.table_name))
        result = self._execute_sql(sql, fetch=FetchMode.ONE)
        return not (result and result.get("exists", False))

    def clear(self) -> None:
        """Clears the PostgreSQL table using TRUNCATE."""
        logger.warning(f"Clearing all messages from PostgreSQL table '{self.table_name}' using TRUNCATE.")
        sql = SQL("TRUNCATE TABLE {table_name};").format(table_name=Identifier(self.table_name))
        self._execute_sql(sql)
        logger.info(f"PostgreSQL Memory ({self.table_name}): Cleared table.")
