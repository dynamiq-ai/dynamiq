import json
import re
import sqlite3
import uuid
from typing import ClassVar

from pydantic import ConfigDict, Field
from typing_extensions import Annotated

from dynamiq.memory.backends.base import MemoryBackend
from dynamiq.prompts import Message


class SQLiteError(Exception):
    """Base exception class for SQLite-related errors in the memory backend."""

    pass


class SQLite(MemoryBackend):
    """SQLite implementation of the memory storage backend."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str = "SQLite"
    db_path: Annotated[str, Field(default="conversations.db")]
    index_name: Annotated[str, Field(default="conversations")]

    # SQL Query Constants
    CREATE_TABLE_QUERY: ClassVar[
        str
    ] = """
        CREATE TABLE IF NOT EXISTS {index_name} (
            id TEXT PRIMARY KEY,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            metadata TEXT,
            timestamp REAL
        )
    """

    VALIDATE_TABLE_QUERY: ClassVar[str] = "SELECT name FROM sqlite_master WHERE type='table' AND name=?"
    INSERT_MESSAGE_QUERY: ClassVar[
        str
    ] = """
        INSERT INTO {index_name} (id, role, content, metadata, timestamp)
        VALUES (?, ?, ?, ?, ?)
    """
    SELECT_ALL_MESSAGES_QUERY: ClassVar[
        str
    ] = """
        SELECT id, role, content, metadata, timestamp
        FROM {index_name}
        ORDER BY timestamp ASC
    """
    CHECK_IF_EMPTY_QUERY: ClassVar[str] = "SELECT COUNT(*) FROM {index_name}"
    CLEAR_TABLE_QUERY: ClassVar[str] = "DELETE FROM {index_name}"
    SEARCH_MESSAGES_QUERY: ClassVar[
        str
    ] = """
        SELECT id, role, content, metadata
        FROM {index_name}
    """

    @property
    def to_dict_exclude_params(self):
        """Define parameters to exclude during serialization."""
        return super().to_dict_exclude_params | {
            "CREATE_TABLE_QUERY": True,
            "VALIDATE_TABLE_QUERY": True,
            "INSERT_MESSAGE_QUERY": True,
            "SELECT_ALL_MESSAGES_QUERY": True,
            "CHECK_IF_EMPTY_QUERY": True,
            "CLEAR_TABLE_QUERY": True,
            "SEARCH_MESSAGES_QUERY": True,
        }

    def to_dict(self, include_secure_params: bool = False, **kwargs) -> dict:
        """Converts the instance to a dictionary.

        Args:
            include_secure_params (bool): Whether to include secure parameters
            **kwargs: Additional arguments

        Returns:
            dict: Dictionary representation of the instance
        """
        kwargs.pop("include_secure_params", None)
        data = super().to_dict(**kwargs)

        if not include_secure_params:
            data.pop("db_path", None)

        return data

    def model_post_init(self, __context) -> None:
        """Initialize the SQLite database after model initialization."""
        try:
            self._validate_table_name(create_if_not_exists=True)
        except Exception as e:
            raise SQLiteError(f"Error initializing SQLite backend: {e}") from e

    def _validate_table_name(self, create_if_not_exists: bool = False) -> None:
        """Validates the table name to prevent SQL injection and optionally creates it."""
        if not re.match(r"^[A-Za-z0-9_]+$", self.index_name):
            raise SQLiteError(f"Invalid table name: '{self.index_name}'")

        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(self.VALIDATE_TABLE_QUERY, (self.index_name,))
                result = cursor.fetchone()

                if result is None:
                    if create_if_not_exists:
                        self._create_table()
                    else:
                        raise SQLiteError(f"Table '{self.index_name}' does not exist in the database.")

        except sqlite3.Error as e:
            raise SQLiteError(f"Error validating or creating table: {e}") from e

    def _create_table(self) -> None:
        """Creates the messages table."""
        query = self.CREATE_TABLE_QUERY.format(index_name=self.index_name)
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(query)
                conn.commit()
        except sqlite3.Error as e:
            raise SQLiteError(f"Error creating table: {e}") from e

    def add(self, message: Message) -> None:
        """Stores a message in the SQLite database."""
        try:
            query = self.INSERT_MESSAGE_QUERY.format(index_name=self.index_name)
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                message_id = str(uuid.uuid4())
                cursor.execute(
                    query,
                    (
                        message_id,
                        message.role.value,
                        message.content,
                        json.dumps(message.metadata),
                        message.metadata.get("timestamp", 0),
                    ),
                )
                conn.commit()

        except sqlite3.Error as e:
            raise SQLiteError(f"Error adding message to database: {e}") from e

    def get_all(self) -> list[Message]:
        """Retrieves all messages from the SQLite database."""
        try:
            query = self.SELECT_ALL_MESSAGES_QUERY.format(index_name=self.index_name)
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(query)
                rows = cursor.fetchall()
            return [Message(role=row[1], content=row[2], metadata=json.loads(row[3] or "{}")) for row in rows]

        except sqlite3.Error as e:
            raise SQLiteError(f"Error retrieving messages from database: {e}") from e

    def is_empty(self) -> bool:
        """Checks if the SQLite database is empty."""
        try:
            query = self.CHECK_IF_EMPTY_QUERY.format(index_name=self.index_name)
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(query)
                count = cursor.fetchone()[0]
            return count == 0

        except sqlite3.Error as e:
            raise SQLiteError(f"Error checking if database is empty: {e}") from e

    def clear(self) -> None:
        """Clears the SQLite database by deleting all rows in the table."""
        try:
            query = self.CLEAR_TABLE_QUERY.format(index_name=self.index_name)
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(query)
                conn.commit()
        except sqlite3.Error as e:
            raise SQLiteError(f"Error clearing database: {e}") from e

    def search(self, query: str | None = None, limit: int = 10, filters: dict | None = None) -> list[Message]:
        """Searches for messages in SQLite based on the query and/or filters."""
        try:
            where_clauses = []
            params = []

            if query:
                where_clauses.append("content LIKE ?")
                params.append(f"%{query}%")

            if filters:
                for key, value in filters.items():
                    if isinstance(value, list):
                        placeholders = ",".join("?" for _ in value)
                        where_clauses.append(f"json_extract(metadata, '$.{key}') IN ({placeholders})")
                        params.extend(value)
                    else:
                        if isinstance(value, str) and "%" in value:
                            where_clauses.append(f"json_extract(metadata, '$.{key}') LIKE ?")
                            params.append(value)
                        else:
                            where_clauses.append(f"json_extract(metadata, '$.{key}') = ?")
                            params.append(value)

            query_str = self.SEARCH_MESSAGES_QUERY.format(index_name=self.index_name)
            if where_clauses:
                query_str += f" WHERE {' AND '.join(where_clauses)}"
            query_str += " ORDER BY id DESC LIMIT ?"
            params.append(limit)

            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(query_str, params)
                rows = cursor.fetchall()

            return [Message(role=row[1], content=row[2], metadata=json.loads(row[3] or "{}")) for row in rows]

        except sqlite3.Error as e:
            raise SQLiteError(f"Error searching in database: {e}") from e
