import json
import re
import sqlite3
import uuid

from dynamiq.memory.backend.base import MemoryBackend
from dynamiq.prompts import Message


class SQLiteError(Exception):
    """Base exception class for SQLite-related errors in the memory backend."""

    pass


class SQLite(MemoryBackend):
    """SQLite implementation of the memory storage backend."""

    name = "SQLite"

    def __init__(self, db_path: str = "conversations.db", index_name: str = "conversations"):
        """Initializes the SQLite memory storage."""
        self.db_path = db_path
        self.index_name = index_name

        try:
            self._validate_table_name(create_if_not_exists=True)
        except Exception as e:
            raise SQLiteError(f"Error initializing SQLite backend: {e}") from e

    def _validate_table_name(self, create_if_not_exists: bool = False):
        """Validates the table name to prevent SQL injection and optionally creates it."""
        if not re.match(r"^[A-Za-z0-9_]+$", self.index_name):
            raise SQLiteError(f"Invalid table name: '{self.index_name}'")

        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (self.index_name,)
                )  # nosec B608
                result = cursor.fetchone()

                if result is None:
                    if create_if_not_exists:
                        self._create_table()
                    else:
                        raise SQLiteError(f"Table '{self.index_name}' does not exist in the database.")

        except sqlite3.Error as e:
            raise SQLiteError(f"Error validating or creating table: {e}") from e

    def _create_table(self):
        """Creates the messages table."""
        query = f"""
        CREATE TABLE IF NOT EXISTS {self.index_name} (
            id TEXT PRIMARY KEY,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            metadata TEXT,
            timestamp REAL

        )
        """  # nosec B608
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(query)  # nosec B608
                conn.commit()
        except sqlite3.Error as e:
            raise SQLiteError(f"Error creating table: {e}") from e

    def add(self, message: Message):
        """Stores a message in the SQLite database."""
        try:
            self._validate_table_name()  # Ensure table exists
            query = f"""
            INSERT INTO {self.index_name} (id, role, content, metadata, timestamp)
            VALUES (?, ?, ?, ?, ?)
            """  # nosec B608
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
                )  # nosec B608
                conn.commit()

        except sqlite3.Error as e:
            raise SQLiteError(f"Error adding message to database: {e}") from e

    def get_all(self) -> list[Message]:
        """Retrieves all messages from the SQLite database."""
        try:
            self._validate_table_name()  # Ensure table exists
            query = f"SELECT id, role, content, metadata, timestamp FROM {self.index_name} ORDER BY timestamp ASC"  # nosec B608 # noqa: E501
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(query)  # nosec B608
                rows = cursor.fetchall()
            return [
                Message(role=row[1], content=row[2], metadata=json.loads(row[3] or "{}"), timestamp=row[4])
                for row in rows
            ]

        except sqlite3.Error as e:
            raise SQLiteError(f"Error retrieving messages from database: {e}") from e

    def is_empty(self) -> bool:
        """Checks if the SQLite database is empty."""
        try:
            self._validate_table_name()
            query = f"SELECT COUNT(*) FROM {self.index_name}"  # nosec B608
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(query)  # nosec B608
                count = cursor.fetchone()[0]
            return count == 0

        except sqlite3.Error as e:
            raise SQLiteError(f"Error checking if database is empty: {e}") from e

    def clear(self):
        """Clears the SQLite database by deleting all rows in the table."""
        try:
            self._validate_table_name()
            query = f"DELETE FROM {self.index_name}"  # nosec B608
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(query)  # nosec B608
                conn.commit()
        except sqlite3.Error as e:
            raise SQLiteError(f"Error clearing database: {e}") from e

    def search(self, query: str = None, search_limit: int = None, filters: dict = None) -> list[Message]:
        """Searches for messages in SQLite based on the query and/or filters."""
        search_limit = search_limit or self.config.search_limit  # Use default if not provided
        try:
            self._validate_table_name()
            where_clauses = []
            params = []

            if query:
                where_clauses.append("content LIKE ?")
                params.append(f"%{query}%")

            if filters:
                for key, value in filters.items():
                    if isinstance(value, list):
                        where_clauses.append(f"json_extract(metadata, '$.{key}') LIKE ?")
                        params.append(f"%{json.dumps(value)}%")
                    else:
                        where_clauses.append(f"json_extract(metadata, '$.{key}') = ?")
                        params.append(value)

            query_str = f"SELECT id, role, content, metadata FROM {self.index_name}"  # nosec B608
            if where_clauses:
                query_str += f" WHERE {' AND '.join(where_clauses)}"
            query_str += " ORDER BY id DESC LIMIT ?"
            params.append(search_limit)

            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(query_str, params)
                rows = cursor.fetchall()
            return [Message(role=row[1], content=row[2], metadata=json.loads(row[3] or "{}")) for row in rows]

        except sqlite3.Error as e:
            raise SQLiteError(f"Error searching in database: {e}") from e
