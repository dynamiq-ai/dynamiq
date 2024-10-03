import json
import re
import sqlite3

from dynamiq.memory.backend.base import Backend
from dynamiq.prompts import Message


class SQLiteError(Exception):
    """Base exception class for SQLite-related errors in the memory backend."""

    pass


class SQLite(Backend):
    """SQLite implementation of the memory storage backend."""

    name = "SQLite"

    def __init__(self, db_path: str = "conversations.db", table_name: str = "conversations"):
        """Initializes the SQLite memory storage."""
        self.db_path = db_path
        self.table_name = table_name

        try:
            self._validate_table_name(create_if_not_exists=True)
        except Exception as e:
            raise SQLiteError(f"Error initializing SQLite backend: {e}") from e

    def _validate_table_name(self, create_if_not_exists: bool = False):
        """Validates the table name to prevent SQL injection and optionally creates it."""
        if not re.match(r"^[A-Za-z0-9_]+$", self.table_name):
            raise SQLiteError(f"Invalid table name: '{self.table_name}'")

        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (self.table_name,)
                )  # nosec B608
                result = cursor.fetchone()

                if result is None:
                    if create_if_not_exists:
                        self._create_table()
                        print(f"Table '{self.table_name}' created successfully.")
                    else:
                        raise SQLiteError(f"Table '{self.table_name}' does not exist in the database.")
                else:
                    print(f"Using existing SQLite table '{self.table_name}'.")

        except sqlite3.Error as e:
            raise SQLiteError(f"Error validating or creating table: {e}") from e

    def _create_table(self):
        """Creates the messages table."""
        query = f"""
        CREATE TABLE IF NOT EXISTS {self.table_name} (
            id TEXT PRIMARY KEY,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            metadata TEXT
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
            INSERT INTO {self.table_name} (id, role, content, metadata)
            VALUES (?, ?, ?, ?)
            """  # nosec B608
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    query, (message.id, message.role.value, message.content, json.dumps(message.metadata))
                )  # nosec B608
                conn.commit()

        except sqlite3.Error as e:
            raise SQLiteError(f"Error adding message to database: {e}") from e

    def get_all(self) -> list[Message]:
        """Retrieves all messages from the SQLite database."""
        try:
            self._validate_table_name()  # Ensure table exists
            query = f"SELECT id, role, content, metadata FROM {self.table_name}"  # nosec B608
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(query)  # nosec B608
                rows = cursor.fetchall()
            return [Message(id=row[0], role=row[1], content=row[2], metadata=json.loads(row[3])) for row in rows]

        except sqlite3.Error as e:
            raise SQLiteError(f"Error retrieving messages from database: {e}") from e

    def is_empty(self) -> bool:
        """Checks if the SQLite database is empty."""
        try:
            self._validate_table_name()
            query = f"SELECT COUNT(*) FROM {self.table_name}"  # nosec B608
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
            query = f"DELETE FROM {self.table_name}"  # nosec B608
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(query)  # nosec B608
                conn.commit()
        except sqlite3.Error as e:
            raise SQLiteError(f"Error clearing database: {e}") from e

    def search(self, query: str, search_limit: int) -> list[Message]:
        """Searches for messages in SQLite based on the query."""
        try:
            self._validate_table_name()
            query_str = f"""
            SELECT id, role, content, metadata
            FROM {self.table_name}
            WHERE content LIKE ?
            ORDER BY id DESC LIMIT ?
            """  # nosec B608
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(query_str, (f"%{query}%", search_limit))  # nosec B608
                rows = cursor.fetchall()
            return [Message(id=row[0], role=row[1], content=row[2], metadata=json.loads(row[3])) for row in rows]
        except sqlite3.Error as e:
            raise SQLiteError(f"Error searching in database: {e}") from e
