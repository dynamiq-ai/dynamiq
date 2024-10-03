import json
import re
import sqlite3

from dynamiq.memory.backend.base import Backend
from dynamiq.prompts import Message


class SQLite(Backend):
    """SQLite implementation of the memory storage backend."""

    def __init__(self, db_path: str = "conversations.db", table_name: str = "conversations"):
        """Initializes the SQLite memory storage."""
        self.db_path = db_path
        self.table_name = table_name
        self._validate_table_name(create_if_not_exists=True)  # Validate and create table if needed

    def _validate_table_name(self, create_if_not_exists: bool = False):
        """Validates the table name to prevent SQL injection and optionally creates it."""
        if not re.match(r"^[A-Za-z0-9_]+$", self.table_name):
            raise ValueError(f"Invalid table name: {self.table_name}")

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (self.table_name,))
            result = cursor.fetchone()

            if result is None:
                if create_if_not_exists:
                    self._create_table()
                else:
                    raise ValueError(f"Table {self.table_name} does not exist in the database.")

    def _create_table(self):
        """Creates the messages table if it doesn't exist."""
        query = """
        CREATE TABLE IF NOT EXISTS {} (
            id TEXT PRIMARY KEY,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            metadata TEXT
        )
        """.format(
            self.table_name
        )  # nosec B608
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(query)  # nosec B608
            conn.commit()

    def add(self, message: Message):
        """Stores a message in the SQLite database."""
        self._validate_table_name(create_if_not_exists=True)
        query = f"INSERT INTO {self.table_name} (id, role, content, metadata) VALUES (?, ?, ?, ?)"  # nosec B608
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                query,
                (message.id, message.role.value, message.content, json.dumps(message.metadata)),
            )
            conn.commit()

    def get_all(self) -> list[Message]:
        """Retrieves all messages from the SQLite database."""
        self._validate_table_name(create_if_not_exists=True)
        query = f"SELECT id, role, content, metadata FROM {self.table_name}"  # nosec B608
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(query)  # nosec B608
            rows = cursor.fetchall()
            return [Message(id=row[0], role=row[1], content=row[2], metadata=json.loads(row[3])) for row in rows]

    def is_empty(self) -> bool:
        """Checks if the SQLite database is empty."""
        self._validate_table_name(create_if_not_exists=True)
        query = f"SELECT COUNT(*) FROM {self.table_name}"  # nosec B608
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(query)  # nosec B608
            count = cursor.fetchone()[0]
            return count == 0

    def clear(self):
        """Clears the SQLite database by deleting all rows in the table."""
        self._validate_table_name(create_if_not_exists=True)
        query = f"DELETE FROM {self.table_name}"  # nosec B608
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(query)  # nosec B608
            conn.commit()

    def search(self, query: str, search_limit: int) -> list[Message]:
        """Searches for messages in SQLite based on the query."""
        self._validate_table_name(create_if_not_exists=True)
        query_str = "SELECT id, role, content, metadata FROM ? WHERE content LIKE ? ORDER BY id DESC LIMIT ?"
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(query_str, (self.table_name, f"%{query}%", search_limit))
            rows = cursor.fetchall()
        return [Message(id=row[0], role=row[1], content=row[2], metadata=json.loads(row[3])) for row in rows]
