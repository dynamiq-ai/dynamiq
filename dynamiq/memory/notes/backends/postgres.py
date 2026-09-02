from enum import Enum
from typing import Any

import psycopg
from psycopg.sql import SQL, Composable, Identifier
from psycopg.types.json import Jsonb
from pydantic import ConfigDict, Field, PrivateAttr

from dynamiq.connections import PostgreSQL as PostgreSQLConnection
from dynamiq.memory.notes.base import NotesBackend, NotesStorageError
from dynamiq.memory.notes.schemas import Note, NoteIndexEntry
from dynamiq.memory.notes.types import NoteWriteOutcome
from dynamiq.utils.logger import logger


class FetchMode(str, Enum):
    ONE = "one"
    ALL = "all"
    ROWCOUNT = "rowcount"


class PostgresNotesError(NotesStorageError):
    """Raised on PostgreSQL notes backend failures."""

    pass


# `INCLUDE` on a PRIMARY KEY needs Postgres 11+. The covering columns make the per-run
# index listing an index-only scan, so it never reads note bodies off the heap.
_CREATE_TABLE_TEMPLATE = SQL(
    """
    CREATE TABLE IF NOT EXISTS {table} (
        user_id     TEXT NOT NULL,
        title       TEXT NOT NULL,
        description TEXT NOT NULL DEFAULT '',
        content     TEXT NOT NULL,
        metadata    JSONB NOT NULL DEFAULT '{{}}'::jsonb,
        created_at  TIMESTAMPTZ NOT NULL,
        updated_at  TIMESTAMPTZ NOT NULL,
        PRIMARY KEY (user_id, title) INCLUDE (description, updated_at)
    )
    """
)

_NOTE_COLUMNS = SQL("user_id, title, description, content, metadata, created_at, updated_at")

# `created_at` is deliberately absent from the SET list so it survives an overwrite.
# `xmax = 0` distinguishes an insert from an update within the same statement.
_UPSERT_TEMPLATE = SQL(
    """
    INSERT INTO {table} (user_id, title, description, content, metadata, created_at, updated_at)
    VALUES (%s, %s, %s, %s, %s, %s, %s)
    ON CONFLICT (user_id, title) DO UPDATE SET
        description = EXCLUDED.description,
        content     = EXCLUDED.content,
        metadata    = EXCLUDED.metadata,
        updated_at  = EXCLUDED.updated_at
    RETURNING (xmax = 0) AS inserted, {columns}
    """
)

_INDEX_ORDER_CLAUSES: dict[str, Composable] = {
    "title": SQL("title ASC"),
    "updated_at": SQL("updated_at DESC, title ASC"),
}


def _row_to_note(row: dict) -> Note:
    return Note(
        user_id=row["user_id"],
        title=row["title"],
        description=row["description"],
        content=row["content"],
        metadata=row["metadata"] or {},
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


class PostgresNotesBackend(NotesBackend):
    """Notes storage backed by a single Postgres table keyed on `(user_id, title)`."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str = "postgres-notes-backend"
    connection: PostgreSQLConnection = Field(default_factory=PostgreSQLConnection)
    table_name: str = "user_notes"
    create_if_not_exist: bool = True

    _conn: psycopg.Connection | None = PrivateAttr(default=None)
    _is_closed: bool = PrivateAttr(default=False)

    @property
    def to_dict_exclude_params(self) -> dict[str, bool]:
        return super().to_dict_exclude_params | {"_conn": True, "_is_closed": True, "connection": True}

    def to_dict(self, include_secure_params: bool = False, for_tracing: bool = False, **kwargs) -> dict[str, Any]:
        data = super().to_dict(include_secure_params=include_secure_params, for_tracing=for_tracing, **kwargs)
        data["connection"] = self.connection.to_dict(
            include_secure_params=include_secure_params, for_tracing=for_tracing, **kwargs
        )
        return data

    @property
    def _table(self) -> Identifier:
        return Identifier(self.table_name)

    def close(self) -> None:
        """Explicitly close the underlying connection. Safe to call multiple times."""
        if self._conn and not self._conn.closed:
            try:
                self._conn.close()
            except Exception as e:
                logger.error(f"Error closing PostgreSQL notes connection: {e}")
            finally:
                self._is_closed = True
        else:
            self._is_closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def _check_connection_state(self) -> None:
        if self._is_closed:
            raise PostgresNotesError("PostgreSQL notes backend has been closed. Create a new instance to reconnect.")

    def _execute_sql(
        self, sql_query: Composable | str, params: tuple | list | None = None, fetch: FetchMode | None = None
    ):
        """Run one statement, connecting or reconnecting as needed.

        Connecting lazily here (rather than in `model_post_init`) means the backend can be
        constructed from YAML or serialized with `to_dict()` without a live database.
        """
        self._check_connection_state()

        if self._conn is None or self._conn.closed:
            try:
                self._conn = self.connection.connect()
            except Exception as e:
                raise PostgresNotesError(f"Failed to establish PostgreSQL connection: {e}") from e

        try:
            with self._conn.cursor() as cur:
                cur.execute(sql_query, params)
                if fetch == FetchMode.ONE:
                    return cur.fetchone()
                if fetch == FetchMode.ALL:
                    return cur.fetchall()
                if fetch == FetchMode.ROWCOUNT:
                    return cur.rowcount
                return None
        except psycopg.Error as e:
            sql_str = sql_query.as_string(self._conn) if isinstance(sql_query, Composable) else str(sql_query)
            logger.error(f"PostgreSQL notes error: {e}\nSQL: {sql_str}\nParams: {params}")
            raise PostgresNotesError(f"PostgreSQL error: {e}") from e

    def _ensure_storage(self) -> None:
        if self.create_if_not_exist:
            self.ensure_table()

    def ensure_table(self) -> None:
        """Create the notes table if absent. Safe to call repeatedly."""
        self._execute_sql(_CREATE_TABLE_TEMPLATE.format(table=self._table))

    def drop_table(self) -> None:
        """Drop the notes table. Test-only helper."""
        self._execute_sql(SQL("DROP TABLE IF EXISTS {table}").format(table=self._table))

    def upsert(self, note: Note) -> tuple[Note, NoteWriteOutcome]:
        row = self._execute_sql(
            _UPSERT_TEMPLATE.format(table=self._table, columns=_NOTE_COLUMNS),
            (
                note.user_id,
                note.title,
                note.description,
                note.content,
                Jsonb(note.metadata),
                note.created_at,
                note.updated_at,
            ),
            fetch=FetchMode.ONE,
        )
        if row is None:
            raise PostgresNotesError(f"Upsert of note {note.title!r} returned no row")
        outcome = NoteWriteOutcome.CREATED if row["inserted"] else NoteWriteOutcome.UPDATED
        return _row_to_note(row), outcome

    def get_many(self, *, user_id: str, titles: list[str]) -> list[Note]:
        if not titles:
            return []
        rows = self._execute_sql(
            SQL("SELECT {columns} FROM {table} WHERE user_id = %s AND title = ANY(%s)").format(
                columns=_NOTE_COLUMNS, table=self._table
            ),
            (user_id, list(titles)),
            fetch=FetchMode.ALL,
        )
        return [_row_to_note(row) for row in rows or []]

    def delete_by_title(self, *, user_id: str, title: str) -> bool:
        row = self._execute_sql(
            SQL("DELETE FROM {table} WHERE user_id = %s AND title = %s RETURNING title").format(table=self._table),
            (user_id, title),
            fetch=FetchMode.ONE,
        )
        return row is not None

    def list_index(self, *, user_id: str, limit: int, order_by: str) -> list[NoteIndexEntry]:
        if limit <= 0:
            return []
        order_clause = _INDEX_ORDER_CLAUSES.get(order_by)
        if order_clause is None:
            raise PostgresNotesError(f"Unsupported index order_by: {order_by!r}")
        rows = self._execute_sql(
            SQL(
                "SELECT title, description, updated_at FROM {table} WHERE user_id = %s ORDER BY {order} LIMIT %s"
            ).format(table=self._table, order=order_clause),
            (user_id, limit),
            fetch=FetchMode.ALL,
        )
        return [
            NoteIndexEntry(title=row["title"], description=row["description"], updated_at=row["updated_at"])
            for row in rows or []
        ]

    def list_by_user(self, *, user_id: str, limit: int) -> list[Note]:
        if limit <= 0:
            return []
        rows = self._execute_sql(
            SQL("SELECT {columns} FROM {table} WHERE user_id = %s ORDER BY title ASC LIMIT %s").format(
                columns=_NOTE_COLUMNS, table=self._table
            ),
            (user_id, limit),
            fetch=FetchMode.ALL,
        )
        return [_row_to_note(row) for row in rows or []]

    def delete_user(self, *, user_id: str) -> int:
        deleted = self._execute_sql(
            SQL("DELETE FROM {table} WHERE user_id = %s").format(table=self._table),
            (user_id,),
            fetch=FetchMode.ROWCOUNT,
        )
        return max(deleted or 0, 0)
