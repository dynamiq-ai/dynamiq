from datetime import datetime
from typing import Any

import psycopg
from pgvector.psycopg import register_vector
from psycopg.rows import dict_row
from psycopg.sql import SQL, Composed, Identifier
from psycopg.types.json import Jsonb
from pydantic import ConfigDict, Field, PrivateAttr

from dynamiq.connections import PostgreSQL as PostgreSQLConnection
from dynamiq.memory.long_term.base import LongTermMemoryBackend
from dynamiq.memory.long_term.schemas import Fact

_CREATE_EXTENSION_SQL = SQL("CREATE EXTENSION IF NOT EXISTS vector")

_CREATE_TABLE_TEMPLATE = SQL(
    """
    CREATE TABLE IF NOT EXISTS {table} (
        id          TEXT PRIMARY KEY,
        content     TEXT NOT NULL,
        hash        TEXT NOT NULL,
        user_id     TEXT NOT NULL,
        metadata    JSONB NOT NULL DEFAULT '{{}}'::jsonb,
        embedding   vector({dim}) NOT NULL,
        created_at  TIMESTAMPTZ NOT NULL,
        updated_at  TIMESTAMPTZ NOT NULL
    )
    """
)

_CREATE_USER_ID_INDEX_TEMPLATE = SQL("CREATE INDEX IF NOT EXISTS {idx} ON {table} (user_id)")

_CREATE_USER_HASH_INDEX_TEMPLATE = SQL("CREATE UNIQUE INDEX IF NOT EXISTS {idx} ON {table} (user_id, hash)")


def _scope_where_clause(scope: dict[str, str]) -> tuple[Composed, list]:
    """Build a parameterised WHERE clause from a scope dict."""
    if not scope:
        return SQL("TRUE"), []
    clauses = [SQL("{key} = %s").format(key=Identifier(key)) for key in scope.keys()]
    return SQL(" AND ").join(clauses), list(scope.values())


def _row_to_fact(row) -> Fact:
    return Fact(
        id=row["id"],
        content=row["content"],
        hash=row["hash"],
        user_id=row["user_id"],
        metadata=row["metadata"] or {},
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


_FACT_COLUMNS = SQL("id, content, hash, user_id, metadata, created_at, updated_at")


class PostgresLongTermMemoryBackend(LongTermMemoryBackend):
    """Long-term memory backend backed by Postgres + pgvector."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str = "postgres-long-term-memory-backend"
    connection: PostgreSQLConnection = Field(default_factory=PostgreSQLConnection)
    table_name: str = "user_facts"
    dimension: int = 1536

    _conn: psycopg.Connection | None = PrivateAttr(default=None)

    @property
    def to_dict_exclude_params(self) -> dict[str, bool]:
        return super().to_dict_exclude_params | {"_conn": True, "connection": True}

    def to_dict(self, include_secure_params: bool = False, for_tracing: bool = False, **kwargs) -> dict[str, Any]:
        data = super().to_dict(include_secure_params=include_secure_params, for_tracing=for_tracing, **kwargs)
        data["connection"] = self.connection.to_dict(
            for_tracing=for_tracing, include_secure_params=include_secure_params, **kwargs
        )
        return data

    def model_post_init(self, __context) -> None:
        self._conn = self.connection.connect()
        self._conn.autocommit = True
        with self._conn.cursor() as cur:
            cur.execute(_CREATE_EXTENSION_SQL)
        register_vector(self._conn)

    @property
    def _table(self) -> Identifier:
        """Return the table name wrapped as a safe SQL identifier."""
        return Identifier(self.table_name)

    def _ensure_storage(self) -> None:
        self.ensure_table()

    def ensure_table(self) -> None:
        """Create the facts table and indexes if absent. Safe to call repeatedly."""
        with self._conn.cursor() as cur:
            cur.execute(_CREATE_EXTENSION_SQL)
            cur.execute(_CREATE_TABLE_TEMPLATE.format(table=self._table, dim=SQL(str(self.dimension))))
            cur.execute(
                _CREATE_USER_ID_INDEX_TEMPLATE.format(
                    idx=Identifier(f"{self.table_name}_user_id_idx"),
                    table=self._table,
                )
            )
            cur.execute(
                _CREATE_USER_HASH_INDEX_TEMPLATE.format(
                    idx=Identifier(f"{self.table_name}_user_hash_uidx"),
                    table=self._table,
                )
            )

    def recreate_table(self) -> None:
        """Drop and re-create the facts table. Test-only helper."""
        with self._conn.cursor() as cur:
            cur.execute(SQL("DROP TABLE IF EXISTS {table}").format(table=self._table))
        self.ensure_table()

    def drop_table(self) -> None:
        """Drop the facts table if it exists. Test-only helper."""
        with self._conn.cursor() as cur:
            cur.execute(SQL("DROP TABLE IF EXISTS {table}").format(table=self._table))

    def insert(self, fact: Fact, embedding: list[float]) -> None:
        with self._conn.cursor() as cur:
            cur.execute(
                SQL("INSERT INTO {table} ({cols}, embedding) " "VALUES (%s, %s, %s, %s, %s, %s, %s, %s)").format(
                    table=self._table, cols=_FACT_COLUMNS
                ),
                (
                    fact.id,
                    fact.content,
                    fact.hash,
                    fact.user_id,
                    Jsonb(fact.metadata),
                    fact.created_at,
                    fact.updated_at,
                    embedding,
                ),
            )

    def get(self, fact_id: str) -> Fact | None:
        with self._conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                SQL("SELECT {cols} FROM {table} WHERE id = %s").format(
                    cols=_FACT_COLUMNS,
                    table=self._table,
                ),
                (fact_id,),
            )
            row = cur.fetchone()
        return _row_to_fact(row) if row else None

    def get_by_hash(self, *, user_id: str, content_hash: str) -> Fact | None:
        with self._conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                SQL("SELECT {cols} FROM {table} WHERE user_id = %s AND hash = %s").format(
                    cols=_FACT_COLUMNS, table=self._table
                ),
                (user_id, content_hash),
            )
            row = cur.fetchone()
        return _row_to_fact(row) if row else None

    def delete(self, fact_id: str) -> None:
        with self._conn.cursor() as cur:
            cur.execute(
                SQL("DELETE FROM {table} WHERE id = %s").format(table=self._table),
                (fact_id,),
            )

    def update(
        self,
        fact_id: str,
        *,
        content: str,
        content_hash: str,
        embedding: list[float],
        metadata: dict,
        updated_at: datetime,
    ) -> None:
        with self._conn.cursor() as cur:
            cur.execute(
                SQL(
                    "UPDATE {table} SET content = %s, hash = %s, embedding = %s, "
                    "metadata = %s, updated_at = %s WHERE id = %s"
                ).format(table=self._table),
                (content, content_hash, embedding, Jsonb(metadata), updated_at, fact_id),
            )

    def search(
        self,
        *,
        query_embedding: list[float],
        scope: dict[str, str],
        limit: int,
    ) -> list[tuple[Fact, float]]:
        where, params = _scope_where_clause(scope)
        with self._conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                SQL(
                    "SELECT {cols}, 1 - (embedding <=> %s::vector) AS score "
                    "FROM {table} WHERE {where} "
                    "ORDER BY embedding <=> %s::vector LIMIT %s"
                ).format(cols=_FACT_COLUMNS, table=self._table, where=where),
                [query_embedding] + params + [query_embedding, limit],
            )
            rows = cur.fetchall()
        return [(_row_to_fact(row), float(row["score"])) for row in rows]

    def list_by_scope(self, scope: dict[str, str], limit: int = 100) -> list[Fact]:
        where, params = _scope_where_clause(scope)
        with self._conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                SQL("SELECT {cols} FROM {table} WHERE {where} " "ORDER BY created_at DESC LIMIT %s").format(
                    cols=_FACT_COLUMNS, table=self._table, where=where
                ),
                params + [limit],
            )
            rows = cur.fetchall()
        return [_row_to_fact(row) for row in rows]

    def delete_scope(self, scope: dict[str, str]) -> int:
        if not scope:
            raise ValueError("delete_scope requires a non-empty scope")
        where, params = _scope_where_clause(scope)
        with self._conn.cursor() as cur:
            cur.execute(
                SQL("DELETE FROM {table} WHERE {where}").format(
                    table=self._table,
                    where=where,
                ),
                params,
            )
            return cur.rowcount
