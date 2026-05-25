"""pgvector-backed long-term memory backend.

Uses psycopg (v3) + the pgvector extension. Stores facts in a single
table with a vector column for embeddings, a JSONB column for metadata,
and (user_id, hash) uniqueness for dedup.
"""
import psycopg
from pgvector.psycopg import register_vector
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from pydantic import ConfigDict, PrivateAttr

from dynamiq.memory.long_term.base import LongTermMemoryBackend
from dynamiq.memory.long_term.schemas import Fact


_SCHEMA_SQL = """
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS {table} (
    id          TEXT PRIMARY KEY,
    content     TEXT NOT NULL,
    hash        TEXT NOT NULL,
    user_id     TEXT NOT NULL,
    metadata    JSONB NOT NULL DEFAULT '{{}}'::jsonb,
    embedding   vector({dim}) NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL,
    updated_at  TIMESTAMPTZ NOT NULL
);

CREATE INDEX IF NOT EXISTS {table}_user_id_idx ON {table} (user_id);
CREATE UNIQUE INDEX IF NOT EXISTS {table}_user_hash_uidx ON {table} (user_id, hash);
"""


def _scope_to_where(scope: dict[str, str]) -> tuple[str, list]:
    """Translate a scope dict into a parameterised SQL WHERE clause.

    `scope` is always `{"user_id": ...}` in v1; the loop is shaped so
    forward extensions (agent_id, run_id) drop in without rewriting.
    """
    if not scope:
        return "TRUE", []
    clauses = [f"{key} = %s" for key in scope.keys()]
    return " AND ".join(clauses), list(scope.values())


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


class PgvectorFactBackend(LongTermMemoryBackend):
    """Long-term memory backend backed by Postgres + pgvector."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    dsn: str
    table_name: str = "user_facts"
    dimension: int = 1536

    _conn: psycopg.Connection | None = PrivateAttr(default=None)

    def model_post_init(self, __context) -> None:
        self._conn = psycopg.connect(self.dsn, autocommit=True)
        register_vector(self._conn)

    # --- schema management (test/admin helpers, not part of the ABC) ---

    def ensure_table(self) -> None:
        """Create the facts table and indexes if absent. Safe to call repeatedly."""
        with self._conn.cursor() as cur:
            cur.execute(_SCHEMA_SQL.format(table=self.table_name, dim=self.dimension))

    def recreate_table(self) -> None:
        """Drop and re-create the facts table. For tests only."""
        with self._conn.cursor() as cur:
            cur.execute(f"DROP TABLE IF EXISTS {self.table_name}")
        self.ensure_table()

    def drop_table(self) -> None:
        with self._conn.cursor() as cur:
            cur.execute(f"DROP TABLE IF EXISTS {self.table_name}")

    # --- LongTermMemoryBackend implementation ---

    def insert(self, fact: Fact, embedding: list[float]) -> None:
        with self._conn.cursor() as cur:
            cur.execute(
                f"""
                INSERT INTO {self.table_name}
                (id, content, hash, user_id, metadata, embedding, created_at, updated_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    fact.id,
                    fact.content,
                    fact.hash,
                    fact.user_id,
                    Jsonb(fact.metadata),
                    embedding,
                    fact.created_at,
                    fact.updated_at,
                ),
            )

    def get(self, fact_id: str) -> Fact | None:
        with self._conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                f"SELECT id, content, hash, user_id, metadata, created_at, updated_at "
                f"FROM {self.table_name} WHERE id = %s",
                (fact_id,),
            )
            row = cur.fetchone()
        return _row_to_fact(row) if row else None

    def get_by_hash(self, *, user_id: str, content_hash: str) -> Fact | None:
        with self._conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                f"SELECT id, content, hash, user_id, metadata, created_at, updated_at "
                f"FROM {self.table_name} WHERE user_id = %s AND hash = %s",
                (user_id, content_hash),
            )
            row = cur.fetchone()
        return _row_to_fact(row) if row else None

    def delete(self, fact_id: str) -> None:
        with self._conn.cursor() as cur:
            cur.execute(f"DELETE FROM {self.table_name} WHERE id = %s", (fact_id,))

    def search(
        self,
        *,
        query_embedding: list[float],
        scope: dict[str, str],
        limit: int,
    ) -> list[tuple[Fact, float]]:
        where, params = _scope_to_where(scope)
        with self._conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                f"""
                SELECT id, content, hash, user_id, metadata, created_at, updated_at,
                       1 - (embedding <=> %s::vector) AS score
                FROM {self.table_name}
                WHERE {where}
                ORDER BY embedding <=> %s::vector
                LIMIT %s
                """,
                [query_embedding] + params + [query_embedding, limit],
            )
            rows = cur.fetchall()
        return [(_row_to_fact(row), float(row["score"])) for row in rows]

    def list_by_scope(self, scope: dict[str, str], limit: int = 100) -> list[Fact]:
        where, params = _scope_to_where(scope)
        with self._conn.cursor(row_factory=dict_row) as cur:
            cur.execute(
                f"SELECT id, content, hash, user_id, metadata, created_at, updated_at "
                f"FROM {self.table_name} WHERE {where} "
                f"ORDER BY created_at DESC LIMIT %s",
                params + [limit],
            )
            rows = cur.fetchall()
        return [_row_to_fact(row) for row in rows]

    def delete_scope(self, scope: dict[str, str]) -> int:
        where, params = _scope_to_where(scope)
        with self._conn.cursor() as cur:
            cur.execute(f"DELETE FROM {self.table_name} WHERE {where}", params)
            return cur.rowcount
