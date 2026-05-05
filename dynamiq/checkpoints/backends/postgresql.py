from __future__ import annotations

import json
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any

import psycopg
from psycopg.sql import SQL, Composable, Identifier
from pydantic import ConfigDict, Field, PrivateAttr

from dynamiq.checkpoints.backends.base import CheckpointBackend
from dynamiq.checkpoints.types import CheckpointStatus
from dynamiq.checkpoints.utils import decode_checkpoint_data
from dynamiq.connections import PostgreSQL as PostgreSQLConnection
from dynamiq.utils import encode_reversible
from dynamiq.utils.logger import logger

if TYPE_CHECKING:
    from dynamiq.checkpoints.checkpoint import FlowCheckpoint


class FetchMode(str, Enum):
    ONE = "one"
    ALL = "all"


class PostgresCheckpointError(Exception):
    """Raised on PostgreSQL checkpoint backend failures."""

    pass


class PostgreSQL(CheckpointBackend):
    """PostgreSQL-backed checkpoint storage."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str = Field(default="PostgreSQLCheckpoint")
    connection: PostgreSQLConnection = Field(default_factory=PostgreSQLConnection)
    table_name: str = Field(default="flow_checkpoints")
    create_if_not_exist: bool = Field(default=True)

    _conn: psycopg.Connection | None = PrivateAttr(default=None)
    _is_closed: bool = PrivateAttr(default=False)

    @property
    def to_dict_exclude_params(self) -> dict[str, bool]:
        return {"_conn": True, "_is_closed": True, "connection": True}

    def to_dict(self, include_secure_params: bool = False, for_tracing: bool = False, **kwargs) -> dict[str, Any]:
        exclude = kwargs.pop("exclude", self.to_dict_exclude_params.copy())
        data = self.model_dump(exclude=exclude, **kwargs)
        data["connection"] = self.connection.to_dict(for_tracing=for_tracing)
        if "type" not in data:
            data["type"] = self.type
        return data

    def model_post_init(self, __context: Any) -> None:
        try:
            self._conn = self.connection.connect()
            self._is_closed = False
            if self.create_if_not_exist:
                self._create_table_and_indices()
            logger.debug(f"PostgreSQL checkpoint backend connected to table '{self.table_name}'.")
        except Exception as e:
            logger.error(f"Failed to initialize PostgreSQL checkpoint table '{self.table_name}': {e}")
            raise PostgresCheckpointError(f"Failed to initialize PostgreSQL checkpoint backend: {e}") from e

    def close(self) -> None:
        """Explicitly close the underlying connection. Safe to call multiple times."""
        if self._conn and not self._conn.closed:
            try:
                self._conn.close()
            except Exception as e:
                logger.error(f"Error closing PostgreSQL checkpoint connection: {e}")
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
            raise PostgresCheckpointError(
                "PostgreSQL checkpoint backend has been closed. Create a new instance to reconnect."
            )

    def _execute_sql(
        self, sql_query: Composable | str, params: tuple | list | None = None, fetch: FetchMode | None = None
    ):
        self._check_connection_state()

        if self._conn is None or self._conn.closed:
            logger.warning("PostgreSQL checkpoint connection lost; attempting to reconnect.")
            try:
                self._conn = self.connection.connect()
            except Exception as e:
                raise PostgresCheckpointError(f"Failed to re-establish PostgreSQL connection: {e}") from e

        try:
            with self._conn.cursor() as cur:
                cur.execute(sql_query, params)
                if fetch == FetchMode.ONE:
                    return cur.fetchone()
                if fetch == FetchMode.ALL:
                    return cur.fetchall()
                return None
        except psycopg.Error as e:
            sql_str = sql_query.as_string(self._conn) if isinstance(sql_query, Composable) else str(sql_query)
            logger.error(f"PostgreSQL checkpoint error: {e}\nSQL: {sql_str}\nParams: {params}")
            raise PostgresCheckpointError(f"PostgreSQL error: {e}") from e

    def _create_table_and_indices(self) -> None:
        table_sql = SQL(
            """
            CREATE TABLE IF NOT EXISTS {table_name} (
                id TEXT PRIMARY KEY,
                flow_id TEXT NOT NULL,
                run_id TEXT NOT NULL,
                wf_run_id TEXT,
                status TEXT NOT NULL,
                parent_checkpoint_id TEXT,
                created_at TIMESTAMPTZ NOT NULL,
                updated_at TIMESTAMPTZ NOT NULL,
                data JSONB NOT NULL
            );
            """
        ).format(table_name=Identifier(self.table_name))
        self._execute_sql(table_sql)

        short = "".join(filter(str.isalnum, self.table_name))[:16] or "fc"
        for index_name, columns in (
            (f"idx_{short}_flow_created", "(flow_id, created_at DESC)"),
            (f"idx_{short}_run", "(run_id)"),
            (f"idx_{short}_wf_run", "(wf_run_id)"),
            (f"idx_{short}_status", "(status)"),
        ):
            self._execute_sql(
                SQL("CREATE INDEX IF NOT EXISTS {idx} ON {table_name} " + columns + ";").format(
                    idx=Identifier(index_name),
                    table_name=Identifier(self.table_name),
                )
            )

    def _row_to_checkpoint(self, row: dict) -> FlowCheckpoint | None:
        from dynamiq.checkpoints.checkpoint import FlowCheckpoint  # deferred: avoids import cycle at package init

        data = row.get("data")
        if data is None:
            return None
        if isinstance(data, str):
            data = json.loads(data)
        try:
            return FlowCheckpoint(**decode_checkpoint_data(data))
        except Exception:
            checkpoint_id = data.get("id") if isinstance(data, dict) else None
            logger.warning(f"PostgreSQL checkpoint: failed to decode row {checkpoint_id}", exc_info=True)
            return None

    def save(self, checkpoint: FlowCheckpoint) -> str:
        payload = checkpoint.to_dict()
        sql = SQL(
            """
            INSERT INTO {table_name}
                (id, flow_id, run_id, wf_run_id, status, parent_checkpoint_id, created_at, updated_at, data)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (id) DO UPDATE SET
                flow_id = EXCLUDED.flow_id,
                run_id = EXCLUDED.run_id,
                wf_run_id = EXCLUDED.wf_run_id,
                status = EXCLUDED.status,
                parent_checkpoint_id = EXCLUDED.parent_checkpoint_id,
                created_at = EXCLUDED.created_at,
                updated_at = EXCLUDED.updated_at,
                data = EXCLUDED.data;
            """
        ).format(table_name=Identifier(self.table_name))

        params = (
            checkpoint.id,
            checkpoint.flow_id,
            checkpoint.run_id,
            checkpoint.wf_run_id,
            checkpoint.status.value if isinstance(checkpoint.status, CheckpointStatus) else str(checkpoint.status),
            checkpoint.parent_checkpoint_id,
            checkpoint.created_at,
            checkpoint.updated_at,
            json.dumps(payload, default=encode_reversible),
        )
        self._execute_sql(sql, params)
        return checkpoint.id

    def load(self, checkpoint_id: str) -> FlowCheckpoint | None:
        sql = SQL("SELECT data FROM {table_name} WHERE id = %s;").format(table_name=Identifier(self.table_name))
        row = self._execute_sql(sql, (checkpoint_id,), fetch=FetchMode.ONE)
        if not row:
            return None
        return self._row_to_checkpoint(row)

    def update(self, checkpoint: FlowCheckpoint) -> None:
        exists_sql = SQL("SELECT 1 FROM {table_name} WHERE id = %s;").format(table_name=Identifier(self.table_name))
        if not self._execute_sql(exists_sql, (checkpoint.id,), fetch=FetchMode.ONE):
            raise FileNotFoundError(f"Checkpoint {checkpoint.id} not found")
        self.save(checkpoint)

    def delete(self, checkpoint_id: str) -> bool:
        sql = SQL("DELETE FROM {table_name} WHERE id = %s RETURNING id;").format(table_name=Identifier(self.table_name))
        row = self._execute_sql(sql, (checkpoint_id,), fetch=FetchMode.ONE)
        return bool(row)

    def _build_select(self, where: SQL, params: list, limit: int | None) -> tuple[SQL, list]:
        sql = SQL("SELECT data FROM {table_name} ").format(table_name=Identifier(self.table_name))
        if where:
            sql = sql + where
        sql = sql + SQL(" ORDER BY created_at DESC")
        if limit is not None and limit > 0:
            sql = sql + SQL(" LIMIT %s")
            params = list(params) + [limit]
        return sql, params

    def get_list_by_flow(
        self,
        flow_id: str,
        *,
        status: CheckpointStatus | None = None,
        limit: int | None = None,
        before: datetime | None = None,
    ) -> list[FlowCheckpoint]:
        clauses = [SQL("flow_id = %s")]
        params: list = [flow_id]
        if status is not None:
            clauses.append(SQL("status = %s"))
            params.append(status.value)
        if before is not None:
            clauses.append(SQL("created_at < %s"))
            params.append(before)

        where = SQL("WHERE ") + SQL(" AND ").join(clauses)
        sql, params = self._build_select(where, params, limit)
        rows = self._execute_sql(sql, params, fetch=FetchMode.ALL) or []
        return [cp for cp in (self._row_to_checkpoint(r) for r in rows) if cp]

    def get_latest_by_flow(self, flow_id: str, *, status: CheckpointStatus | None = None) -> FlowCheckpoint | None:
        results = self.get_list_by_flow(flow_id, status=status, limit=1)
        return results[0] if results else None

    def get_list_by_run(self, run_id: str, *, limit: int | None = None) -> list[FlowCheckpoint]:
        where = SQL("WHERE run_id = %s OR wf_run_id = %s")
        sql, params = self._build_select(where, [run_id, run_id], limit)
        rows = self._execute_sql(sql, params, fetch=FetchMode.ALL) or []
        return [cp for cp in (self._row_to_checkpoint(r) for r in rows) if cp]

    def get_list_by_flow_and_run(
        self,
        flow_id: str,
        run_id: str,
        *,
        wf_run_id: str | None = None,
        status: CheckpointStatus | None = None,
        limit: int | None = None,
    ) -> list[FlowCheckpoint]:
        ids_to_match = [run_id]
        if wf_run_id and wf_run_id != run_id:
            ids_to_match.append(wf_run_id)

        clauses = [
            SQL("flow_id = %s"),
            SQL("(run_id = ANY(%s) OR wf_run_id = ANY(%s))"),
        ]
        params: list = [flow_id, ids_to_match, ids_to_match]
        if status is not None:
            clauses.append(SQL("status = %s"))
            params.append(status.value)

        where = SQL("WHERE ") + SQL(" AND ").join(clauses)
        sql, params = self._build_select(where, params, limit)
        rows = self._execute_sql(sql, params, fetch=FetchMode.ALL) or []
        return [cp for cp in (self._row_to_checkpoint(r) for r in rows) if cp]

    def clear(self) -> None:
        """Delete every checkpoint in the table."""
        sql = SQL("TRUNCATE TABLE {table_name};").format(table_name=Identifier(self.table_name))
        self._execute_sql(sql)
