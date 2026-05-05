"""Tests for the PostgreSQL checkpoint backend.

Mocks ``psycopg.connect`` to mirror the convention from
``tests/integration/nodes/tools/test_sql_executor.py`` and other
Postgres-touching tests in this repo. No real database required.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import psycopg
import pytest

from dynamiq.checkpoints.backends.postgresql import FetchMode, PostgresCheckpointError
from dynamiq.checkpoints.backends.postgresql import PostgreSQL as PostgresCheckpointBackend
from dynamiq.checkpoints.checkpoint import FlowCheckpoint
from dynamiq.connections import PostgreSQL as PostgresConn


@pytest.fixture
def mock_cursor():
    """psycopg cursor mock that supports the ``with conn.cursor() as cur`` pattern."""
    cursor = MagicMock()
    cursor.__enter__.return_value = cursor
    cursor.__exit__.return_value = None
    cursor.fetchone.return_value = None
    cursor.fetchall.return_value = []
    return cursor


@pytest.fixture
def mock_connection(mock_cursor):
    conn = MagicMock()
    conn.closed = False
    conn.cursor.return_value = mock_cursor
    return conn


@pytest.fixture
def patch_psycopg_connect(mocker, mock_connection):
    return mocker.patch("psycopg.connect", return_value=mock_connection)


@pytest.fixture
def backend(patch_psycopg_connect) -> PostgresCheckpointBackend:
    return PostgresCheckpointBackend(
        connection=PostgresConn(host="h", port=5432, database="d", user="u", password="p"),
        table_name="flow_checkpoints_test",
        create_if_not_exist=False,
    )


def test_init_failure_wraps_exception(mocker):
    mocker.patch("psycopg.connect", side_effect=Exception("boom"))
    with pytest.raises(PostgresCheckpointError):
        PostgresCheckpointBackend(
            connection=PostgresConn(host="h", port=5432, database="d", user="u", password="p"),
            create_if_not_exist=False,
        )


def test_save_returns_checkpoint_id(backend):
    cp = FlowCheckpoint(flow_id="flow-1", run_id="run-1", original_input={"q": "hi"})
    assert backend.save(cp) == cp.id


def test_load_returns_none_when_missing(backend, mock_cursor):
    mock_cursor.fetchone.return_value = None
    assert backend.load("missing") is None


def test_update_missing_raises_file_not_found(backend, mock_cursor):
    mock_cursor.fetchone.return_value = None
    cp = FlowCheckpoint(flow_id="flow-1", run_id="run-1")
    with pytest.raises(FileNotFoundError):
        backend.update(cp)


def test_delete_returns_true_when_row_returned(backend, mock_cursor):
    mock_cursor.fetchone.return_value = {"id": "abc"}
    assert backend.delete("abc") is True


def test_delete_returns_false_when_no_row(backend, mock_cursor):
    mock_cursor.fetchone.return_value = None
    assert backend.delete("abc") is False


def test_close_calls_connection_close_once_and_is_idempotent(backend, mock_connection):
    backend.close()
    mock_connection.close.assert_called_once()

    mock_connection.closed = True
    backend.close()
    assert mock_connection.close.call_count == 1


def test_calls_after_close_raise(backend):
    backend.close()
    with pytest.raises(PostgresCheckpointError):
        backend.load("anything")


def test_execute_sql_wraps_psycopg_errors(backend, mock_cursor):
    mock_cursor.execute.side_effect = psycopg.Error("simulated failure")
    with pytest.raises(PostgresCheckpointError):
        backend._execute_sql("SELECT 1", fetch=FetchMode.ONE)
