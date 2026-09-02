import json
import os
import threading
from urllib.parse import urlparse

import pytest
from psycopg.sql import SQL

from dynamiq.connections import PostgreSQL as PostgreSQLConnection
from dynamiq.memory.notes import NoteDeleteStatus, NoteWriteOutcome
from dynamiq.memory.notes.backends.postgres import FetchMode, PostgresNotesBackend, PostgresNotesError

DSN = os.getenv("POSTGRES_DSN")
pytestmark = pytest.mark.skipif(DSN is None, reason="POSTGRES_DSN not set")

TABLE = "test_user_notes"
USER_ID = "user-notes-1"
OTHER_USER_ID = "user-notes-2"


def _connection_from_dsn(dsn: str) -> PostgreSQLConnection:
    parsed = urlparse(dsn)
    return PostgreSQLConnection(
        host=parsed.hostname or "localhost",
        port=parsed.port or 5432,
        database=(parsed.path or "/postgres").lstrip("/"),
        user=parsed.username or "postgres",
        password=parsed.password or "",
    )


def _make_backend(**kwargs) -> PostgresNotesBackend:
    return PostgresNotesBackend(connection=_connection_from_dsn(DSN), table_name=TABLE, **kwargs)


@pytest.fixture
def backend():
    b = _make_backend()
    b.drop_table()
    b.ensure_table()
    yield b
    b.drop_table()
    b.close()


def test_ddl_is_created_lazily_on_first_use():
    """Nothing runs in `model_post_init`; the table appears on the first operation."""
    fresh = _make_backend()
    fresh.drop_table()

    fresh.write(user_id=USER_ID, title="Lazy", description="d", content="c")

    found, _ = fresh.read(user_id=USER_ID, titles=["Lazy"])
    assert found[0].content == "c"
    fresh.drop_table()
    fresh.close()


def test_upsert_creates_then_updates_preserving_created_at(backend):
    first, first_outcome = backend.write(user_id=USER_ID, title="Runbook", description="v1", content="body v1")
    second, second_outcome = backend.write(user_id=USER_ID, title="Runbook", description="v2", content="body v2")

    assert first_outcome is NoteWriteOutcome.CREATED
    assert second_outcome is NoteWriteOutcome.UPDATED
    assert second.created_at == first.created_at
    assert second.updated_at > first.updated_at
    assert (second.description, second.content) == ("v2", "body v2")
    assert len(backend.list_all(user_id=USER_ID)) == 1


def test_metadata_round_trips_as_jsonb(backend):
    backend.write(user_id=USER_ID, title="Tagged", description="d", content="c", metadata={"category": "ops", "n": 3})

    found, _ = backend.read(user_id=USER_ID, titles=["Tagged"])

    assert found[0].metadata == {"category": "ops", "n": 3}


def test_the_same_title_under_two_users_is_two_rows(backend):
    backend.write(user_id=USER_ID, title="Shared", description="mine", content="mine")
    backend.write(user_id=OTHER_USER_ID, title="Shared", description="theirs", content="theirs")

    mine, _ = backend.read(user_id=USER_ID, titles=["Shared"])
    theirs, _ = backend.read(user_id=OTHER_USER_ID, titles=["Shared"])

    assert (mine[0].content, theirs[0].content) == ("mine", "theirs")


def test_concurrent_writers_on_one_title_produce_a_single_row(backend):
    """`ON CONFLICT` serialises on the primary key: no UniqueViolation escapes."""
    backends = [_make_backend() for _ in range(4)]
    barrier = threading.Barrier(len(backends))
    errors: list[Exception] = []

    def write(b, i):
        try:
            barrier.wait()
            b.write(user_id=USER_ID, title="Contended", description=f"d{i}", content=f"c{i}")
        except Exception as e:  # pragma: no cover - only on a real failure
            errors.append(e)

    threads = [threading.Thread(target=write, args=(b, i)) for i, b in enumerate(backends)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    for b in backends:
        b.close()

    assert errors == []
    assert len(backend.list_all(user_id=USER_ID)) == 1


def test_get_many_returns_only_the_requested_subset(backend):
    for title in ("a", "b", "c"):
        backend.write(user_id=USER_ID, title=title, description="d", content="c")

    found, missing = backend.read(user_id=USER_ID, titles=["a", "c", "zzz"])

    assert [note.title for note in found] == ["a", "c"]
    assert missing == ["zzz"]


def test_delete_reports_status_and_leaves_other_users_alone(backend):
    backend.write(user_id=USER_ID, title="Mine", description="d", content="c")
    backend.write(user_id=OTHER_USER_ID, title="Mine", description="d", content="c")

    assert backend.delete(user_id=USER_ID, title="Mine") is NoteDeleteStatus.DELETED
    assert backend.delete(user_id=USER_ID, title="Mine") is NoteDeleteStatus.NOT_FOUND
    assert len(backend.list_all(user_id=OTHER_USER_ID)) == 1


def test_index_never_reads_note_bodies(backend):
    huge = "x" * 100_000
    backend.write(user_id=USER_ID, title="Big", description="a big note", content=huge)

    entries, truncated = backend.index(user_id=USER_ID)

    assert [(e.title, e.description) for e in entries] == [("Big", "a big note")]
    assert truncated is False
    assert huge not in json.dumps([e.model_dump(mode="json") for e in entries])


def test_index_is_ordered_by_title_and_can_order_by_recency(backend):
    backend.write(user_id=USER_ID, title="beta", description="d", content="c")
    backend.write(user_id=USER_ID, title="alpha", description="d", content="c")
    backend.write(user_id=USER_ID, title="beta", description="d2", content="c2")

    by_title, _ = backend.index(user_id=USER_ID)
    by_recency, _ = backend.index(user_id=USER_ID, order_by="updated_at")

    assert [e.title for e in by_title] == ["alpha", "beta"]
    assert [e.title for e in by_recency] == ["beta", "alpha"]


def test_index_listing_uses_an_index_only_scan(backend):
    """The covering PRIMARY KEY is what keeps the per-run hot path off the table heap."""
    backend.write(user_id=USER_ID, title="Big", description="d", content="x" * 100_000)

    plan = backend._execute_sql(
        SQL(
            "EXPLAIN (FORMAT JSON) SELECT title, description, updated_at FROM {table} "
            "WHERE user_id = %s ORDER BY title ASC LIMIT %s"
        ).format(table=backend._table),
        (USER_ID, 10),
        fetch=FetchMode.ONE,
    )
    rendered = json.dumps(plan)

    assert "Index Only Scan" in rendered
    assert f"{TABLE}_pkey" in rendered


def test_index_reports_truncation(backend):
    capped = _make_backend(max_index_entries=2)
    for i in range(5):
        backend.write(user_id=USER_ID, title=f"note-{i}", description="d", content="c")

    entries, truncated = capped.index(user_id=USER_ID)
    capped.close()

    assert len(entries) == 2
    assert truncated is True


def test_clear_user_returns_the_deleted_count(backend):
    for i in range(3):
        backend.write(user_id=USER_ID, title=f"note-{i}", description="d", content="c")
    backend.write(user_id=OTHER_USER_ID, title="Theirs", description="d", content="c")

    assert backend.clear_user(user_id=USER_ID) == 3
    assert backend.clear_user(user_id=USER_ID) == 0
    assert len(backend.list_all(user_id=OTHER_USER_ID)) == 1


def test_a_dropped_socket_reconnects_silently(backend):
    """The failure mode `PostgresLongTermMemoryBackend` cannot recover from."""
    backend.write(user_id=USER_ID, title="Before", description="d", content="c")
    backend._conn.close()

    backend.write(user_id=USER_ID, title="After", description="d", content="c")

    assert {note.title for note in backend.list_all(user_id=USER_ID)} == {"Before", "After"}


def test_close_makes_further_operations_fail(backend):
    closed = _make_backend()
    closed.write(user_id=USER_ID, title="One", description="d", content="c")
    closed.close()

    with pytest.raises(PostgresNotesError, match="closed"):
        closed.list_all(user_id=USER_ID)


def test_to_dict_needs_no_live_connection():
    data = _make_backend().to_dict()

    assert data["type"] == "dynamiq.memory.notes.backends.PostgresNotesBackend"
    assert isinstance(data["connection"], dict)
