"""Every memory backend must honour the same get_all/search contract.

MemoryBackend.get_all documents "returns the most recent messages ... sorted by
timestamp (oldest first)". InMemory and SQLite implemented that; PostgreSQL ordered
ASC and then LIMIT-ed, so it returned the *oldest* N. The effect was silent: past
memory_limit an agent would only ever see the opening messages of a conversation.

SQLite.search had a related bug -- it ordered by `id`, a random uuid4, so "the n most
recent matches" was an arbitrary n in arbitrary order.

These run against the backends that need no live server. The PostgreSQL SQL is
asserted separately below, since it cannot be exercised without a database.
"""

import pytest

from dynamiq.memory.backends import InMemory, SQLite
from dynamiq.prompts import Message, MessageRole

TOTAL = 10
LIMIT = 3


def _messages() -> list[Message]:
    """Ten user messages, monotonically increasing timestamps."""
    return [
        Message(role=MessageRole.USER, content=f"msg-{i}", metadata={"timestamp": float(i)})
        for i in range(TOTAL)
    ]


@pytest.fixture(params=["in_memory", "sqlite"])
def backend(request, tmp_path):
    if request.param == "in_memory":
        return InMemory()
    return SQLite(db_path=str(tmp_path / "conv.db"), index_name="conversations")


def test_get_all_with_limit_returns_the_most_recent(backend):
    for message in _messages():
        backend.add(message)

    result = backend.get_all(limit=LIMIT)

    assert [m.content for m in result] == ["msg-7", "msg-8", "msg-9"], (
        f"{backend.name}.get_all(limit={LIMIT}) must return the newest {LIMIT} "
        "messages, oldest-first"
    )


def test_get_all_without_limit_returns_everything_oldest_first(backend):
    for message in _messages():
        backend.add(message)

    assert [m.content for m in backend.get_all()] == [f"msg-{i}" for i in range(TOTAL)]


def test_get_all_limit_larger_than_stored_is_not_padded_or_truncated(backend):
    for message in _messages()[:2]:
        backend.add(message)

    assert [m.content for m in backend.get_all(limit=LIMIT)] == ["msg-0", "msg-1"]


def test_sqlite_search_with_limit_returns_the_most_recent_matches(tmp_path):
    """SQLite search is substring + recency (unlike InMemory, which is BM25-ranked).

    It ordered by `id` -- a random uuid4 -- so the "n most recent matches" were an
    arbitrary n in arbitrary order. This is the path Memory.get_agent_conversation
    uses for every retrieval strategy.
    """
    backend = SQLite(db_path=str(tmp_path / "conv.db"), index_name="conversations")
    for message in _messages():
        backend.add(message)

    result = backend.search(query="msg", limit=LIMIT)

    assert sorted(m.content for m in result) == ["msg-7", "msg-8", "msg-9"], (
        f"SQLite.search must return the newest {LIMIT} matches, not an arbitrary subset"
    )


@pytest.fixture
def postgres_backend(mocker):
    """A PostgreSQL backend whose SQL execution is captured instead of run.

    Postgres needs a live server, so this stubs _execute_sql and model_post_init
    (which would otherwise connect to create the table).
    """
    from dynamiq.memory.backends.postgresql import PostgreSQL

    mocker.patch.object(PostgreSQL, "model_post_init", lambda self, __context: None)
    backend = PostgreSQL(table_name="conversations")
    executed = mocker.patch.object(PostgreSQL, "_execute_sql", return_value=[])
    return backend, executed


def _rendered_sql(executed_mock) -> str:
    """Render the psycopg Composable that was handed to _execute_sql."""
    return executed_mock.call_args.args[0].as_string(None)


def test_postgres_get_all_takes_newest_rows_when_limited(postgres_backend):
    """ASC + LIMIT is the bug: it takes the oldest N rows."""
    backend, executed = postgres_backend

    backend.get_all(limit=LIMIT)

    sql = _rendered_sql(executed)
    assert "DESC" in sql and "LIMIT" in sql, (
        f"limited get_all must order DESC so LIMIT takes the newest rows; got: {sql}"
    )


def test_postgres_get_all_unlimited_stays_ascending(postgres_backend):
    backend, executed = postgres_backend

    backend.get_all()

    sql = _rendered_sql(executed)
    assert "ASC" in sql and "LIMIT" not in sql


def test_postgres_get_all_returns_oldest_first(postgres_backend, mocker):
    """Rows arrive newest-first when limited; the caller must still see oldest-first."""
    backend, executed = postgres_backend
    executed.return_value = [
        {"message_id": f"id-{i}", "role": "user", "content": f"msg-{i}",
         "metadata": {}, "timestamp": float(i)}
        for i in (9, 8, 7)  # DESC order, as the database would return it
    ]

    result = backend.get_all(limit=LIMIT)

    assert [m.content for m in result] == ["msg-7", "msg-8", "msg-9"]
