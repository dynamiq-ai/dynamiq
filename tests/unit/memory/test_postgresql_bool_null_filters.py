"""PostgreSQL search filters must match booleans and None as JSONB stores them."""

from types import SimpleNamespace

import pytest

from dynamiq.memory.backends.postgresql import PostgreSQL


def _where(filters):
    # No database connection is needed to build the clause.
    clause, params = PostgreSQL._build_where_clause(
        SimpleNamespace(metadata_col="metadata", content_col="content"), None, filters
    )
    return clause.as_string(None), params


@pytest.mark.parametrize("value, text", [(True, "true"), (False, "false")])
def test_boolean_filter_uses_jsonb_text(value, text):
    _, params = _where({"pinned": value})

    assert params == ["pinned", text]


def test_boolean_list_filter_uses_jsonb_text():
    _, params = _where({"pinned": [True, False]})

    assert params == ["pinned", "true", "false"]


def test_none_filter_uses_is_null():
    sql, params = _where({"user_id": None})

    assert "IS NULL" in sql
    assert " = " not in sql
    assert params == ["user_id"]


def test_list_with_none_matches_values_or_null():
    sql, params = _where({"user_id": ["u1", None], "pinned": True})

    assert 'WHERE ("metadata"->>%s IN (%s) OR "metadata"->>%s IS NULL) AND ' in sql
    assert params == ["user_id", "u1", "user_id", "pinned", "true"]
