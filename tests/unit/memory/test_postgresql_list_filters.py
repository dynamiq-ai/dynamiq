"""PostgreSQL search filters must treat list values as membership.

`_build_where_clause` compared `metadata->>key` to `str(value)`. A list such as
`session_id=["s-a", "s-b"]` therefore looked for the text `"['s-a', 's-b']"` and
matched nothing. InMemory, SQLite and DynamoDB treat a list as "the stored value
is one of these".
"""

from types import SimpleNamespace

from dynamiq.memory.backends.postgresql import PostgreSQL


def _where(query, filters):
    # The method only reads the column names, so no database connection is needed.
    clause, params = PostgreSQL._build_where_clause(
        SimpleNamespace(metadata_col="metadata", content_col="content"),
        query,
        filters,
    )
    return clause.as_string(None), params


def test_list_filter_uses_membership():
    sql, params = _where(None, {"session_id": ["s-a", "s-b"]})

    assert " IN (%s, %s)" in sql
    assert params == ["session_id", "s-a", "s-b"]


def test_scalar_filter_still_uses_equality():
    sql, params = _where(None, {"session_id": "s-a"})

    assert " IN " not in sql
    assert params == ["session_id", "s-a"]


def test_list_values_are_compared_as_text():
    # metadata->>key returns text, the same as the scalar path's str(value).
    sql, params = _where(None, {"n": [1, 2]})

    assert " IN (%s, %s)" in sql
    assert params == ["n", "1", "2"]


def test_empty_list_matches_nothing_without_an_empty_in():
    # IN () is a syntax error. An empty membership matches no row.
    sql, params = _where(None, {"session_id": []})

    assert "IN ()" not in sql
    assert "FALSE" in sql
    assert params == []


def test_list_filter_is_anded_with_the_query_and_other_filters():
    sql, params = _where("hello", {"session_id": ["s-a", "s-b"], "user_id": "u1"})

    assert " IN (%s, %s)" in sql
    assert "ILIKE" in sql
    assert " AND " in sql
    assert params == ["session_id", "s-a", "s-b", "user_id", "u1", "%hello%"]
