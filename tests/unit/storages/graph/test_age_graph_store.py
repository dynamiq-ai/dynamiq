import json
from unittest.mock import MagicMock

import pytest

from dynamiq.storages.graph.age import ApacheAgeGraphStore


def _age_store():
    client = MagicMock()
    client.cursor.return_value.__enter__.return_value = MagicMock()
    return ApacheAgeGraphStore(client=client, graph_name="graph")


def test_age_write_graph_is_gated_off():
    # write_graph is implemented only on the Neo4j store; AGE does not support writing and raises.
    store = _age_store()
    assert store.supports_write_graph() is False
    with pytest.raises(NotImplementedError):
        store.write_graph(
            nodes=[{"labels": ["PERSON"], "identity_key": "id", "properties": {"id": "jane"}}],
            relationships=[],
        )


def test_age_run_cypher_executes_query():
    client = MagicMock()
    cursor = MagicMock()
    client.cursor.return_value.__enter__.return_value = cursor
    cursor.fetchall.return_value = [{"result": {"name": "Ada"}}]

    store = ApacheAgeGraphStore(client=client, graph_name="graph")

    records, summary, keys = store.run_cypher("MATCH (n) RETURN n AS result", parameters={"name": "Ada"})

    assert records == [{"name": "Ada"}]
    assert summary["query"] == "MATCH (n) RETURN n AS result"
    assert keys == []

    last_call = cursor.execute.call_args_list[-1]
    sql, params = last_call.args
    assert "cypher" in sql
    assert "agtype_to_json" in sql
    assert params[0] == "graph"
    assert params[1] == "MATCH (n) RETURN n AS result"
    assert params[2] == json.dumps({"name": "Ada"})
