import json
from unittest.mock import MagicMock

from dynamiq.storages.graph.age import ApacheAgeGraphStore


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
