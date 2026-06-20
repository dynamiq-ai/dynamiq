import json
from unittest.mock import MagicMock

import pytest

from dynamiq.storages.graph.age import ApacheAgeGraphStore


def _age_store():
    client = MagicMock()
    client.cursor.return_value.__enter__.return_value = MagicMock()
    return ApacheAgeGraphStore(client=client, graph_name="graph")


def test_age_write_graph_is_gated_off():
    # The single-column node builder exists, but AGE writes stay disabled until live-validated.
    store = _age_store()
    assert store.supports_write_graph() is False
    with pytest.raises(NotImplementedError):
        store.write_graph(
            nodes=[{"labels": ["PERSON"], "identity_key": "id", "properties": {"id": "jane"}}],
            relationships=[],
        )


def test_age_node_builder_emits_single_column_unbatched():
    store = _age_store()
    nodes = [
        {"labels": ["PERSON", "Entity"], "identity_key": "id", "properties": {"id": "jane", "name": "Jane"}},
        {"labels": ["ORG"], "identity_key": "id", "properties": {"id": "acme"}},
    ]
    statements = store._build_node_statements(nodes)

    assert len(statements) == 2  # one MERGE per node (NOT batched, unlike Neo4j/Neptune)
    query0, params0 = statements[0]
    assert query0 == (
        "MERGE (n:PERSON:Entity {id: $node_id})\nON CREATE SET n += $node_props\nRETURN n AS result"
    )
    assert params0 == {"node_id": "jane", "node_props": {"id": "jane", "name": "Jane"}}

    # Edge builder is inherited -> its single-column ``RETURN r`` is already AGE-compatible.
    edge_statements = store._build_relationship_statements(
        [{"type": "KNOWS", "start_label": "PERSON", "end_label": "PERSON",
          "start_identity": "a", "end_identity": "b", "properties": {}}]
    )
    assert edge_statements[0][0].endswith("RETURN r")


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
