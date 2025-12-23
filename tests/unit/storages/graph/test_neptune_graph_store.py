from unittest.mock import MagicMock

from dynamiq.storages.graph.neptune import NeptuneGraphStore


def test_neptune_run_cypher_with_params():
    client = MagicMock()
    response = MagicMock()
    response.json.return_value = {"results": [{"name": "Ada"}]}
    client.post.return_value = response

    store = NeptuneGraphStore(client=client, endpoint="https://localhost:8182/openCypher", verify_ssl=False)
    records, summary, keys = store.run_cypher("MATCH (n) RETURN n", parameters={"name": "Ada"})

    assert records == [{"name": "Ada"}]
    assert summary["query"] == "MATCH (n) RETURN n"
    assert keys == []
    client.post.assert_called_once()


def test_neptune_introspect_schema():
    client = MagicMock()
    response_labels = MagicMock()
    response_labels.json.return_value = {"results": [{"labels": ["Person"]}]}
    response_rels = MagicMock()
    response_rels.json.return_value = {"results": [{"type": "KNOWS"}]}
    response_node_props = MagicMock()
    response_node_props.json.return_value = {"results": [{"props": {"name": "Ada"}}]}
    response_rel_props = MagicMock()
    response_rel_props.json.return_value = {"results": [{"props": {"since": 2020}}]}
    response_triples = MagicMock()
    response_triples.json.return_value = {
        "results": [{"result": {"from": ["Person"], "edge": "KNOWS", "to": ["Person"]}}]
    }
    client.post.side_effect = [
        response_labels,
        response_rels,
        response_node_props,
        response_rel_props,
        response_triples,
    ]

    store = NeptuneGraphStore(client=client, endpoint="https://localhost:8182/openCypher", verify_ssl=False)
    schema = store.introspect_schema(include_properties=True)

    assert schema["labels"] == ["Person"]
    assert schema["relationship_types"] == ["KNOWS"]
    assert schema["node_properties"][0]["labels"] == "Person"
    assert schema["node_properties"][0]["properties"][0]["property"] == "name"
    assert schema["relationship_properties"][0]["type"] == "KNOWS"
    assert schema["relationship_properties"][0]["properties"][0]["property"] == "since"
    assert schema["relationships"] == ["(:`Person`)-[:`KNOWS`]->(:`Person`)"]
