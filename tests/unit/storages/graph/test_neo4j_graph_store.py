"""Unit tests for the shared write_graph path as exercised by Neo4jGraphStore (no live DB).

Characterization test: locks the exact Cypher the shared ``BaseGraphStore.write_graph`` builders emit and
the counter totals Neo4j reads back, so the base-class refactor cannot silently change Neo4j behavior.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from dynamiq.storages.graph.neo4j import Neo4jGraphStore


def _summary(*, nodes_created=0, relationships_created=0, properties_set=0):
    """A fake Neo4j ResultSummary with the counters write_graph reads."""
    return SimpleNamespace(
        counters=SimpleNamespace(
            nodes_created=nodes_created,
            relationships_created=relationships_created,
            properties_set=properties_set,
        )
    )


def _store(execute_query_side_effect):
    client = MagicMock()
    client.execute_query.side_effect = execute_query_side_effect
    return Neo4jGraphStore(client=client), client


def test_write_graph_batches_nodes_and_writes_edges():
    nodes = [
        {"labels": ["PERSON", "Entity"], "identity_key": "id", "properties": {"id": "jane", "name": "Jane"}},
        {"labels": ["ORG", "Entity"], "identity_key": "id", "properties": {"id": "acme", "name": "Acme"}},
    ]
    relationships = [
        {
            "type": "WORKS_AT",
            "start_label": "PERSON",
            "end_label": "ORG",
            "start_identity_key": "id",
            "end_identity_key": "id",
            "start_identity": "jane",
            "end_identity": "acme",
            "properties": {"source_doc_id": "doc-1", "role": "CFO"},
            "identity_keys": ["source_doc_id"],
        }
    ]
    # Node statement reports relationships_created=0; edge statement reports nodes_created=0 -- the
    # invariant that makes "add all three counters every statement" equal the old split tally.
    node_result = ([], _summary(nodes_created=2, relationships_created=0, properties_set=4), ["n0", "n1"])
    edge_result = ([], _summary(nodes_created=0, relationships_created=1, properties_set=2), ["r"])
    store, client = _store([node_result, edge_result])

    result = store.write_graph(nodes=nodes, relationships=relationships)

    # --- one batched node query, one per-edge query ---
    assert client.execute_query.call_count == 2
    node_call, edge_call = client.execute_query.call_args_list

    node_query = node_call.args[0]
    assert node_query == (
        "MERGE (n0:PERSON:Entity {id: $node_0_id})\n"
        "ON CREATE SET n0 += $node_0_props\n"
        "MERGE (n1:ORG:Entity {id: $node_1_id})\n"
        "ON CREATE SET n1 += $node_1_props\n"
        "RETURN n0, n1"
    )
    assert node_call.kwargs["parameters_"] == {
        "node_0_props": {"id": "jane", "name": "Jane"},
        "node_0_id": "jane",
        "node_1_props": {"id": "acme", "name": "Acme"},
        "node_1_id": "acme",
    }

    edge_query = edge_call.args[0]
    assert edge_query == (
        "MATCH (s:PERSON {id: $start_id})\n"
        "MATCH (e:ORG {id: $end_id})\n"
        "MERGE (s)-[r:WORKS_AT {source_doc_id: $rkey_0_0}]->(e)\n"
        "SET r += $props\n"
        "RETURN r"
    )
    assert edge_call.kwargs["parameters_"] == {
        "start_id": "jane",
        "end_id": "acme",
        "props": {"source_doc_id": "doc-1", "role": "CFO"},
        "rkey_0_0": "doc-1",
    }

    # --- aggregated stats: node + edge summaries summed ---
    assert result["nodes_created"] == 2
    assert result["relationships_created"] == 1
    assert result["properties_set"] == 6  # 4 (nodes) + 2 (edge)
    assert result["keys"] == ["r"]  # last statement's keys


def test_write_graph_edge_without_identity_keys_omits_merge_props():
    relationships = [
        {
            "type": "KNOWS",
            "start_label": "PERSON",
            "end_label": "PERSON",
            "start_identity": "a",
            "end_identity": "b",
            "properties": {},
        }
    ]
    store, client = _store([([], _summary(relationships_created=1), ["r"])])

    store.write_graph(nodes=[], relationships=relationships)

    edge_query = client.execute_query.call_args_list[0].args[0]
    assert "MERGE (s)-[r:KNOWS]->(e)" in edge_query  # no {merge-key} block


def test_write_graph_empty_payload_raises():
    store, _ = _store([])
    with pytest.raises(ValueError):
        store.write_graph(nodes=[], relationships=[])


def test_write_graph_node_missing_identity_key_raises():
    store, _ = _store([])
    with pytest.raises(ValueError):
        store.write_graph(
            nodes=[{"labels": ["PERSON"], "identity_key": "id", "properties": {"name": "no id here"}}],
            relationships=[],
        )


def test_supports_write_graph_flag():
    store, _ = _store([])
    assert store.supports_write_graph() is True
