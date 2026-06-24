"""Unit tests for the shared write_graph path as exercised by Neo4jGraphStore (no live DB).

Characterization test: locks the exact Cypher Neo4j's ``UNWIND`` bulk builders emit (node upserts grouped
by label-set, edge upserts grouped by structure) and the counter totals Neo4j reads back, so a refactor
cannot silently change Neo4j write behavior.
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
    # Neo4j writes each label-group with one UNWIND statement (PERSON:Entity, ORG:Entity) and the edge
    # with one UNWIND statement -> three statements. Each reports its own counters; summing all three on
    # every statement still equals the totals (node statements report relationships_created=0, the edge
    # statement reports nodes_created=0).
    person_result = ([], _summary(nodes_created=1, relationships_created=0, properties_set=2), ["n"])
    org_result = ([], _summary(nodes_created=1, relationships_created=0, properties_set=2), ["n"])
    edge_result = ([], _summary(nodes_created=0, relationships_created=1, properties_set=2), ["r"])
    store, client = _store([person_result, org_result, edge_result])

    result = store.write_graph(nodes=nodes, relationships=relationships)

    # --- one UNWIND node query per label-group, one UNWIND edge query per structural group ---
    assert client.execute_query.call_count == 3
    person_call, org_call, edge_call = client.execute_query.call_args_list

    assert person_call.args[0] == (
        "UNWIND $rows AS row\n"
        "MERGE (n:PERSON:Entity {id: row.id})\n"
        "ON CREATE SET n += row.props\n"
        "RETURN n"
    )
    assert person_call.kwargs["parameters_"] == {
        "rows": [{"id": "jane", "props": {"id": "jane", "name": "Jane"}}]
    }
    assert org_call.args[0] == (
        "UNWIND $rows AS row\n"
        "MERGE (n:ORG:Entity {id: row.id})\n"
        "ON CREATE SET n += row.props\n"
        "RETURN n"
    )
    assert org_call.kwargs["parameters_"] == {
        "rows": [{"id": "acme", "props": {"id": "acme", "name": "Acme"}}]
    }

    assert edge_call.args[0] == (
        "UNWIND $rows AS row\n"
        "MATCH (s:PERSON {id: row.start_id})\n"
        "MATCH (e:ORG {id: row.end_id})\n"
        "MERGE (s)-[r:WORKS_AT {source_doc_id: row.source_doc_id}]->(e)\n"
        "SET r += row.props\n"
        "RETURN r"
    )
    assert edge_call.kwargs["parameters_"] == {
        "rows": [
            {
                "start_id": "jane",
                "end_id": "acme",
                "props": {"source_doc_id": "doc-1", "role": "CFO"},
                "source_doc_id": "doc-1",
            }
        ]
    }

    # --- aggregated stats: all statement summaries summed ---
    assert result["nodes_created"] == 2
    assert result["relationships_created"] == 1
    assert result["properties_set"] == 6  # 2 + 2 (nodes) + 2 (edge)
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
