"""Unit tests for the shared write_graph path as exercised by Neo4jGraphStore (no live DB).

Characterization test: locks the exact Cypher Neo4j's ``UNWIND`` bulk builders emit (node upserts grouped
by label-set, edge upserts grouped by structure) and the counter totals Neo4j reads back, so a refactor
cannot silently change Neo4j write behavior.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from dynamiq.storages.graph.neo4j import Neo4jGraphStore


def _summary(*, nodes_created=0, relationships_created=0, properties_set=0, nodes_deleted=0, relationships_deleted=0):
    """A fake Neo4j ResultSummary with the counters write_graph / delete_documents read."""
    return SimpleNamespace(
        counters=SimpleNamespace(
            nodes_created=nodes_created,
            relationships_created=relationships_created,
            properties_set=properties_set,
            nodes_deleted=nodes_deleted,
            relationships_deleted=relationships_deleted,
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


def test_delete_documents_removes_doc_scoped_nodes_and_edges_keeps_entities():
    # Two statements: DETACH DELETE the doc-scoped AttributeValue nodes (with their edges), then DELETE
    # the remaining per-document edges. Entity nodes are never matched by either query.
    value_del = ([], _summary(nodes_deleted=2, relationships_deleted=2), [])
    edge_del = ([], _summary(relationships_deleted=1), [])
    store, client = _store([value_del, edge_del])

    result = store.delete_documents(["d1", "d2"], doc_scoped_labels=["AttributeValue"])

    assert client.execute_query.call_count == 2
    value_call, edge_call = client.execute_query.call_args_list
    assert value_call.args[0] == (
        "MATCH ()-[r]->(v:`AttributeValue`)\n"
        "WHERE r.source_doc_id IN $doc_ids\n"
        "DETACH DELETE v"
    )
    assert value_call.kwargs["parameters_"] == {"doc_ids": ["d1", "d2"]}
    assert edge_call.args[0] == "MATCH ()-[r]->() WHERE r.source_doc_id IN $doc_ids DELETE r"
    assert edge_call.kwargs["parameters_"] == {"doc_ids": ["d1", "d2"]}
    # nodes: 2 from the value statement; relationships: 2 (value edges) + 1 (entity-entity edge).
    assert result == {"nodes_deleted": 2, "relationships_deleted": 3}


def test_delete_documents_empty_ids_is_noop():
    store, client = _store([])
    result = store.delete_documents([], doc_scoped_labels=["AttributeValue"])
    assert client.execute_query.call_count == 0
    assert result == {"nodes_deleted": 0, "relationships_deleted": 0}


def test_delete_documents_with_no_doc_scoped_labels_only_deletes_edges():
    store, client = _store([([], _summary(relationships_deleted=4), [])])

    result = store.delete_documents(["d1"])

    assert client.execute_query.call_count == 1  # just the edge-delete statement
    assert client.execute_query.call_args_list[0].args[0] == (
        "MATCH ()-[r]->() WHERE r.source_doc_id IN $doc_ids DELETE r"
    )
    assert result == {"nodes_deleted": 0, "relationships_deleted": 4}


def test_delete_documents_rejects_invalid_label():
    store, _ = _store([])
    with pytest.raises(ValueError, match="Invalid document-scoped label"):
        store.delete_documents(["d1"], doc_scoped_labels=["bad-label!"])


def test_delete_documents_custom_provenance_key():
    # e.g. deleting by knowledgebase file id: edges carry r.file_id via flattened chunk metadata.
    store, client = _store([([], _summary(relationships_deleted=2), [])])

    result = store.delete_documents(["f1"], provenance_key="file_id")

    assert client.execute_query.call_args_list[0].args[0] == ("MATCH ()-[r]->() WHERE r.file_id IN $doc_ids DELETE r")
    assert result == {"nodes_deleted": 0, "relationships_deleted": 2}


def test_delete_documents_rejects_injection_shaped_provenance_key():
    store, client = _store([])
    with pytest.raises(ValueError, match="Invalid provenance key"):
        store.delete_documents(["d1"], provenance_key="file_id IS NOT NULL OR true //")
    assert client.execute_query.call_count == 0  # rejected before any query runs
