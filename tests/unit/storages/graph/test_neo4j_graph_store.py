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
        {"labels": ["PERSON", "Entity"], "id": "jane", "name": "Jane", "properties": {}},
        {"labels": ["ORG", "Entity"], "id": "acme", "name": "Acme", "properties": {}},
    ]
    relationships = [
        {
            "type": "WORKS_AT",
            "start_label": "PERSON",
            "end_label": "ORG",
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


def test_write_graph_node_missing_id_raises():
    store, _ = _store([])
    with pytest.raises(ValueError):
        store.write_graph(
            nodes=[{"labels": ["PERSON"], "properties": {"name": "no id here"}}],
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


class _GraphState:
    """A tiny in-memory property graph that EXECUTES the statements ``delete_documents`` emits.

    Applies each statement's documented semantics (provenance-filtered edge match, ``DETACH DELETE`` of
    value nodes, the folded zero-degree orphan sweep) to real state, so the end-to-end test asserts the
    NET EFFECT on the graph — what survives a document's deletion — instead of query strings. Any
    statement shape it does not recognize fails the test.
    """

    def __init__(self, nodes: dict[str, set[str]], edges: list[dict]):
        self.nodes = {node_id: set(labels) for node_id, labels in nodes.items()}
        self.edges = [dict(edge) for edge in edges]  # {"start", "end", "type", "doc"}

    def _degree(self, node_id: str) -> int:
        return sum(1 for edge in self.edges if node_id in (edge["start"], edge["end"]))

    def _sweep(self, candidates: set[str]) -> int:
        """The folded orphan sweep: delete candidate Entity nodes left with zero edges."""
        swept = 0
        for node_id in sorted(candidates):
            if node_id in self.nodes and "Entity" in self.nodes[node_id] and self._degree(node_id) == 0:
                del self.nodes[node_id]
                swept += 1
        return swept

    def execute_query(self, query, parameters_, **kwargs):
        doc_ids = set(parameters_["doc_ids"])
        nodes_deleted = relationships_deleted = 0
        if "DETACH DELETE v" in query:  # doc-scoped value-node statement
            assert "WHERE (n:`Entity`) AND NOT (n)--()" in query  # sweep folded into the same statement
            hits = [e for e in self.edges if e["doc"] in doc_ids and "AttributeValue" in self.nodes[e["end"]]]
            owners = {e["start"] for e in hits}
            for value_id in {e["end"] for e in hits}:  # DETACH: the node and every edge touching it
                relationships_deleted += self._degree(value_id)
                self.edges = [e for e in self.edges if value_id not in (e["start"], e["end"])]
                del self.nodes[value_id]
                nodes_deleted += 1
            nodes_deleted += self._sweep(owners)
        elif "DELETE r" in query:  # edge statement
            assert "WHERE (n:`Entity`) AND NOT (n)--()" in query
            hits = [e for e in self.edges if e["doc"] in doc_ids]
            endpoints = {e["start"] for e in hits} | {e["end"] for e in hits}
            self.edges = [e for e in self.edges if e["doc"] not in doc_ids]
            relationships_deleted += len(hits)
            nodes_deleted += self._sweep(endpoints)
        else:
            raise AssertionError(f"unexpected delete statement:\n{query}")
        return [], _summary(nodes_deleted=nodes_deleted, relationships_deleted=relationships_deleted), []


def test_delete_documents_end_to_end_removes_document_edges_and_orphans():
    # Full document-deletion scenario executed against in-memory graph state. Two documents:
    #   d1: (steve)-[:FOUNDED]->(apple), (apple)-[:MAKES]->(iphone), apple + acme revenue attributes
    #   d2: (apple)-[:MAKES]->(iphone)   — the same fact as d1's, deliberately its own edge
    # Deleting d1 must remove ALL of d1's edges and value nodes, sweep the entities only d1 sustained
    # (steve: relationship-only; acme: attribute-only), and leave everything else untouched.
    graph = _GraphState(
        nodes={
            "apple": {"Entity"},
            "iphone": {"Entity"},
            "steve": {"Entity"},
            "acme": {"Entity"},
            "v_apple_rev": {"AttributeValue"},
            "v_acme_rev": {"AttributeValue"},
            "ghost": {"Entity"},  # edgeless before the delete: the sweep must never touch it
        },
        edges=[
            {"start": "steve", "end": "apple", "type": "FOUNDED", "doc": "d1"},
            {"start": "apple", "end": "iphone", "type": "MAKES", "doc": "d1"},
            {"start": "apple", "end": "v_apple_rev", "type": "HAS_ATTRIBUTE", "doc": "d1"},
            {"start": "acme", "end": "v_acme_rev", "type": "HAS_ATTRIBUTE", "doc": "d1"},
            {"start": "apple", "end": "iphone", "type": "MAKES", "doc": "d2"},
        ],
    )
    client = MagicMock()
    client.execute_query.side_effect = graph.execute_query
    store = Neo4jGraphStore(client=client)

    result = store.delete_documents(["d1"], doc_scoped_labels=["AttributeValue"], orphan_labels=["Entity"])

    # every d1 edge is gone; d2's identical MAKES claim survives untouched
    assert [e["doc"] for e in graph.edges] == ["d2"]
    # d1's value nodes and the entities it alone sustained are gone; shared entities survive through
    # d2's edge; the pre-existing isolated node is never matched by the sweep
    assert set(graph.nodes) == {"apple", "iphone", "ghost"}
    # counters: 4 nodes (2 value nodes + acme + steve), 4 edges (FOUNDED, MAKES d1, 2 HAS_ATTRIBUTE)
    assert result == {"nodes_deleted": 4, "relationships_deleted": 4}


def test_delete_documents_rejects_invalid_orphan_label():
    store, client = _store([])
    with pytest.raises(ValueError, match="Invalid orphan label"):
        store.delete_documents(["d1"], orphan_labels=["bad-label!"])
    assert client.execute_query.call_count == 0  # rejected before any query runs
