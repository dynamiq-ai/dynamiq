"""Unit tests for ``KnowledgeGraphWriter`` write-time entity resolution (no LLM, no graph DB).

Identity contract: LLM-produced ids are intra-extraction wiring only — node identity is decided
exclusively by trigram name similarity against existing same-label nodes. Match -> adopt the
existing node's id; no match -> fresh UUID.
"""

import uuid

from dynamiq.connections import Neo4j
from dynamiq.nodes.extractors import KnowledgeGraphWriter, Ontology
from dynamiq.nodes.extractors.entity_extractor import ATTRIBUTE_VALUE_LABEL, HAS_ATTRIBUTE_TYPE

from .test_entity_extractor import StubLLM


def make_writer(existing: dict[str, list[tuple[str, str]]]) -> KnowledgeGraphWriter:
    """Writer with a pre-populated per-call candidate cache, so no store is ever queried."""
    writer = KnowledgeGraphWriter(
        llm=StubLLM(),
        connection=Neo4j(uri="bolt://localhost:7687", username="neo4j", password="password"),
        ontology=Ontology(entity_types=[], relationship_types=[]),  # required; unused by resolution tests
        is_postponed_component_init=True,
    )
    writer._existing_cache = existing
    return writer


def entity(label: str, entity_id: str, name: str) -> dict:
    return {"labels": [label], "identity_key": "id", "properties": {"id": entity_id, "name": name}}


def edge(rel_type: str, start_label: str, end_label: str, start_id: str, end_id: str, props: dict = None) -> dict:
    return {
        "type": rel_type,
        "start_label": start_label,
        "end_label": end_label,
        "start_identity_key": "id",
        "end_identity_key": "id",
        "start_identity": start_id,
        "end_identity": end_id,
        "properties": props or {},
    }


def is_uuid(value: str) -> bool:
    try:
        uuid.UUID(value)
        return True
    except ValueError:
        return False


class TestResolveAgainstGraph:
    def test_new_entity_gets_uuid_not_llm_id(self):
        writer = make_writer({"ORGANIZATION": []})
        nodes = [entity("ORGANIZATION", "acme", "Acme Capital")]
        rels = [edge("USES", "ORGANIZATION", "SYSTEM", "acme", "tradingx")]

        resolved_nodes, resolved_rels = writer._resolve_against_graph(nodes, rels)

        new_id = resolved_nodes[0]["properties"]["id"]
        assert new_id != "acme" and is_uuid(new_id)
        # the edge endpoint followed the entity onto its new id
        assert resolved_rels[0]["start_identity"] == new_id

    def test_matches_existing_by_name_despite_different_id(self):
        writer = make_writer({"ORGANIZATION": [("uuid-existing", "Acme Capital")]})
        nodes = [entity("ORGANIZATION", "acme-capital-llc", "Acme Capital LLC")]
        rels = [edge("WORKS_AT", "PERSON", "ORGANIZATION", "jane", "acme-capital-llc")]

        resolved_nodes, resolved_rels = writer._resolve_against_graph(nodes, rels)

        assert resolved_nodes[0]["properties"]["id"] == "uuid-existing"
        assert resolved_rels[0]["end_identity"] == "uuid-existing"

    def test_llm_id_never_participates_in_identity(self):
        # An existing node whose ID equals the LLM id but whose NAME is dissimilar must NOT capture
        # the new entity: identity is name-only.
        writer = make_writer({"ORGANIZATION": [("acme", "Globex Corporation")]})
        nodes = [entity("ORGANIZATION", "acme", "Acme Capital")]

        resolved_nodes, _ = writer._resolve_against_graph(nodes, [])

        new_id = resolved_nodes[0]["properties"]["id"]
        assert new_id != "acme" and is_uuid(new_id)

    def test_same_batch_entities_converge_by_name(self):
        # Two documents surfaced the same org under different LLM ids -> one node, one id.
        writer = make_writer({"ORGANIZATION": []})
        nodes = [
            entity("ORGANIZATION", "acme", "Acme Capital"),
            entity("ORGANIZATION", "acme-capital", "Acme Capital"),
        ]
        rels = [
            edge("USES", "ORGANIZATION", "SYSTEM", "acme", "tradingx"),
            edge("USES", "ORGANIZATION", "SYSTEM", "acme-capital", "tradingx"),
        ]

        resolved_nodes, resolved_rels = writer._resolve_against_graph(nodes, rels)

        assert len(resolved_nodes) == 1
        node_id = resolved_nodes[0]["properties"]["id"]
        assert {r["start_identity"] for r in resolved_rels} == {node_id}

    def test_dissimilar_names_stay_separate(self):
        writer = make_writer({"PERSON": []})
        nodes = [entity("PERSON", "jane", "Jane Doe"), entity("PERSON", "john", "John Smith")]

        resolved_nodes, _ = writer._resolve_against_graph(nodes, [])

        ids = {n["properties"]["id"] for n in resolved_nodes}
        assert len(ids) == 2

    def test_attribute_value_ids_rederived_from_owner(self):
        # Value ids are "{wiring_id}::{key}::{doc_id}"; only the owner (wiring) segment is remapped,
        # so values from different documents keep distinct ids after resolution.
        writer = make_writer({"PERSON": [("uuid-jane", "Jane Doe")]})
        nodes = [
            entity("PERSON", "jane@d1", "Jane Doe"),
            {
                "labels": [ATTRIBUTE_VALUE_LABEL],
                "identity_key": "id",
                "properties": {"id": "jane@d1::salary::d1", "value": "$250,000"},
            },
        ]
        rels = [
            edge(
                HAS_ATTRIBUTE_TYPE, "PERSON", ATTRIBUTE_VALUE_LABEL, "jane@d1", "jane@d1::salary::d1",
                {"key": "salary"},
            )
        ]

        resolved_nodes, resolved_rels = writer._resolve_against_graph(nodes, rels)

        value_node = next(n for n in resolved_nodes if n["labels"][0] == ATTRIBUTE_VALUE_LABEL)
        assert value_node["properties"]["id"] == "uuid-jane::salary::d1"
        assert resolved_rels[0]["start_identity"] == "uuid-jane"
        assert resolved_rels[0]["end_identity"] == "uuid-jane::salary::d1"

    def test_same_llm_id_from_different_documents_does_not_alias(self):
        # Two documents used the same LLM slug for DIFFERENT people: doc-scoped wiring ids keep them
        # apart, and name resolution gives each its own identity.
        writer = make_writer({"PERSON": []})
        nodes = [
            entity("PERSON", "mercury@d1", "Mercury Johnson"),
            entity("PERSON", "mercury@d2", "Mercury Lee"),
        ]

        resolved_nodes, _ = writer._resolve_against_graph(nodes, [])

        assert len({n["properties"]["id"] for n in resolved_nodes}) == 2
