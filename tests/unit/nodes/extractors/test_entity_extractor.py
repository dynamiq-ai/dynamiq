"""Unit tests for ``EntityExtractor`` transform/sanitize logic (no LLM, no Neo4j)."""

import json
from typing import ClassVar

import pytest

from dynamiq.nodes.extractors import EntityExtractor, Ontology
from dynamiq.nodes.node import Node, NodeGroup
from dynamiq.storages.graph.neo4j.neo4j import Neo4jGraphStore
from dynamiq.types import Document

# ``ontology`` is required on EntityExtractor; this permissive schema covers the types the tests emit so
# enforcement keeps them. Tests that exercise attribute reification or guidance pass their own ontology.
_ONTOLOGY = Ontology(entity_types=["Hedge Fund", "Person", "Org"], relationship_types=["works at"])


class StubLLM(Node):
    """Minimal Node that returns a canned ``content`` payload instead of calling a real LLM."""

    group: ClassVar = NodeGroup.LLMS
    name: str = "stub-llm"
    response_content: str = "{}"

    def execute(self, input_data, config=None, **kwargs):
        return {"content": self.response_content}


class SequenceStubLLM(Node):
    """Stub LLM that returns one canned response per call, in order (for recovery tests)."""

    group: ClassVar = NodeGroup.LLMS
    name: str = "sequence-stub-llm"
    responses: list[str] = []
    calls: int = 0

    def execute(self, input_data, config=None, **kwargs):
        content = self.responses[min(self.calls, len(self.responses) - 1)]
        self.calls += 1
        return {"content": content}


def _assert_valid_for_neo4j(nodes, relationships):
    """Every label/type/key produced must pass Neo4jGraphStore's validators."""
    for node in nodes:
        Neo4jGraphStore._format_labels(node["labels"])
        Neo4jGraphStore._format_property_key(node["identity_key"])
    for rel in relationships:
        Neo4jGraphStore._format_relationship_type(rel["type"])
        Neo4jGraphStore._format_single_label(rel["start_label"])
        Neo4jGraphStore._format_single_label(rel["end_label"])


class TestSanitizeIdentifier:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("Hedge Fund", "HEDGE_FUND"),
            ("works at", "WORKS_AT"),
            ("person", "PERSON"),
            ("  spaced  ", "SPACED"),
            ("multi--dash__name", "MULTI_DASH_NAME"),
            ("2foo", "_2FOO"),  # leading digit gets prefixed
            ("***", "ENTITY"),  # nothing valid left -> fallback
            ("", "ENTITY"),
        ],
    )
    def test_sanitize(self, raw, expected):
        assert EntityExtractor._sanitize_identifier(raw) == expected

    def test_output_always_valid_for_neo4j(self):
        for raw in ["Hedge Fund", "1abc", "a.b.c", "EVENT/2024", "Ünïcödé"]:
            assert _IDENTIFIER_MATCHES(EntityExtractor._sanitize_identifier(raw))


def _IDENTIFIER_MATCHES(value: str) -> bool:
    import re

    return re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", value) is not None


class TestToWriteGraphPayload:
    def test_builds_write_graph_shape_and_drops_dangling(self):
        entities = [
            # ``aum`` is an undeclared inline property: nodes are identity-only, so it must be dropped.
            {"id": "acme", "type": "Hedge Fund", "name": "Acme Capital", "properties": {"aum": "10B"}},
            {"id": "jane", "type": "person", "name": "Jane Doe"},
        ]
        relationships = [
            {"source_id": "jane", "target_id": "acme", "type": "works at", "properties": {"since": 2020}},
            {"source_id": "jane", "target_id": "ghost", "type": "KNOWS"},  # dangling -> dropped
        ]
        extractor = EntityExtractor(llm=StubLLM(), ontology=_ONTOLOGY)
        nodes, graph_rels = extractor._to_write_graph_payload(entities, relationships)

        # Every entity carries its type label first plus the shared "Entity" label (the full-text index hook),
        # and ONLY identity properties (id + name) -- the emitted ``aum`` is dropped, never inlined on the node.
        assert nodes == [
            {
                "labels": ["HEDGE_FUND", "Entity"],
                "identity_key": "id",
                "properties": {"id": "acme", "name": "Acme Capital"},
            },
            {"labels": ["PERSON", "Entity"], "identity_key": "id", "properties": {"id": "jane", "name": "Jane Doe"}},
        ]
        assert graph_rels == [
            {
                "type": "WORKS_AT",
                "start_label": "PERSON",
                "end_label": "HEDGE_FUND",
                "start_identity_key": "id",
                "end_identity_key": "id",
                "start_identity": "jane",
                "end_identity": "acme",
                "properties": {"since": 2020},
            }
        ]
        _assert_valid_for_neo4j(nodes, graph_rels)

    def test_entity_without_type_or_id_is_skipped(self):
        entities = [{"id": "x"}, {"type": "Person"}, {"id": "y", "type": "Person"}]
        nodes, _ = EntityExtractor(llm=StubLLM(), ontology=_ONTOLOGY)._to_write_graph_payload(entities, [])
        assert [n["properties"]["id"] for n in nodes] == ["y"]

    def test_relationship_description_rides_on_edge_properties(self):
        entities = [{"id": "jane", "type": "Person", "name": "Jane"}, {"id": "acme", "type": "Org", "name": "Acme"}]
        relationships = [
            {"source_id": "jane", "target_id": "acme", "type": "WORKS_AT", "description": "CFO since 2020"}
        ]
        _, graph_rels = EntityExtractor(llm=StubLLM(), ontology=_ONTOLOGY)._to_write_graph_payload(entities, relationships)
        assert graph_rels[0]["properties"]["description"] == "CFO since 2020"

    def test_entity_description_is_not_written_to_the_node(self):
        # Descriptions are edge-only: an entity-level description must never land on the (shared) node.
        entities = [{"id": "jane", "type": "Person", "name": "Jane", "description": "the CFO"}]
        nodes, _ = EntityExtractor(llm=StubLLM(), ontology=_ONTOLOGY)._to_write_graph_payload(entities, [])
        assert nodes[0]["properties"] == {"id": "jane", "name": "Jane"}


class TestTypeGuidance:
    def test_ontology_type_descriptions_annotate_the_guidance(self):
        ontology = Ontology(
            entity_types=["Person", "Org"],
            relationship_types=["WORKS_AT"],
            entity_descriptions={"Person": "an individual human"},  # Org intentionally undescribed
            relationship_descriptions={"WORKS_AT": "employment of a person by an organization"},
        )
        guidance = EntityExtractor(llm=StubLLM(), ontology=ontology)._build_type_guidance()

        assert "Person (an individual human)" in guidance
        assert "Org" in guidance and "Org (" not in guidance  # bare when no description
        assert "WORKS_AT (employment of a person by an organization)" in guidance


class TestParseLLMJson:
    def test_plain_json(self):
        assert EntityExtractor._parse_llm_json('{"entities": [], "relationships": []}') == {
            "entities": [],
            "relationships": [],
        }

    def test_markdown_fenced(self):
        raw = '```json\n{"entities": [{"id": "a"}], "relationships": []}\n```'
        assert EntityExtractor._parse_llm_json(raw)["entities"] == [{"id": "a"}]

    def test_prose_around_json(self):
        raw = 'Here you go:\n{"entities": [], "relationships": []}\nHope that helps!'
        assert EntityExtractor._parse_llm_json(raw) == {"entities": [], "relationships": []}

    def test_garbage_returns_none(self):
        assert EntityExtractor._parse_llm_json("not json at all") is None

    def test_none_returns_none(self):
        assert EntityExtractor._parse_llm_json(None) is None

    def test_non_dict_json_returns_none(self):
        assert EntityExtractor._parse_llm_json('[{"id": "a"}]') is None


class TestExecuteEndToEndWithStubLLM:
    def test_execute_transforms_stub_llm_output(self):
        payload = {
            "entities": [
                {"id": "acme", "type": "Hedge Fund", "name": "Acme Capital"},
                {"id": "jane", "type": "Person", "name": "Jane Doe"},
            ],
            "relationships": [{"source_id": "jane", "target_id": "acme", "type": "works at"}],
        }
        llm = StubLLM(response_content=json.dumps(payload))
        extractor = EntityExtractor(llm=llm, ontology=_ONTOLOGY)

        result = extractor.execute(
            EntityExtractor.input_schema(documents=[Document(content="Jane Doe works at Acme Capital.")])
        )

        assert {n["labels"][0] for n in result["nodes"]} == {"HEDGE_FUND", "PERSON"}
        assert len(result["relationships"]) == 1
        assert result["relationships"][0]["type"] == "WORKS_AT"
        _assert_valid_for_neo4j(result["nodes"], result["relationships"])

    def test_execute_recovers_from_unparseable_output(self):
        payload = {
            "entities": [{"id": "acme", "type": "Org", "name": "Acme"}],
            "relationships": [],
        }
        llm = SequenceStubLLM(responses=["sorry, here is the graph you asked for", json.dumps(payload)])
        extractor = EntityExtractor(llm=llm, ontology=_ONTOLOGY)

        result = extractor.execute(EntityExtractor.input_schema(documents=[Document(content="Acme.")]))

        assert llm.calls == 2  # initial call + one repair round-trip
        assert [n["properties"]["name"] for n in result["nodes"]] == ["Acme"]

    def test_execute_skips_document_when_recovery_fails(self):
        llm = SequenceStubLLM(responses=["garbage", "still garbage"])
        extractor = EntityExtractor(llm=llm, ontology=_ONTOLOGY)

        result = extractor.execute(EntityExtractor.input_schema(documents=[Document(content="Acme.")]))

        assert llm.calls == 2
        assert result["nodes"] == []
        assert result["relationships"] == []

    def test_execute_promotes_declared_attributes_to_doc_scoped_value_nodes(self):
        from dynamiq.nodes.extractors import Ontology
        from dynamiq.nodes.extractors.entity_extractor import ATTRIBUTE_VALUE_LABEL, HAS_ATTRIBUTE_TYPE

        payload = {
            "entities": [{"id": "jane", "type": "Person", "name": "Jane Doe", "properties": {"salary": "$250,000"}}],
            "relationships": [],
        }
        ontology = Ontology(entity_types=["Person"], relationship_types=[], attributes={"Person": ["salary"]})
        extractor = EntityExtractor(llm=StubLLM(response_content=json.dumps(payload)), ontology=ontology)
        document = Document(id="doc-1", content="Jane Doe's salary is $250,000.")

        result = extractor.execute(EntityExtractor.input_schema(documents=[document]))

        person = next(n for n in result["nodes"] if n["labels"][0] == "PERSON")
        assert person["properties"]["id"] == "jane@doc-1"  # wiring id is doc-scoped
        assert "salary" not in person["properties"]  # promoted to an edge, not a node property

        value_node = next(n for n in result["nodes"] if n["labels"][0] == ATTRIBUTE_VALUE_LABEL)
        assert value_node["properties"] == {"id": "jane@doc-1::salary::doc-1", "value": "$250,000"}

        attr_edge = next(r for r in result["relationships"] if r["type"] == HAS_ATTRIBUTE_TYPE)
        assert attr_edge["start_identity"] == "jane@doc-1"
        assert attr_edge["end_identity"] == "jane@doc-1::salary::doc-1"
        assert attr_edge["properties"]["key"] == "salary"

    def test_execute_stamps_doc_discriminator_on_relationships(self):
        # Each relationship carries a scalar source_doc_id + identity_keys so the store keeps the SAME
        # fact from different documents as separate edges (no ACL overwrite on merge).
        payload = {
            "entities": [
                {"id": "jane", "type": "Person", "name": "Jane Doe"},
                {"id": "acme", "type": "Org", "name": "Acme"},
            ],
            "relationships": [{"source_id": "jane", "target_id": "acme", "type": "works at"}],
        }
        extractor = EntityExtractor(llm=StubLLM(response_content=json.dumps(payload)), ontology=_ONTOLOGY)
        result = extractor.execute(
            EntityExtractor.input_schema(documents=[Document(id="doc-1", content="Jane works at Acme.")])
        )

        rel = next(r for r in result["relationships"] if r["type"] == "WORKS_AT")
        assert rel["properties"]["source_doc_id"] == "doc-1"
        assert rel["properties"]["source_doc_ids"] == ["doc-1"]
        assert rel["identity_keys"] == ["source_doc_id"]

    def test_execute_emits_doc_scoped_nodes_per_document(self):
        payload = {
            "entities": [{"id": "acme", "type": "Org", "name": "Acme"}],
            "relationships": [],
        }
        llm = StubLLM(response_content=json.dumps(payload))
        extractor = EntityExtractor(llm=llm, ontology=_ONTOLOGY)
        doc_one, doc_two = Document(id="d1", content="doc one"), Document(id="d2", content="doc two")

        result = extractor.execute(EntityExtractor.input_schema(documents=[doc_one, doc_two]))

        # The same LLM id from two documents must NOT alias: wiring ids are doc-scoped, and identity
        # is assigned later by KnowledgeGraphWriter name resolution (which converges them by name).
        assert {(n["labels"][0], n["properties"]["id"]) for n in result["nodes"]} == {
            ("ORG", "acme@d1"),
            ("ORG", "acme@d2"),
        }
        assert {n["properties"]["name"] for n in result["nodes"]} == {"Acme"}
