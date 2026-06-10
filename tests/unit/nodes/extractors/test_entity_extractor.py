"""Unit tests for ``EntityExtractor`` transform/sanitize logic (no LLM, no Neo4j)."""

import json
from typing import ClassVar

import pytest

from dynamiq.nodes.extractors import EntityExtractor
from dynamiq.nodes.node import Node, NodeGroup
from dynamiq.storages.graph.neo4j.neo4j import Neo4jGraphStore
from dynamiq.types import Document


class StubLLM(Node):
    """Minimal Node that returns a canned ``content`` payload instead of calling a real LLM."""

    group: ClassVar = NodeGroup.LLMS
    name: str = "stub-llm"
    response_content: str = "{}"

    def execute(self, input_data, config=None, **kwargs):
        return {"content": self.response_content}


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


class TestDedupeEntities:
    def test_dedupes_by_id_keeping_first(self):
        entities = [
            {"id": "acme", "type": "Org", "name": "Acme Capital"},
            {"id": "acme", "type": "Org", "name": "Acme dup"},
            {"id": "jane", "type": "Person", "name": "Jane"},
            {"type": "Person", "name": "no-id-dropped"},
        ]
        deduped = EntityExtractor._dedupe_entities(entities)
        assert [e["id"] for e in deduped] == ["acme", "jane"]
        assert deduped[0]["name"] == "Acme Capital"


class TestToWriteGraphPayload:
    def test_builds_write_graph_shape_and_drops_dangling(self):
        entities = [
            {"id": "acme", "type": "Hedge Fund", "name": "Acme Capital", "properties": {"aum": "10B"}},
            {"id": "jane", "type": "person", "name": "Jane Doe"},
        ]
        relationships = [
            {"source_id": "jane", "target_id": "acme", "type": "works at", "properties": {"since": 2020}},
            {"source_id": "jane", "target_id": "ghost", "type": "KNOWS"},  # dangling -> dropped
        ]
        nodes, graph_rels = EntityExtractor._to_write_graph_payload(entities, relationships)

        assert nodes == [
            {
                "labels": ["HEDGE_FUND"],
                "identity_key": "id",
                "properties": {"aum": "10B", "id": "acme", "name": "Acme Capital"},
            },
            {"labels": ["PERSON"], "identity_key": "id", "properties": {"id": "jane", "name": "Jane Doe"}},
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
        nodes, _ = EntityExtractor._to_write_graph_payload(entities, [])
        assert [n["properties"]["id"] for n in nodes] == ["y"]


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

    def test_garbage_returns_empty(self):
        assert EntityExtractor._parse_llm_json("not json at all") == {"entities": [], "relationships": []}

    def test_none_returns_empty(self):
        assert EntityExtractor._parse_llm_json(None) == {"entities": [], "relationships": []}


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
        extractor = EntityExtractor(llm=llm)

        result = extractor.execute(
            EntityExtractor.input_schema(documents=[Document(content="Jane Doe works at Acme Capital.")])
        )

        assert {n["labels"][0] for n in result["nodes"]} == {"HEDGE_FUND", "PERSON"}
        assert len(result["relationships"]) == 1
        assert result["relationships"][0]["type"] == "WORKS_AT"
        _assert_valid_for_neo4j(result["nodes"], result["relationships"])

    def test_execute_merges_across_documents_and_dedupes(self):
        payload = {
            "entities": [{"id": "acme", "type": "Org", "name": "Acme"}],
            "relationships": [],
        }
        llm = StubLLM(response_content=json.dumps(payload))
        extractor = EntityExtractor(llm=llm)

        result = extractor.execute(
            EntityExtractor.input_schema(documents=[Document(content="doc one"), Document(content="doc two")])
        )
        # Same entity id surfaced by both docs -> deduped to one node.
        assert len(result["nodes"]) == 1
        assert result["nodes"][0]["properties"]["id"] == "acme"
