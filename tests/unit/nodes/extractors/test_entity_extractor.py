"""Unit tests for ``KnowledgeGraphEntityExtractor`` transform/sanitize logic (no LLM, no Neo4j)."""

import json
from typing import ClassVar

import pytest

from dynamiq.nodes.knowledge_graphs import KnowledgeGraphEntityExtractor, Ontology
from dynamiq.nodes.knowledge_graphs.entity_extractor import ATTRIBUTE_VALUE_LABEL, ENTITY_LABEL, HAS_ATTRIBUTE_TYPE
from dynamiq.nodes.node import Node, NodeGroup
from dynamiq.storages.graph.neo4j.neo4j import Neo4jGraphStore
from dynamiq.types import Document

# ``ontology`` is required on KnowledgeGraphEntityExtractor; this permissive schema covers the types the tests emit so
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


class FailingStubLLM(Node):
    """Stub LLM whose every run fails (non-SUCCESS) — simulates a systemic error like bad credentials."""

    group: ClassVar = NodeGroup.LLMS
    name: str = "failing-stub-llm"

    def execute(self, input_data, config=None, **kwargs):
        raise RuntimeError("llm boom")


class FlakyStubLLM(Node):
    """Stub LLM that fails on the given 0-based call indices and returns ``response_content`` otherwise."""

    group: ClassVar = NodeGroup.LLMS
    name: str = "flaky-stub-llm"
    response_content: str = "{}"
    fail_calls: list[int] = []
    calls: int = 0

    def execute(self, input_data, config=None, **kwargs):
        index = self.calls
        self.calls += 1
        if index in self.fail_calls:
            raise RuntimeError("transient llm boom")
        return {"content": self.response_content}


def _assert_valid_for_neo4j(nodes, relationships):
    """Every label/type/key produced must pass Neo4jGraphStore's validators."""
    for node in nodes:
        Neo4jGraphStore._format_labels(node["labels"])
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
        assert KnowledgeGraphEntityExtractor._sanitize_identifier(raw) == expected

    def test_output_always_valid_for_neo4j(self):
        for raw in ["Hedge Fund", "1abc", "a.b.c", "EVENT/2024", "Ünïcödé"]:
            assert _IDENTIFIER_MATCHES(KnowledgeGraphEntityExtractor._sanitize_identifier(raw))


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
        extractor = KnowledgeGraphEntityExtractor(llm=StubLLM(), ontology=_ONTOLOGY)
        nodes, graph_rels = extractor._to_write_graph_payload(entities, relationships)

        # Every entity carries its type label first plus the shared "Entity" label (the full-text index hook),
        # and ONLY identity (id + name) -- the emitted ``aum`` is dropped, never inlined on the node.
        assert nodes == [
            {"labels": ["HEDGE_FUND", "Entity"], "id": "acme", "name": "Acme Capital", "properties": {}},
            {"labels": ["PERSON", "Entity"], "id": "jane", "name": "Jane Doe", "properties": {}},
        ]
        # Endpoint names are snapshotted onto the edge (src_name/dst_name) so retrieval renders from the
        # ACL-bearing edge, never the shared/merged entity node. The LLM's own props stay in ``properties``.
        assert graph_rels == [
            {
                "type": "WORKS_AT",
                "start_label": "PERSON",
                "end_label": "HEDGE_FUND",
                "start_identity": "jane",
                "end_identity": "acme",
                "src_name": "Jane Doe",
                "dst_name": "Acme Capital",
                "properties": {"since": 2020},
            }
        ]
        _assert_valid_for_neo4j(nodes, graph_rels)

    def test_entity_without_type_or_id_is_skipped(self):
        entities = [{"id": "x"}, {"type": "Person"}, {"id": "y", "type": "Person"}]
        extractor = KnowledgeGraphEntityExtractor(llm=StubLLM(), ontology=_ONTOLOGY)
        nodes, _ = extractor._to_write_graph_payload(entities, [])
        assert [n["id"] for n in nodes] == ["y"]

    def test_relationship_description_rides_on_the_edge(self):
        entities = [{"id": "jane", "type": "Person", "name": "Jane"}, {"id": "acme", "type": "Org", "name": "Acme"}]
        relationships = [
            {"source_id": "jane", "target_id": "acme", "type": "WORKS_AT", "description": "CFO since 2020"}
        ]
        extractor = KnowledgeGraphEntityExtractor(llm=StubLLM(), ontology=_ONTOLOGY)
        _, graph_rels = extractor._to_write_graph_payload(entities, relationships)
        assert graph_rels[0]["description"] == "CFO since 2020"

    def test_entity_description_is_not_written_to_the_node(self):
        # Descriptions are edge-only: an entity-level description must never land on the (shared) node.
        entities = [{"id": "jane", "type": "Person", "name": "Jane", "description": "the CFO"}]
        extractor = KnowledgeGraphEntityExtractor(llm=StubLLM(), ontology=_ONTOLOGY)
        nodes, _ = extractor._to_write_graph_payload(entities, [])
        assert nodes[0] == {"labels": ["PERSON", "Entity"], "id": "jane", "name": "Jane", "properties": {}}


class TestTypeGuidance:
    def test_ontology_type_descriptions_annotate_the_guidance(self):
        ontology = Ontology(
            entity_types=["Person", "Org"],
            relationship_types=["WORKS_AT"],
            entity_descriptions={"Person": "an individual human"},  # Org intentionally undescribed
            relationship_descriptions={"WORKS_AT": "employment of a person by an organization"},
        )
        guidance = KnowledgeGraphEntityExtractor(llm=StubLLM(), ontology=ontology)._build_type_guidance()

        assert "Person (an individual human)" in guidance
        assert "Org" in guidance and "Org (" not in guidance  # bare when no description
        assert "WORKS_AT (employment of a person by an organization)" in guidance


class TestParseLLMJson:
    def test_plain_json(self):
        assert KnowledgeGraphEntityExtractor._parse_llm_json('{"entities": [], "relationships": []}') == {
            "entities": [],
            "relationships": [],
        }

    def test_markdown_fenced(self):
        raw = '```json\n{"entities": [{"id": "a"}], "relationships": []}\n```'
        assert KnowledgeGraphEntityExtractor._parse_llm_json(raw)["entities"] == [{"id": "a"}]

    def test_prose_around_json(self):
        raw = 'Here you go:\n{"entities": [], "relationships": []}\nHope that helps!'
        assert KnowledgeGraphEntityExtractor._parse_llm_json(raw) == {"entities": [], "relationships": []}

    def test_garbage_returns_none(self):
        assert KnowledgeGraphEntityExtractor._parse_llm_json("not json at all") is None

    def test_none_returns_none(self):
        assert KnowledgeGraphEntityExtractor._parse_llm_json(None) is None

    def test_non_dict_json_returns_none(self):
        assert KnowledgeGraphEntityExtractor._parse_llm_json('[{"id": "a"}]') is None


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
        extractor = KnowledgeGraphEntityExtractor(llm=llm, ontology=_ONTOLOGY)

        result = extractor.execute(
            KnowledgeGraphEntityExtractor.input_schema(documents=[Document(content="Jane Doe works at Acme Capital.")])
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
        extractor = KnowledgeGraphEntityExtractor(llm=llm, ontology=_ONTOLOGY)

        result = extractor.execute(KnowledgeGraphEntityExtractor.input_schema(documents=[Document(content="Acme.")]))

        assert llm.calls == 2  # initial call + one repair round-trip
        assert [n["name"] for n in result["nodes"]] == ["Acme"]

    def test_execute_skips_document_when_recovery_fails(self):
        llm = SequenceStubLLM(responses=["garbage", "still garbage"])
        extractor = KnowledgeGraphEntityExtractor(llm=llm, ontology=_ONTOLOGY)

        result = extractor.execute(KnowledgeGraphEntityExtractor.input_schema(documents=[Document(content="Acme.")]))

        assert llm.calls == 2
        assert result["nodes"] == []
        assert result["relationships"] == []

    def test_execute_promotes_declared_attributes_to_doc_scoped_value_nodes(self):
        from dynamiq.nodes.knowledge_graphs import Ontology
        from dynamiq.nodes.knowledge_graphs.entity_extractor import ATTRIBUTE_VALUE_LABEL, HAS_ATTRIBUTE_TYPE

        payload = {
            "entities": [{"id": "jane", "type": "Person", "name": "Jane Doe", "properties": {"salary": "$250,000"}}],
            "relationships": [],
        }
        ontology = Ontology(entity_types=["Person"], relationship_types=[], attributes={"Person": ["salary"]})
        extractor = KnowledgeGraphEntityExtractor(llm=StubLLM(response_content=json.dumps(payload)), ontology=ontology)
        document = Document(id="doc-1", content="Jane Doe's salary is $250,000.")

        result = extractor.execute(KnowledgeGraphEntityExtractor.input_schema(documents=[document]))

        person = next(n for n in result["nodes"] if n["labels"][0] == "PERSON")
        assert person["id"] == "jane@doc-1"  # wiring id is doc-scoped
        assert "salary" not in person["properties"]  # promoted to an edge, not a node property

        value_node = next(n for n in result["nodes"] if n["labels"][0] == ATTRIBUTE_VALUE_LABEL)
        assert value_node["id"] == "jane@doc-1::salary::doc-1"
        assert value_node["properties"] == {"value": "$250,000"}

        attr_edge = next(r for r in result["relationships"] if r["type"] == HAS_ATTRIBUTE_TYPE)
        assert attr_edge["start_identity"] == "jane@doc-1"
        assert attr_edge["end_identity"] == "jane@doc-1::salary::doc-1"
        assert attr_edge["properties"]["key"] == "salary"
        # Endpoint-name snapshot rides on the ACL-bearing edge: owner name + the value itself, so retrieval
        # renders "Jane Doe -[salary]-> $250,000" without dereferencing the shared node or the value node.
        assert attr_edge["src_name"] == "Jane Doe"
        assert attr_edge["dst_name"] == "$250,000"

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
        extractor = KnowledgeGraphEntityExtractor(llm=StubLLM(response_content=json.dumps(payload)), ontology=_ONTOLOGY)
        result = extractor.execute(
            KnowledgeGraphEntityExtractor.input_schema(documents=[Document(id="doc-1", content="Jane works at Acme.")])
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
        extractor = KnowledgeGraphEntityExtractor(llm=llm, ontology=_ONTOLOGY)
        doc_one, doc_two = Document(id="d1", content="doc one"), Document(id="d2", content="doc two")

        result = extractor.execute(KnowledgeGraphEntityExtractor.input_schema(documents=[doc_one, doc_two]))

        # The same LLM id from two documents must NOT alias: wiring ids are doc-scoped, and identity
        # is assigned later by KnowledgeGraphWriter name resolution (which converges them by name).
        assert {(n["labels"][0], n["id"]) for n in result["nodes"]} == {
            ("ORG", "acme@d1"),
            ("ORG", "acme@d2"),
        }
        assert {n["name"] for n in result["nodes"]} == {"Acme"}

    def test_single_llm_failure_skips_only_that_document(self):
        # A transient LLM failure on one document must NOT abort the batch; remaining documents still process.
        payload = {"entities": [{"id": "acme", "type": "Org", "name": "Acme"}], "relationships": []}
        llm = FlakyStubLLM(fail_calls=[0], response_content=json.dumps(payload))  # doc 1 fails, doc 2 succeeds
        extractor = KnowledgeGraphEntityExtractor(llm=llm, ontology=_ONTOLOGY)
        docs = [Document(id="d1", content="doc one"), Document(id="d2", content="doc two")]

        result = extractor.execute(KnowledgeGraphEntityExtractor.input_schema(documents=docs))

        # Only doc 2's entity survives; doc 1 was skipped, not fatal.
        assert [n["name"] for n in result["nodes"]] == ["Acme"]

    def test_all_documents_failing_raises_systemic_error(self):
        # When EVERY document fails, that's systemic (e.g. bad credentials) — raise instead of an empty graph.
        extractor = KnowledgeGraphEntityExtractor(llm=FailingStubLLM(), ontology=_ONTOLOGY)
        docs = [Document(id="d1", content="a"), Document(id="d2", content="b")]

        with pytest.raises(ValueError, match="all 2 document"):
            extractor.execute(KnowledgeGraphEntityExtractor.input_schema(documents=docs))

    def test_documents_without_id_get_one_so_acl_edges_stay_separate(self):
        # An explicit id=None used to drop the source_doc_id discriminator: identical facts from two docs
        # would MERGE into one edge and the last allowed_principals would overwrite the other (ACL leak).
        # Now every id-less document is assigned a real id, so each fact keeps its own edge + ACL.
        payload = {
            "entities": [
                {"id": "jane", "type": "Person", "name": "Jane Doe"},
                {"id": "acme", "type": "Org", "name": "Acme"},
            ],
            "relationships": [{"source_id": "jane", "target_id": "acme", "type": "works at"}],
        }
        llm = StubLLM(response_content=json.dumps(payload))
        extractor = KnowledgeGraphEntityExtractor(llm=llm, ontology=_ONTOLOGY)
        doc_a = Document(id=None, content="Jane works at Acme.", metadata={"allowed_principals": ["group:a"]})
        doc_b = Document(id=None, content="Jane works at Acme.", metadata={"allowed_principals": ["group:b"]})

        result = extractor.execute(KnowledgeGraphEntityExtractor.input_schema(documents=[doc_a, doc_b]))

        # Ids are assigned on copies, so the caller's input objects are left untouched.
        assert doc_a.id is None and doc_b.id is None
        # Each id-less document was assigned a distinct, non-null id on the returned copies.
        out_a, out_b = result["documents"]
        assert out_a.id is not None and out_b.id is not None and out_a.id != out_b.id

        rels = [r for r in result["relationships"] if r["type"] == "WORKS_AT"]
        assert len(rels) == 2  # two separate edges, not one merged edge
        assert all(r["identity_keys"] == ["source_doc_id"] for r in rels)
        # Each edge carries its OWN document discriminator and its OWN ACL -- neither overwrites the other.
        assert {r["properties"]["source_doc_id"] for r in rels} == {out_a.id, out_b.id}
        assert {tuple(r["properties"]["allowed_principals"]) for r in rels} == {("group:a",), ("group:b",)}


class TestEnforceOntology:
    """Ontology enforcement must never strand an AttributeValue node without its ACL-bearing edge.

    A value (e.g. a salary) lives in a separate node, but its access scope rides on the HAS_ATTRIBUTE
    edge -- so if the owning entity is dropped (and the edge with it), the value node must be dropped too,
    never left as an orphan: a sensitive value with no edge means no ACL can gate it.
    """

    def _extractor(self):
        ontology = Ontology(entity_types=["Person"], relationship_types=[], attributes={"Person": ["salary"]})
        return KnowledgeGraphEntityExtractor(llm=StubLLM(), ontology=ontology)

    def _attribute_graph(self, owner_label):
        # (owner)-[:HAS_ATTRIBUTE {key}]->(:AttributeValue {value}); owner_label decides if the owner survives.
        nodes = [
            {"labels": [owner_label, ENTITY_LABEL], "id": "p1", "name": "Jane", "properties": {}},
            {"labels": [ATTRIBUTE_VALUE_LABEL], "id": "p1::salary", "properties": {"value": "$250,000"}},
        ]
        rels = [
            {
                "type": HAS_ATTRIBUTE_TYPE,
                "start_label": owner_label,
                "end_label": ATTRIBUTE_VALUE_LABEL,
                "start_identity": "p1",
                "end_identity": "p1::salary",
                "src_name": "Jane",
                "dst_name": "$250,000",
                "properties": {"key": "salary"},
            }
        ]
        return nodes, rels

    def test_orphaned_attribute_value_dropped_when_owner_removed(self):
        # Off-ontology owner label -> owner dropped, its HAS_ATTRIBUTE edge dropped, value must NOT survive.
        nodes, rels = self._attribute_graph("EMPLOYEE")
        kept_nodes, kept_rels = self._extractor()._enforce_ontology(nodes, rels)
        assert kept_nodes == []  # no orphaned AttributeValue left behind
        assert kept_rels == []

    def test_attribute_value_kept_when_owner_survives(self):
        # In-ontology owner label -> owner + edge + value all survive together.
        nodes, rels = self._attribute_graph("PERSON")
        kept_nodes, kept_rels = self._extractor()._enforce_ontology(nodes, rels)
        assert sorted(n["labels"][0] for n in kept_nodes) == [ATTRIBUTE_VALUE_LABEL, "PERSON"]
        assert [r["type"] for r in kept_rels] == [HAS_ATTRIBUTE_TYPE]
