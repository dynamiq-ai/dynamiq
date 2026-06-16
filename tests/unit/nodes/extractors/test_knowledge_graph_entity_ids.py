"""Unit tests for KnowledgeGraphWriter's per-chunk entity-id attachment (no LLM, no graph DB)."""

from dynamiq.nodes.extractors.entity_extractor import KG_ENTITY_IDS_KEY
from dynamiq.nodes.extractors.knowledge_graph import KnowledgeGraphWriter, _entity_ids_by_doc
from dynamiq.types import Document


class TestEntityIdsByDoc:
    def test_groups_resolved_endpoint_ids_by_source_doc(self):
        resolved = [
            {"type": "WORKS_AT", "start_identity": "uuid-jane", "end_identity": "uuid-acme",
             "properties": {"source_doc_id": "c1"}},
            {"type": "USES", "start_identity": "uuid-acme", "end_identity": "uuid-helios",
             "properties": {"source_doc_id": "c2"}},
        ]
        assert _entity_ids_by_doc(resolved) == {
            "c1": ["uuid-acme", "uuid-jane"],   # sorted
            "c2": ["uuid-acme", "uuid-helios"],
        }

    def test_excludes_attribute_value_endpoint_but_keeps_owner(self):
        resolved = [
            {"type": "HAS_ATTRIBUTE", "start_identity": "uuid-jane", "end_identity": "uuid-jane::salary",
             "properties": {"source_doc_id": "c1"}},
        ]
        # The owner entity (start) is kept; the AttributeValue holder (end) is not.
        assert _entity_ids_by_doc(resolved) == {"c1": ["uuid-jane"]}

    def test_skips_relationships_without_source_doc(self):
        assert _entity_ids_by_doc([{"type": "R", "start_identity": "a", "end_identity": "b", "properties": {}}]) == {}


class TestAttachEntityIds:
    def test_tags_each_chunk_with_its_ids_without_mutating_input(self):
        docs = [
            Document(id="c1", content="Jane works at Acme.", metadata={"source": "f"}),
            Document(id="c2", content="Acme uses Helios."),
        ]
        resolved = [
            {"type": "WORKS_AT", "start_identity": "uuid-jane", "end_identity": "uuid-acme",
             "properties": {"source_doc_id": "c1"}},
            {"type": "USES", "start_identity": "uuid-acme", "end_identity": "uuid-helios",
             "properties": {"source_doc_id": "c2"}},
        ]

        out = KnowledgeGraphWriter._attach_entity_ids(docs, resolved)

        by_id = {d.id: d for d in out}
        assert by_id["c1"].metadata[KG_ENTITY_IDS_KEY] == ["uuid-acme", "uuid-jane"]
        assert by_id["c1"].metadata["source"] == "f"  # existing metadata preserved
        assert by_id["c2"].metadata[KG_ENTITY_IDS_KEY] == ["uuid-acme", "uuid-helios"]
        # Input documents are not mutated.
        assert KG_ENTITY_IDS_KEY not in (docs[0].metadata or {})

    def test_chunk_with_no_facts_gets_empty_list(self):
        docs = [Document(id="c1", content="nothing linked")]
        out = KnowledgeGraphWriter._attach_entity_ids(docs, [])
        assert out[0].metadata[KG_ENTITY_IDS_KEY] == []
