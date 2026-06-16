"""Unit tests for HybridGraphVectorRetriever: entity-id seeding, fact source-chunk grounding, render."""

from dynamiq.nodes.extractors.entity_extractor import KG_ENTITY_IDS_KEY
from dynamiq.nodes.retrievers.hybrid_graph_vector import HybridGraphVectorRetriever
from dynamiq.types import Document


def _fact(content, source_doc_id=None, source_doc_ids=None):
    md = {}
    if source_doc_id is not None:
        md["source_doc_id"] = source_doc_id
    if source_doc_ids is not None:
        md["source_doc_ids"] = source_doc_ids
    return Document(content=content, metadata=md)


class TestSeedEntityIds:
    def test_collects_distinct_ids_in_order(self):
        passages = [
            Document(content="a", metadata={KG_ENTITY_IDS_KEY: ["uuid-jane", "uuid-acme"]}),
            Document(content="b", metadata={KG_ENTITY_IDS_KEY: ["uuid-acme", "uuid-berlin"]}),  # acme deduped
        ]
        assert HybridGraphVectorRetriever._seed_entity_ids(passages, 10) == ["uuid-jane", "uuid-acme", "uuid-berlin"]

    def test_caps_number_of_seeds(self):
        passages = [Document(content="a", metadata={KG_ENTITY_IDS_KEY: ["a", "b", "c"]})]
        assert HybridGraphVectorRetriever._seed_entity_ids(passages, 2) == ["a", "b"]


class TestFactSourceIds:
    def test_combines_list_and_scalar(self):
        fact = _fact("A -[R]-> B", source_doc_id="c1", source_doc_ids=["c2", "c3"])
        assert HybridGraphVectorRetriever._fact_source_ids(fact) == ["c2", "c3", "c1"]

    def test_scalar_not_duplicated(self):
        fact = _fact("A -[R]-> B", source_doc_id="c1", source_doc_ids=["c1"])
        assert HybridGraphVectorRetriever._fact_source_ids(fact) == ["c1"]


class TestMissingSourceIds:
    def test_only_ids_not_already_retrieved(self):
        facts = [_fact("f1", source_doc_id="c1"), _fact("f2", source_doc_id="c7"), _fact("f3", source_doc_id="c7")]
        # c1 is a retrieved passage; c7 is missing (and deduped across facts).
        assert HybridGraphVectorRetriever._missing_source_ids(facts, {"c1"}) == ["c7"]


class TestPassageGroundedRanking:
    def test_facts_from_retrieved_passages_come_first(self):
        facts = [
            _fact("hub spur 1", source_doc_id="c99"),          # from a non-retrieved chunk
            _fact("Jane WORKS_AT Acme", source_doc_id="c1"),    # from a retrieved passage
            _fact("hub spur 2", source_doc_id="c98"),
            _fact("Acme USES Helios", source_doc_id="c2"),      # from a retrieved passage
        ]
        ranked = HybridGraphVectorRetriever._passage_grounded_first(facts, {"c1", "c2"})
        assert [f.content for f in ranked] == [
            "Jane WORKS_AT Acme",   # grounded, original order preserved
            "Acme USES Helios",
            "hub spur 1",           # ungrounded, original order preserved
            "hub spur 2",
        ]

    def test_stable_when_nothing_grounded(self):
        facts = [_fact("a", source_doc_id="x"), _fact("b", source_doc_id="y")]
        ranked = HybridGraphVectorRetriever._passage_grounded_first(facts, {"c1"})
        assert [f.content for f in ranked] == ["a", "b"]


class TestIteration:
    def test_fact_neighbor_ids(self):
        fact = Document(content="Jane -[WORKS_AT]-> Acme", metadata={"source_id": "id-jane", "target_id": "id-acme"})
        assert HybridGraphVectorRetriever._fact_neighbor_ids(fact) == ["id-jane", "id-acme"]

    def test_fact_neighbor_ids_missing(self):
        assert HybridGraphVectorRetriever._fact_neighbor_ids(Document(content="x")) == []

    def test_next_frontier_prefers_grounded_dedups_visited_and_caps(self):
        # f1 is grounded (source chunk c1 retrieved), so its neighbors are expanded first.
        f1 = Document(content="f1", metadata={"source_id": "a", "target_id": "b", "source_doc_id": "c1"})
        f2 = Document(content="f2", metadata={"source_id": "a", "target_id": "d", "source_doc_id": "c99"})
        visited = {"a"}  # 'a' is the current seed — must not be re-expanded

        frontier = HybridGraphVectorRetriever._next_frontier([f2, f1], {"c1"}, visited, beam_width=10)

        assert frontier == ["b", "d"]  # b (from grounded f1) before d; 'a' filtered as visited
        assert visited == {"a", "b", "d"}

    def test_next_frontier_beam_cap(self):
        f1 = Document(content="f1", metadata={"source_id": "a", "target_id": "b", "source_doc_id": "c1"})
        f2 = Document(content="f2", metadata={"source_id": "x", "target_id": "y", "source_doc_id": "c99"})
        # beam_width=1 -> only the top (grounded) fact's neighbors are considered.
        frontier = HybridGraphVectorRetriever._next_frontier([f1, f2], {"c1"}, set(), beam_width=1)
        assert frontier == ["a"]


class TestRenderGrounding:
    def test_fact_cited_to_retrieved_passage(self):
        passages = [Document(id="c1", content="Jane works at Acme.")]
        facts = [_fact("Jane Doe -[WORKS_AT]-> Acme", source_doc_id="c1")]

        out = HybridGraphVectorRetriever._render(passages, facts)

        assert out["content"] == (
            "## Passages\n[1] Jane works at Acme.\n\n## Facts\n- Jane Doe -[WORKS_AT]-> Acme   (source: [1])"
        )
        fact_doc = out["documents"][-1]
        assert fact_doc.metadata["origin"] == "fact"
        assert fact_doc.metadata["source_passages"] == [1]

    def test_evidence_chunk_appended_and_cited(self):
        passages = [Document(id="c1", content="Jane works at Acme.")]
        facts = [
            _fact("Jane Doe -[WORKS_AT]-> Acme", source_doc_id="c1"),
            _fact("Acme -[LOCATED_IN]-> Berlin", source_doc_id="c7"),  # source chunk NOT retrieved
        ]
        evidence = [Document(id="c7", content="Acme is based in Berlin.")]

        out = HybridGraphVectorRetriever._render(passages, facts, evidence)

        assert out["content"] == (
            "## Passages\n"
            "[1] Jane works at Acme.\n"
            "[2] (graph evidence) Acme is based in Berlin.\n\n"
            "## Facts\n"
            "- Jane Doe -[WORKS_AT]-> Acme   (source: [1])\n"
            "- Acme -[LOCATED_IN]-> Berlin   (source: [2])"
        )
        origins = [d.metadata["origin"] for d in out["documents"]]
        assert origins == ["passage", "evidence", "fact", "fact"]

    def test_fact_without_known_source_has_no_citation(self):
        out = HybridGraphVectorRetriever._render([], [_fact("A -[R]-> B")])
        assert out["content"] == "## Facts\n- A -[R]-> B"

    def test_empty_returns_placeholder(self):
        assert HybridGraphVectorRetriever._render([], [])["content"] == "No matching context found."

    def test_dedupes_passages_by_content(self):
        passages = [Document(content="same"), Document(content="same"), Document(content="other")]
        out = HybridGraphVectorRetriever._render(passages, [])
        assert out["content"] == "## Passages\n[1] same\n[2] other"
