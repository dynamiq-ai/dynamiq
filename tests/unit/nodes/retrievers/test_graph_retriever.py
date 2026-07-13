"""Unit tests for KnowledgeGraphRetriever: ACL/filter compilation, query building, and rendering (no DB).

The security-critical piece is the pure compiler function ``_compile_edge_filters`` and the fact that
filter VALUES are always bound parameters. Filters use the same structured grammar as the vector-store
retrievers ({"field","operator","value"} / {"operator","conditions"}). A ``StubGraphStore`` returns
canned records so ``execute`` is exercised end-to-end without Neo4j.
"""

from typing import Any, ClassVar

import pytest
from pydantic import ValidationError

from dynamiq.connections import Neo4j
from dynamiq.nodes.embedders.base import TextEmbedder
from dynamiq.nodes.knowledge_graphs import KnowledgeGraphRetriever, Ontology
from dynamiq.nodes.knowledge_graphs.entity_extractor import ENTITY_EMBEDDING_VECTOR_INDEX
from dynamiq.nodes.knowledge_graphs.retriever import GraphRetrieverInputSchema, _compile_edge_filters
from dynamiq.nodes.node import Node, NodeGroup
from dynamiq.storages.graph.neo4j import Neo4jGraphStore


class StubLLM(Node):
    """LLM stub: returns a canned ``content`` (the query-entity JSON) instead of calling a real model."""

    group: ClassVar = NodeGroup.LLMS
    name: str = "stub-llm"
    response_content: str = '{"names": []}'

    def execute(self, input_data, config=None, **kwargs):
        return {"content": self.response_content}


class FailingStubLLM(StubLLM):
    """LLM stub whose run never succeeds — exercises the graceful fallback to the raw query."""

    def execute(self, input_data, config=None, **kwargs):
        raise RuntimeError("llm boom")


class StubReranker(Node):
    """Reranker stub: reverses the candidate facts and caps to ``top_k`` so tests can see rerank ran.

    Records how many candidates it received, to assert the retriever over-fetches (passes the whole
    rendered set) and that the final output is the reranker's, not the raw retrieval order.
    """

    group: ClassVar = NodeGroup.RANKERS
    name: str = "stub-reranker"
    top_k: int = 2
    received: int = 0

    def execute(self, input_data, config=None, **kwargs):
        documents = input_data["documents"] if isinstance(input_data, dict) else input_data.documents
        self.received = len(documents)
        return {"documents": list(reversed(documents))[: self.top_k]}


class FailingStubReranker(StubReranker):
    """Reranker stub whose run never succeeds — exercises graceful degradation to unranked facts."""

    def execute(self, input_data, config=None, **kwargs):
        raise RuntimeError("rerank boom")


class StubTextEmbedder(TextEmbedder):
    """Embeds a query to a fixed-dim vector from its length (no real model); counts calls."""

    name: str = "stub-text-embedder"
    fail: bool = False
    calls: int = 0

    def __init__(self, **kwargs):
        kwargs.setdefault("client", object())  # satisfy ConnectionNode's connection/client requirement
        super().__init__(**kwargs)

    def execute(self, input_data, config=None, **kwargs):
        if self.fail:
            raise RuntimeError("embed boom")
        query = input_data.query if hasattr(input_data, "query") else input_data["query"]
        self.calls += 1
        return {"embedding": [float(len(query)), 1.0, 2.0], "query": query}


_ONTOLOGY = Ontology(entity_types=["Org", "Person"], relationship_types=["WORKS_AT"])


class StubGraphStore:
    """Minimal graph store: records the last query/params and returns canned rows."""

    def __init__(self, rows: list[dict[str, Any]] | None = None):
        self.rows = rows or []
        self.last_query: str | None = None
        self.last_params: dict[str, Any] | None = None

    def run_cypher(self, query, parameters=None, database=None, **kwargs):
        self.last_query = query
        self.last_params = parameters
        return self.rows, None, []

    @staticmethod
    def format_records(records):
        return list(records)


def make_retriever(rows=None, **kwargs) -> KnowledgeGraphRetriever:
    # llm + ontology are optional on the node, but default to an empty-names stub here so query-seeding
    # tests that don't care about extraction fall back to the raw query. Pass llm=None for the no-llm path.
    kwargs.setdefault("llm", StubLLM())
    kwargs.setdefault("ontology", _ONTOLOGY)
    node = KnowledgeGraphRetriever(
        connection=Neo4j(uri="bolt://localhost:7687", username="neo4j", password="password"),
        is_postponed_component_init=True,
        **kwargs,
    )
    node._graph_store = StubGraphStore(rows=rows)
    return node


LOCKED_ACL = {"field": "allowed_principals", "operator": "contains_any", "value": ["group:a"]}


class TestCompileEdgeFilters:
    def test_comparison_equality(self):
        clause, params = _compile_edge_filters("r", {"field": "source_url", "operator": "==", "value": "http://x"})
        assert clause == "r.source_url = $f0"
        assert params == {"f0": "http://x"}

    def test_shorthand_is_normalized_like_vector_stores(self):
        # The {"field": value} shorthand is normalized (via the shared normalize_filters helper) to an AND
        # group, same as the vector-store retrievers accept it.
        clause, params = _compile_edge_filters("r", {"source_url": "http://x"})
        assert clause == "(r.source_url = $f0)"
        assert params == {"f0": "http://x"}

    def test_operators_nested_with_and(self):
        clause, params = _compile_edge_filters(
            "r",
            {
                "operator": "AND",
                "conditions": [
                    {"field": "year", "operator": ">=", "value": 2020},
                    {"field": "year", "operator": "<", "value": 2025},
                ],
            },
        )
        assert clause == "(r.year >= $f0 AND r.year < $f1)"
        assert params == {"f0": 2020, "f1": 2025}

    def test_in_membership(self):
        clause, params = _compile_edge_filters("r", {"field": "workspace", "operator": "in", "value": ["a", "b"]})
        assert clause == "r.workspace IN $f0"
        assert params == {"f0": ["a", "b"]}

    def test_or_logical_grouping(self):
        clause, params = _compile_edge_filters(
            "r",
            {
                "operator": "OR",
                "conditions": [
                    {"field": "a", "operator": "==", "value": 1},
                    {"field": "b", "operator": "==", "value": 2},
                ],
            },
        )
        assert clause == "(r.a = $f0 OR r.b = $f1)"
        assert params == {"f0": 1, "f1": 2}

    def test_contains_any_is_default_deny_list_intersection(self):
        # ACL is expressed as a filter: keep edges whose list shares an element with the value list.
        clause, params = _compile_edge_filters(
            "r", {"field": "allowed_principals", "operator": "contains_any", "value": ["group:a"]}
        )
        assert clause == "size([x IN coalesce(r.allowed_principals, []) WHERE x IN $f0]) > 0"
        assert params == {"f0": ["group:a"]}

    def test_values_are_parameters_not_interpolated(self):
        # An injection-looking value must end up as a bound param, never in the clause text.
        clause, params = _compile_edge_filters("r", {"field": "name", "operator": "==", "value": "' OR 1=1 //"})
        assert clause == "r.name = $f0"
        assert params == {"f0": "' OR 1=1 //"}

    def test_unsafe_key_rejected(self):
        with pytest.raises(ValueError):
            _compile_edge_filters("r", {"field": "bad key; DROP", "operator": "==", "value": "x"})

    def test_unsupported_operator_rejected(self):
        with pytest.raises(ValueError):
            _compile_edge_filters("r", {"field": "year", "operator": "$regex", "value": ".*"})

    def test_unsupported_logical_operator_rejected(self):
        with pytest.raises(ValueError):
            _compile_edge_filters("r", {"operator": "XOR", "conditions": []})


class TestQueryBuilding:
    def test_query_mention_entry_predicate(self):
        node = make_retriever(filters=LOCKED_ACL)
        query, params = node._build_query(GraphRetrieverInputSchema(query="who works at Acme?"), 10)
        assert "toLower($q) CONTAINS toLower(a.name)" in query
        assert params["q"] == "who works at Acme?"
        assert params["lf0"] == ["group:a"]  # locked ACL filter applied
        assert "coalesce(r.allowed_principals, [])" in query

    def test_explicit_entities_seed_by_name_fuzzily(self):
        # Explicit entities are seed NAMES, matched the same (fuzzy) way as extracted ones -- here via the
        # CONTAINS scan (no index), seeded by the entity name, NOT the raw query "x".
        node = make_retriever()
        query, params = node._build_query(GraphRetrieverInputSchema(query="x", entities=["Acme"]), 5)
        assert "toLower($q) CONTAINS toLower(a.name)" in query
        assert params["q"] == "Acme"
        assert "a.name IN $entities" not in query  # no longer an exact anchor
        assert "allowed_principals" not in query  # no locked filters -> no ACL clause

    def test_explicit_entity_ids_entry_predicate(self):
        # Resolved-id seeding (hybrid retrieval) anchors on the unique id, not the ambiguous name.
        node = make_retriever()
        query, params = node._build_query(
            GraphRetrieverInputSchema(query="x", entity_ids=["uuid-acme", "uuid-jane"]), 5
        )
        assert "a.id IN $entity_ids" in query
        assert params["entity_ids"] == ["uuid-acme", "uuid-jane"]

    def test_entity_ids_take_precedence_over_names_and_query(self):
        node = make_retriever()
        query, _ = node._build_query(
            GraphRetrieverInputSchema(query="who works at Acme?", entities=["Acme"], entity_ids=["uuid-acme"]), 5
        )
        assert "a.id IN $entity_ids" in query
        assert "a.name IN $entities" not in query
        assert "CONTAINS toLower(a.name)" not in query

    def test_locked_and_user_filters_both_applied_distinct_params(self):
        # Locked node filter (ACL) and user input filter are both AND-ed in, with non-colliding params.
        node = make_retriever(filters=LOCKED_ACL)
        query, params = node._build_query(
            GraphRetrieverInputSchema(
                query="x",
                entities=["Acme"],
                filters={"field": "source_url", "operator": "==", "value": "http://x"},
            ),
            5,
        )
        assert "$lf0" in query  # locked, always applied
        assert "r.source_url = $uf0" in query  # user, AND-ed on top
        assert params["lf0"] == ["group:a"]
        assert params["uf0"] == "http://x"

    def test_user_filters_cannot_drop_locked(self):
        # Even if the caller supplies their own filters, the locked ACL clause is still present.
        node = make_retriever(filters=LOCKED_ACL)
        query, _ = node._build_query(
            GraphRetrieverInputSchema(
                query="x",
                entities=["Acme"],
                filters={"field": "allowed_principals", "operator": "contains_any", "value": ["group:b"]},
            ),
            5,
        )
        assert query.count("coalesce(r.allowed_principals, [])") == 2  # locked + the user's own, both apply


class TestEntryModes:
    def test_fulltext_index_seek_when_available(self):
        # With the index present, entry is an index seek (no CONTAINS scan), and expansion is anchored.
        node = make_retriever(filters=LOCKED_ACL)
        node._use_fulltext = True
        query, params = node._build_query(GraphRetrieverInputSchema(query="What does Jane use?"), 10)
        assert "db.index.fulltext.queryNodes('entity_name', $q) YIELD node AS a" in query
        assert "CONTAINS" not in query  # no scan
        assert params["q"] == "What~ OR does~ OR Jane~ OR use~"  # fuzzy Lucene OR-query
        # anchored expansion is undirected; per-edge direction comes from the r.src_name/r.dst_name snapshot
        assert "MATCH (a)-[r]-(b)" in query
        assert "coalesce(r.allowed_principals, [])" in query  # ACL still enforced on the edge
        # names render from the edge's own ACL-bearing snapshot, never the shared merged node
        assert "r.src_name AS a_name" in query and "r.dst_name AS b_name" in query
        assert "startNode(r)" not in query and "endNode(r)" not in query  # shared node ids never surface

    def test_scan_fallback_when_index_absent(self):
        node = make_retriever(filters=LOCKED_ACL)
        node._use_fulltext = False
        query, _ = node._build_query(GraphRetrieverInputSchema(query="What does Jane use?"), 10)
        assert "queryNodes" not in query
        assert "toLower($q) CONTAINS toLower(a.name)" in query  # portable scan (seed match still by node name)
        assert "MATCH (a)-[r]->(b)" in query  # directed
        # but rendered names come from the edge snapshot, not the shared node
        assert "r.src_name AS a_name" in query and "r.dst_name AS b_name" in query
        assert "coalesce(a.name" not in query and "coalesce(b.name" not in query

    def test_explicit_entities_use_fulltext_fuzzily_like_extracted(self):
        # With the index present, explicit entities go through the SAME fuzzy full-text seek as extracted
        # names (AND within a name) -- just skipping the LLM extraction step.
        node = make_retriever(filters=LOCKED_ACL)
        node._use_fulltext = True
        query, params = node._build_query(GraphRetrieverInputSchema(query="x", entities=["Acme Capital"]), 5)
        assert "queryNodes" in query
        assert params["q"] == "(Acme~ AND Capital~)"
        assert "CONTAINS" not in query and "a.name IN $entities" not in query

    def test_anchored_query_uses_match_not_optional(self):
        # The "only entities with >=1 ACL-visible edge" property: a plain MATCH drops seeds whose edges
        # are all filtered out (OPTIONAL MATCH would keep them with nulls).
        node = make_retriever(filters=LOCKED_ACL)
        node._use_fulltext = True
        query, _ = node._build_query(GraphRetrieverInputSchema(query="Jane"), 10)
        assert "OPTIONAL MATCH" not in query


class TestExecuteRendering:
    def test_single_hop_renders_documents(self):
        rows = [
            {"a_name": "Jane Doe", "rel": "WORKS_AT", "rprops": {"source_url": "u"}, "b_name": "Acme"},
            {"a_name": "Jane Doe", "rel": "HAS_ATTRIBUTE", "rprops": {"key": "salary"}, "b_name": "$250,000"},
        ]
        node = make_retriever(
            rows=rows, filters={"field": "allowed_principals", "operator": "contains_any", "value": ["u:jane"]}
        )
        out = node.execute(GraphRetrieverInputSchema(query="Jane Doe"))

        contents = [d.content for d in out["documents"]]
        assert contents == ["Jane Doe -[WORKS_AT]-> Acme", "Jane Doe -[salary]-> $250,000"]
        assert out["documents"][0].metadata == {
            "source": "Jane Doe",
            "target": "Acme",
            "rel": "WORKS_AT",
            "source_url": "u",
        }
        assert out["content"] == "- Jane Doe -[WORKS_AT]-> Acme\n- Jane Doe -[salary]-> $250,000"
        assert out["documents"][0].score > out["documents"][1].score  # rank-derived ordering

    def test_single_hop_does_not_expose_endpoint_node_ids(self):
        # Node ids are a public-namespace hash of the canonical name -> never surfaced to the caller.
        rows = [
            {
                "a_name": "Jane Doe",
                "a_id": "id-jane",
                "rel": "WORKS_AT",
                "rprops": {},
                "b_name": "Acme",
                "b_id": "id-acme",
            }
        ]
        node = make_retriever(rows=rows)
        out = node.execute(GraphRetrieverInputSchema(query="Jane Doe"))
        assert "source_id" not in out["documents"][0].metadata
        assert "target_id" not in out["documents"][0].metadata

    def test_empty_result_message(self):
        node = make_retriever(rows=[])
        out = node.execute(GraphRetrieverInputSchema(query="nothing"))
        assert out["documents"] == []
        assert out["content"] == "No matching facts found."

    def test_same_fact_via_multiple_doc_edges_deduped(self):
        # The same fact now lives as one edge per source document; the visible ones must render once.
        rows = [
            {"a_name": "Jane Doe", "rel": "WORKS_AT", "rprops": {"source_doc_id": "docA"}, "b_name": "Acme"},
            {"a_name": "Jane Doe", "rel": "WORKS_AT", "rprops": {"source_doc_id": "docB"}, "b_name": "Acme"},
        ]
        node = make_retriever(rows=rows)
        out = node.execute(GraphRetrieverInputSchema(query="Jane"))
        assert [d.content for d in out["documents"]] == ["Jane Doe -[WORKS_AT]-> Acme"]

    def test_attribute_value_target_rendered_by_value(self):
        # AttributeValue nodes have no name; the store coalesces to .value, surfacing the scoped attribute.
        # The attribute KEY ("salary") labels the relation -- HAS_ATTRIBUTE is bookkeeping; a bare value
        # ("$250,000") would not say WHICH attribute it is. The raw type stays in metadata.
        rows = [{"a_name": "Jane Doe", "rel": "HAS_ATTRIBUTE", "rprops": {"key": "salary"}, "b_name": "$250,000"}]
        node = make_retriever(rows=rows)
        out = node.execute(GraphRetrieverInputSchema(query="Jane"))
        assert out["documents"][0].content == "Jane Doe -[salary]-> $250,000"
        assert out["documents"][0].metadata["rel"] == "HAS_ATTRIBUTE"
        assert out["documents"][0].metadata["key"] == "salary"

    def test_attribute_edge_without_key_falls_back_to_type(self):
        # Defensive: an attribute edge missing its key still renders (the bookkeeping type), never crashes.
        rows = [{"a_name": "Jane Doe", "rel": "HAS_ATTRIBUTE", "rprops": {}, "b_name": "$250,000"}]
        node = make_retriever(rows=rows)
        out = node.execute(GraphRetrieverInputSchema(query="Jane"))
        assert out["documents"][0].content == "Jane Doe -[HAS_ATTRIBUTE]-> $250,000"

    def test_edge_description_is_appended_to_the_fact(self):
        # A description captured on the edge enriches the bare type in the rendered fact (not just metadata).
        rows = [
            {"a_name": "Jane Doe", "rel": "WORKS_AT", "rprops": {"description": "CFO since 2020"}, "b_name": "Acme"}
        ]
        node = make_retriever(rows=rows)
        out = node.execute(GraphRetrieverInputSchema(query="Jane"))
        assert out["documents"][0].content == "Jane Doe -[WORKS_AT]-> Acme: CFO since 2020"

    def test_query_entity_extraction_seeds_fulltext_anding_tokens_within_a_name(self):
        # The LLM extracts ["Acme Capital"] -> the seek ANDs the name's tokens, so it won't match a node
        # that merely shares "Acme" or "Capital" (precision over the noisy whole-question OR).
        node = make_retriever(llm=StubLLM(response_content='{"names": ["Acme Capital"]}'))
        node._use_fulltext = True
        node.execute(GraphRetrieverInputSchema(query="Who works at Acme Capital and where?"))
        assert node._graph_store.last_params["q"] == "(Acme~ AND Capital~)"

    def test_multiple_entities_or_across_names(self):
        # AND within each name, OR across the two entities the question is about.
        node = make_retriever(llm=StubLLM(response_content='{"names": ["Alice Smith", "Helios"]}'))
        node._use_fulltext = True
        node.execute(GraphRetrieverInputSchema(query="Does Alice Smith use Helios?"))
        assert node._graph_store.last_params["q"] == "(Alice~ AND Smith~) OR (Helios~)"

    def test_query_entity_extraction_seeds_contains_scan(self):
        node = make_retriever(llm=StubLLM(response_content='{"names": ["Acme Capital"]}'))
        node._use_fulltext = False
        node.execute(GraphRetrieverInputSchema(query="Who works at Acme Capital and where?"))
        assert node._graph_store.last_params["q"] == "Acme Capital"

    def test_empty_extraction_falls_back_to_whole_query(self):
        node = make_retriever(llm=StubLLM(response_content='{"names": []}'))
        node._use_fulltext = False
        node.execute(GraphRetrieverInputSchema(query="how are things connected?"))
        assert node._graph_store.last_params["q"] == "how are things connected?"

    def test_llm_failure_falls_back_to_whole_query(self):
        node = make_retriever(llm=FailingStubLLM())
        node._use_fulltext = False
        out = node.execute(GraphRetrieverInputSchema(query="who is Jane?"))  # must not raise
        assert node._graph_store.last_params["q"] == "who is Jane?"
        assert out["documents"] == []

    def test_no_llm_falls_back_to_whole_query(self):
        # llm is optional: with no llm, entity extraction is skipped and seeding falls to the raw query.
        node = make_retriever(llm=None, ontology=None)
        node._use_fulltext = False
        out = node.execute(GraphRetrieverInputSchema(query="who is Jane?"))  # must not raise
        assert node._graph_store.last_params["q"] == "who is Jane?"
        assert out["documents"] == []

    def test_summarize_without_llm_is_rejected_at_construction(self):
        # `summarize` composes an answer with the llm, so it can't be enabled without one.
        with pytest.raises(ValidationError):
            make_retriever(llm=None, ontology=None, summarize=True)

    def test_explicit_entities_skip_extraction(self):
        # Explicit entities are used as seed names directly -> the LLM is never consulted, and they seed
        # the fuzzy search the same way extracted names would.
        names_seen = []

        class RecordingLLM(StubLLM):
            def execute(self, input_data, config=None, **kwargs):
                names_seen.append(input_data)
                return {"content": '{"names": ["SHOULD_NOT_BE_USED"]}'}

        node = make_retriever(llm=RecordingLLM())
        node.execute(GraphRetrieverInputSchema(query="anything", entities=["Acme"]))
        assert names_seen == []  # extraction skipped
        assert node._graph_store.last_params["q"] == "Acme"  # seeded by the entity name, not the query


# Three distinct facts so rendering produces 3 documents (no dedup collisions).
_RERANK_ROWS = [
    {"a_name": "Jane", "rel": "WORKS_AT", "rprops": {}, "b_name": "Acme"},
    {"a_name": "Bob", "rel": "WORKS_AT", "rprops": {}, "b_name": "Globex"},
    {"a_name": "Eve", "rel": "WORKS_AT", "rprops": {}, "b_name": "Initech"},
]


class TestReranking:
    def test_reranker_reorders_caps_and_receives_all_candidates(self):
        # The reranker gets ALL rendered facts (top_k = candidate pool) and its output -- reversed, capped
        # to its own top_k=2 -- becomes the result, so precision comes from rerank, not retrieval order.
        reranker = StubReranker(top_k=2)
        node = make_retriever(rows=_RERANK_ROWS, document_reranker=reranker)

        out = node.execute(GraphRetrieverInputSchema(query="who works where?"))

        assert reranker.received == 3  # over-fetch: the whole rendered set was reranked
        assert [d.content for d in out["documents"]] == [
            "Eve -[WORKS_AT]-> Initech",
            "Bob -[WORKS_AT]-> Globex",
        ]
        # The bullet content reflects the reranked, capped set -- not the original 3.
        assert out["content"] == "- Eve -[WORKS_AT]-> Initech\n- Bob -[WORKS_AT]-> Globex"

    def test_reranker_failure_degrades_to_unranked_facts(self):
        # A reranker failure must not fail the read -- it degrades to the unranked facts (like the LLM path).
        node = make_retriever(rows=_RERANK_ROWS, document_reranker=FailingStubReranker())

        out = node.execute(GraphRetrieverInputSchema(query="who works where?"))  # must not raise

        assert [d.content for d in out["documents"]] == [
            "Jane -[WORKS_AT]-> Acme",
            "Bob -[WORKS_AT]-> Globex",
            "Eve -[WORKS_AT]-> Initech",
        ]

    def test_no_reranker_returns_facts_unchanged(self):
        node = make_retriever(rows=_RERANK_ROWS)  # no document_reranker
        out = node.execute(GraphRetrieverInputSchema(query="who works where?"))
        assert len(out["documents"]) == 3

    def test_to_dict_serializes_the_reranker(self):
        node = make_retriever(rows=[], document_reranker=StubReranker())
        data = node.to_dict()
        assert data["document_reranker"]["name"] == "stub-reranker"


class TestVectorSeeding:
    def test_vector_branch_binds_seeds_from_index(self):
        # seed_vectors present -> a per-vector index lookup binds the seed entity `a`; expansion is anchored.
        node = make_retriever(filters=LOCKED_ACL)
        query, params = node._build_query(
            GraphRetrieverInputSchema(query="who drives an automobile?"),
            10,
            seed_names=["car"],
            seed_vectors=[[0.1, 0.2, 0.3]],
        )
        assert "UNWIND $qvecs AS qv" in query
        assert f"db.index.vector.queryNodes('{ENTITY_EMBEDDING_VECTOR_INDEX}', $vk, qv) YIELD node AS a" in query
        assert params["qvecs"] == [[0.1, 0.2, 0.3]]
        assert params["vk"] == node.vector_top_k
        assert "CONTAINS" not in query  # not the scan fallback
        # anchored (undirected) one-hop, ACL still enforced on the edge
        assert "MATCH (a)-[r]-(b)" in query
        assert "coalesce(r.allowed_principals, [])" in query

    def test_entity_ids_take_precedence_over_vectors(self):
        node = make_retriever()
        query, params = node._build_query(
            GraphRetrieverInputSchema(query="x", entity_ids=["uuid-acme"]),
            5,
            seed_vectors=[[0.1, 0.2, 0.3]],
        )
        assert "a.id IN $entity_ids" in query
        assert "vector.queryNodes" not in query

    def test_falls_back_to_scan_without_seed_vectors(self):
        # No seed_vectors -> unchanged behavior (CONTAINS scan when no full-text index).
        node = make_retriever()
        query, _ = node._build_query(GraphRetrieverInputSchema(query="who works at Acme?"), 10)
        assert "vector.queryNodes" not in query
        assert "toLower($q) CONTAINS toLower(a.name)" in query

    def test_seed_vectors_embeds_each_name_when_vector_active(self):
        embedder = StubTextEmbedder(is_postponed_component_init=True)
        node = make_retriever(text_embedder=embedder)
        node._use_vector = True

        vectors = node._seed_vectors(GraphRetrieverInputSchema(query="q"), ["car", "bike"], None, config=None)

        assert vectors == [[3.0, 1.0, 2.0], [4.0, 1.0, 2.0]]  # len("car")=3, len("bike")=4
        assert embedder.calls == 2  # one embed per seed name

    def test_seed_vectors_reuses_query_vector_when_no_names(self):
        # No seed names (extraction found none, or seed_by_query) -> the precomputed whole-query vector is
        # reused as the single seed, with NO extra embed call.
        embedder = StubTextEmbedder(is_postponed_component_init=True)
        node = make_retriever(text_embedder=embedder)
        node._use_vector = True

        vectors = node._seed_vectors(GraphRetrieverInputSchema(query="q"), [], [7.0, 7.0, 7.0], config=None)

        assert vectors == [[7.0, 7.0, 7.0]]
        assert embedder.calls == 0  # reused, not re-embedded

    def test_seed_vectors_none_when_vector_inactive(self):
        embedder = StubTextEmbedder(is_postponed_component_init=True)
        node = make_retriever(text_embedder=embedder)
        node._use_vector = False  # no vector index -> fall back, embedder untouched

        assert node._seed_vectors(GraphRetrieverInputSchema(query="q"), ["car"], None, config=None) is None
        assert embedder.calls == 0

    def test_seed_vectors_none_for_entity_id_anchored(self):
        node = make_retriever(text_embedder=StubTextEmbedder(is_postponed_component_init=True))
        node._use_vector = True
        q = GraphRetrieverInputSchema(query="q", entity_ids=["uuid-x"])
        assert node._seed_vectors(q, [], [1.0], config=None) is None

    def test_seed_vectors_degrades_on_embedder_failure(self):
        node = make_retriever(text_embedder=StubTextEmbedder(is_postponed_component_init=True, fail=True))
        node._use_vector = True
        assert node._seed_vectors(GraphRetrieverInputSchema(query="q"), ["car"], None, config=None) is None

    def test_probe_vector_false_without_embedder(self):
        node = make_retriever()  # no text_embedder
        assert node._probe_vector() is False

    def test_probe_vector_true_when_index_present(self):
        node = make_retriever(text_embedder=StubTextEmbedder(is_postponed_component_init=True))
        store = Neo4jGraphStore.__new__(Neo4jGraphStore)
        store.database = None
        store.run_cypher = lambda *a, **k: ([{"c": 1}], None, [])
        store.format_records = lambda records: list(records)
        node._graph_store = store
        assert node._probe_vector() is True

    def test_to_dict_serializes_the_text_embedder(self):
        node = make_retriever(rows=[], text_embedder=StubTextEmbedder(is_postponed_component_init=True))
        data = node.to_dict()
        assert data["text_embedder"]["name"] == "stub-text-embedder"


class TestFactRerank:
    def test_query_vector_ranks_facts_server_side(self):
        # A query vector -> the neighbourhood is ordered by edge-embedding cosine, so top_k are MOST relevant.
        node = make_retriever(filters=LOCKED_ACL)
        query, params = node._build_query(
            GraphRetrieverInputSchema(query="what does Acme use?"), 10, seed_names=["Acme"], query_vector=[0.1, 0.2]
        )
        assert "vector.similarity.cosine(r.embedding, $qvec)" in query
        assert "ORDER BY CASE WHEN r.embedding IS NULL THEN -1.0" in query
        assert params["qvec"] == [0.1, 0.2]
        assert query.index("ORDER BY") < query.index("LIMIT $limit")  # rank the neighbourhood, THEN limit

    def test_no_query_vector_no_ranking(self):
        node = make_retriever()
        query, params = node._build_query(GraphRetrieverInputSchema(query="who works at Acme?"), 10)
        assert "vector.similarity.cosine" not in query
        assert "qvec" not in params

    def test_render_strips_edge_embedding_from_output(self):
        rows = [
            {
                "a_name": "Acme",
                "a_id": "1",
                "rel": "USES",
                "b_name": "Helios",
                "b_id": "2",
                "rprops": {"embedding": [0.1, 0.2, 0.3], "description": "for trading"},
            }
        ]
        docs = KnowledgeGraphRetriever._render_single_hop(rows)
        assert "embedding" not in docs[0].metadata  # edge vector never surfaces to the caller
        assert docs[0].content == "Acme -[USES]-> Helios: for trading"

    def test_seed_by_query_skips_extraction(self):
        # seed_by_query -> no LLM extraction; seeding falls to the whole-query vector.
        node = make_retriever(seed_by_query=True, llm=StubLLM(response_content='{"names": ["Acme"]}'))
        assert node._seed_entity_names(GraphRetrieverInputSchema(query="who at Acme?"), config=None) == []


class SequencedGraphStore(StubGraphStore):
    """Returns a different canned row batch per run_cypher call; records every (query, params).

    Like a real database, a row only carries ``a_id``/``b_id`` when the query actually RETURNs them —
    so a hop-1 query that forgets the endpoint ids yields an empty frontier here too, instead of the
    canned ids silently keeping multi-hop alive.
    """

    def __init__(self, batches: list[list[dict[str, Any]]]):
        super().__init__()
        self.batches = list(batches)
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def run_cypher(self, query, parameters=None, database=None, **kwargs):
        self.calls.append((query, parameters))
        rows = self.batches.pop(0) if self.batches else []
        returned = {k for k in ("a_id", "b_id") if f"AS {k}" in query}
        return [{k: v for k, v in row.items() if k in returned or k not in ("a_id", "b_id")} for row in rows], None, []


HOP1 = [
    {
        "a_name": "Sven",
        "a_id": "id-sven",
        "rel": "WORKS_AT",
        "rprops": {},
        "b_name": "Nortech",
        "b_id": "id-nortech",
        "anchor_id": "id-sven",  # the MATCH-bound seed — excluded from the hop-2 frontier
    }
]
HOP2 = [{"a_name": "Nortech", "a_id": "id-nortech", "rel": "USES", "rprops": {}, "b_name": "Aegis", "b_id": "id-aegis"}]


def make_multihop_retriever(batches, **kwargs) -> KnowledgeGraphRetriever:
    node = make_retriever(**kwargs)
    node._graph_store = SequencedGraphStore(batches)
    return node


class TestMultiHopBeam:
    def test_default_stays_single_hop(self):
        node = make_multihop_retriever([HOP1])
        node.execute(GraphRetrieverInputSchema(query="Sven"))
        assert len(node._graph_store.calls) == 1  # no expansion queries by default

    def test_two_hops_reaches_the_chain_fact(self):
        # The chain case: hop 1 finds the bridge (Sven -> Nortech), hop 2 expands FROM the bridge's
        # endpoints and reaches the fact the question is actually about (Nortech -> Aegis).
        node = make_multihop_retriever([HOP1, HOP2], max_hops=2)
        out = node.execute(GraphRetrieverInputSchema(query="what does Sven's employer use?"))
        assert [d.content for d in out["documents"]] == [
            "Sven -[WORKS_AT]-> Nortech",
            "Nortech -[USES]-> Aegis",
        ]
        hop_query, hop_params = node._graph_store.calls[1]
        # frontier = NEW nodes only: the seed (anchor) is excluded, so its leftover 1-hop edges never
        # compete in hop 2's beam against true chain facts; it stays in $visited (no walking back).
        assert set(hop_params["frontier"]) == {"id-nortech"}
        assert set(hop_params["visited"]) == {"id-sven", "id-nortech"}
        assert "NOT (startNode(r).id IN $visited AND endNode(r).id IN $visited)" in hop_query
        assert "WITH DISTINCT r" in hop_query

    def test_hop1_query_returns_endpoint_ids_only_for_multi_hop(self):
        # The frontier is built from hop-1's a_id/b_id, so the hop-1 query MUST return them when
        # expansion is on — and stays id-free (the lean pre-multi-hop shape) when it isn't.
        node = make_multihop_retriever([HOP1, HOP2], max_hops=2)
        node.execute(GraphRetrieverInputSchema(query="Sven"))
        hop1_query = node._graph_store.calls[0][0]
        assert "startNode(r).id AS a_id" in hop1_query
        assert "endNode(r).id AS b_id" in hop1_query
        assert "a.id AS anchor_id" in hop1_query  # seed identity, so the frontier can exclude the anchors

        node = make_multihop_retriever([HOP1])
        node.execute(GraphRetrieverInputSchema(query="Sven"))
        assert "a_id" not in node._graph_store.calls[0][0]

    def test_input_max_hops_overrides_node_default(self):
        node = make_multihop_retriever([HOP1, HOP2])  # node default max_hops=1
        node.execute(GraphRetrieverInputSchema(query="Sven", max_hops=2))
        assert len(node._graph_store.calls) == 2

    def test_hop_query_applies_locked_acl(self):
        # The ACL invariant: every hop is filtered by the LOCKED filters, not just hop 1.
        node = make_multihop_retriever([HOP1, HOP2], max_hops=2, filters=LOCKED_ACL)
        node.execute(GraphRetrieverInputSchema(query="Sven"))
        hop_query, hop_params = node._graph_store.calls[1]
        assert "coalesce(r.allowed_principals, [])" in hop_query
        assert hop_params["lf0"] == ["group:a"]

    def test_beam_budget_split_across_hops(self):
        # top_k=50, max_hops=2 -> 25 per hop; explicit beam_width wins.
        node = make_multihop_retriever([HOP1, HOP2], max_hops=2)
        node.execute(GraphRetrieverInputSchema(query="Sven"))
        assert node._graph_store.calls[0][1]["limit"] == 25
        assert node._graph_store.calls[1][1]["limit"] == 25

        node = make_multihop_retriever([HOP1, HOP2], max_hops=2, beam_width=7)
        node.execute(GraphRetrieverInputSchema(query="Sven"))
        assert node._graph_store.calls[1][1]["limit"] == 7

    def test_stops_when_frontier_exhausted(self):
        # hop 2 returns only already-visited endpoints -> no new frontier -> hop 3 never runs.
        hop2_no_new = [
            {
                "a_name": "Sven",
                "a_id": "id-sven",
                "rel": "KNOWS",
                "rprops": {},
                "b_name": "Nortech",
                "b_id": "id-nortech",
            }
        ]
        node = make_multihop_retriever([HOP1, hop2_no_new, HOP2], max_hops=3)
        node.execute(GraphRetrieverInputSchema(query="Sven"))
        assert len(node._graph_store.calls) == 2

    def test_stops_on_empty_hop(self):
        node = make_multihop_retriever([HOP1, []], max_hops=3)
        node.execute(GraphRetrieverInputSchema(query="Sven"))
        assert len(node._graph_store.calls) == 2  # empty hop 2 -> no hop 3

    def test_total_facts_capped_at_top_k(self):
        # An explicit beam_width can exceed the per-hop split; the FINAL count must still honor top_k,
        # dropping deepest-hop facts first (rows are ordered hop-by-hop).
        hop1 = [
            {"a_name": f"P{i}", "a_id": f"p{i}", "rel": "WORKS_AT", "rprops": {}, "b_name": "Acme", "b_id": "acme"}
            for i in range(3)
        ]
        hop2 = [
            {"a_name": "Acme", "a_id": "acme", "rel": "USES", "rprops": {}, "b_name": f"S{i}", "b_id": f"s{i}"}
            for i in range(3)
        ]
        node = make_multihop_retriever([hop1, hop2], max_hops=2, beam_width=5, top_k=4)
        out = node.execute(GraphRetrieverInputSchema(query="Acme"))
        assert len(out["documents"]) == 4  # 6 rendered facts -> capped to top_k
        assert out["documents"][0].content.endswith("Acme")  # hop-1 facts survive the cut
