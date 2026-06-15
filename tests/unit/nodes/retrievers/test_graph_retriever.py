"""Unit tests for GraphRetriever: ACL/filter compilation, query building, and rendering (no DB).

The security-critical pieces are the pure compiler functions (``_acl_clause``, ``_compile_edge_filters``)
and the fact that filter VALUES are always bound parameters. A ``StubGraphStore`` returns canned records
so ``execute`` is exercised end-to-end without Neo4j.
"""

from typing import Any

import pytest

from dynamiq.connections import Neo4j
from dynamiq.nodes.retrievers import GraphRetriever
from dynamiq.nodes.retrievers.graph import GraphRetrieverInputSchema, _compile_edge_filters


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


def make_retriever(rows=None, **kwargs) -> GraphRetriever:
    node = GraphRetriever(
        connection=Neo4j(uri="bolt://localhost:7687", username="neo4j", password="password"),
        is_postponed_component_init=True,
        **kwargs,
    )
    node._graph_store = StubGraphStore(rows=rows)
    return node


LOCKED_ACL = {"allowed_principals": {"$intersects": ["group:a"]}}


class TestCompileEdgeFilters:
    def test_equality_shorthand(self):
        clauses, params = _compile_edge_filters("r", {"source_url": "http://x"})
        assert clauses == ["r.source_url = $f0"]
        assert params == {"f0": "http://x"}

    def test_operators(self):
        clauses, params = _compile_edge_filters("r", {"year": {"$gte": 2020, "$lt": 2025}})
        assert clauses == ["r.year >= $f0_gte", "r.year < $f0_lt"]
        assert params == {"f0_gte": 2020, "f0_lt": 2025}

    def test_in_and_any_membership(self):
        clauses, params = _compile_edge_filters("r", {"workspace": {"$in": ["a", "b"]}, "tags": {"$any": "x"}})
        # keys are sorted -> tags (f0) before workspace (f1)
        assert clauses == ["$f0_any IN r.tags", "r.workspace IN $f1_in"]
        assert params == {"f0_any": "x", "f1_in": ["a", "b"]}

    def test_intersects_is_default_deny_list_intersection(self):
        # ACL is expressed as a filter: keep edges whose list shares an element with the param list.
        clauses, params = _compile_edge_filters("r", {"allowed_principals": {"$intersects": ["group:a"]}})
        assert clauses == ["size([x IN coalesce(r.allowed_principals, []) WHERE x IN $f0_intersects]) > 0"]
        assert params == {"f0_intersects": ["group:a"]}

    def test_values_are_parameters_not_interpolated(self):
        # An injection-looking value must end up as a bound param, never in the clause text.
        clauses, params = _compile_edge_filters("r", {"name": "' OR 1=1 //"})
        assert clauses == ["r.name = $f0"]
        assert params == {"f0": "' OR 1=1 //"}

    def test_unsafe_key_rejected(self):
        with pytest.raises(ValueError):
            _compile_edge_filters("r", {"bad key; DROP": "x"})

    def test_unsupported_operator_rejected(self):
        with pytest.raises(ValueError):
            _compile_edge_filters("r", {"year": {"$regex": ".*"}})


class TestQueryBuilding:
    def test_query_mention_entry_predicate(self):
        node = make_retriever(filters=LOCKED_ACL)
        query, params = node._build_query(GraphRetrieverInputSchema(query="who works at Acme?"), 10)
        assert "toLower($q) CONTAINS toLower(a.name)" in query
        assert params["q"] == "who works at Acme?"
        assert params["lf0_intersects"] == ["group:a"]  # locked ACL filter applied
        assert "coalesce(r.allowed_principals, [])" in query

    def test_explicit_entities_entry_predicate(self):
        node = make_retriever()
        query, params = node._build_query(GraphRetrieverInputSchema(query="x", entities=["Acme"]), 5)
        assert "a.name IN $entities" in query
        assert params["entities"] == ["Acme"]
        assert "allowed_principals" not in query  # no locked filters -> no ACL clause

    def test_locked_and_user_filters_both_applied_distinct_params(self):
        # Locked node filter (ACL) and user input filter are both AND-ed in, with non-colliding params.
        node = make_retriever(filters=LOCKED_ACL)
        query, params = node._build_query(
            GraphRetrieverInputSchema(query="x", entities=["Acme"], filters={"source_url": "http://x"}), 5
        )
        assert "$lf0_intersects" in query  # locked, always applied
        assert "r.source_url = $uf0" in query  # user, AND-ed on top
        assert params["lf0_intersects"] == ["group:a"]
        assert params["uf0"] == "http://x"

    def test_user_filters_cannot_drop_locked(self):
        # Even if the caller supplies their own filters, the locked ACL clause is still present.
        node = make_retriever(filters=LOCKED_ACL)
        query, _ = node._build_query(
            GraphRetrieverInputSchema(query="x", entities=["Acme"], filters={"allowed_principals": {"$intersects": ["group:b"]}}), 5
        )
        assert query.count("coalesce(r.allowed_principals, [])") == 2  # locked + the user's own, both apply

    def test_multi_hop_inlines_validated_depth_and_guards_every_edge(self):
        node = make_retriever(filters=LOCKED_ACL, max_depth=3)
        query, _ = node._build_query(GraphRetrieverInputSchema(query="x", entities=["Acme"]), 5)
        assert "[rels*1..3]" in query
        assert "all(rel IN rels WHERE" in query

    def test_depth_capped(self):
        node = make_retriever(max_depth=99)
        assert node._effective_depth() == 5


class TestEntryModes:
    def test_fulltext_index_seek_when_available(self):
        # With the index present, entry is an index seek (no CONTAINS scan), and expansion is anchored.
        node = make_retriever(filters=LOCKED_ACL)
        node._use_fulltext = True
        query, params = node._build_query(GraphRetrieverInputSchema(query="What does Jane use?"), 10)
        assert "db.index.fulltext.queryNodes('entity_name', $q) YIELD node AS a" in query
        assert "CONTAINS" not in query  # no scan
        assert params["q"] == "What~ OR does~ OR Jane~ OR use~"  # fuzzy Lucene OR-query
        # anchored expansion is undirected; real direction recovered via startNode/endNode
        assert "MATCH (a)-[r]-(b)" in query
        assert "startNode(r)" in query and "endNode(r)" in query
        assert "coalesce(r.allowed_principals, [])" in query  # ACL still enforced on the edge

    def test_scan_fallback_when_index_absent(self):
        node = make_retriever(filters=LOCKED_ACL)
        node._use_fulltext = False
        query, _ = node._build_query(GraphRetrieverInputSchema(query="What does Jane use?"), 10)
        assert "queryNodes" not in query
        assert "toLower($q) CONTAINS toLower(a.name)" in query  # portable scan
        assert "MATCH (a)-[r]->(b)" in query  # directed

    def test_explicit_entities_anchor_on_entity_label(self):
        node = make_retriever(filters=LOCKED_ACL)
        node._use_fulltext = True  # ignored — explicit entities take precedence over the index
        query, params = node._build_query(GraphRetrieverInputSchema(query="x", entities=["Acme"]), 5)
        assert "MATCH (a:Entity)" in query
        assert "a.name IN $entities" in query
        assert "queryNodes" not in query and "CONTAINS" not in query
        assert params["entities"] == ["Acme"]

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
        node = make_retriever(rows=rows, filters={"allowed_principals": {"$intersects": ["u:jane"]}})
        out = node.execute(GraphRetrieverInputSchema(query="Jane Doe"))

        contents = [d.content for d in out["documents"]]
        assert contents == ["Jane Doe -[WORKS_AT]-> Acme", "Jane Doe -[HAS_ATTRIBUTE]-> $250,000"]
        assert out["documents"][0].metadata == {"source": "Jane Doe", "target": "Acme", "rel": "WORKS_AT", "source_url": "u"}
        assert out["content"] == "- Jane Doe -[WORKS_AT]-> Acme\n- Jane Doe -[HAS_ATTRIBUTE]-> $250,000"
        assert out["documents"][0].score > out["documents"][1].score  # rank-derived ordering

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
        rows = [{"a_name": "Jane Doe", "rel": "HAS_ATTRIBUTE", "rprops": {"key": "salary"}, "b_name": "$250,000"}]
        node = make_retriever(rows=rows)
        out = node.execute(GraphRetrieverInputSchema(query="Jane"))
        assert out["documents"][0].content == "Jane Doe -[HAS_ATTRIBUTE]-> $250,000"

    def test_multi_hop_renders_path(self):
        rows = [
            {
                "node_names": ["Jane Doe", "Acme", "TradingX"],
                "rel_types": ["WORKS_AT", "USES"],
                "rel_props": [{"source_doc_ids": ["d1"]}, {"source_doc_ids": ["d2"]}],
            }
        ]
        node = make_retriever(rows=rows, filters={"allowed_principals": {"$intersects": ["u:jane"]}}, max_depth=2)
        out = node.execute(GraphRetrieverInputSchema(query="x", entities=["Jane Doe"]))
        doc = out["documents"][0]
        assert doc.content == "Jane Doe -[WORKS_AT]-> Acme -[USES]-> TradingX"
        assert doc.metadata["source"] == "Jane Doe"
        assert doc.metadata["target"] == "TradingX"
        assert doc.metadata["source_doc_ids"] == ["d1", "d2"]
