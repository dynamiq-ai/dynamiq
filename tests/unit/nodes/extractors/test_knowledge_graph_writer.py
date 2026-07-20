"""Unit tests for the ``KnowledgeGraphWriter`` node (no LLM, no graph DB): write-time entity resolution
and the ``execute`` write path.

Identity contract: LLM-produced ids are intra-extraction wiring only — node identity is decided
exclusively by trigram name similarity against existing same-label nodes. Match -> adopt the
existing node's id; no match -> fresh UUID.
"""

import uuid

import pytest

from dynamiq.connections import Neo4j
from dynamiq.nodes.embedders.base import DocumentEmbedder
from dynamiq.nodes.knowledge_graphs import KnowledgeGraphWriter
from dynamiq.nodes.knowledge_graphs.entity_extractor import (
    ATTRIBUTE_VALUE_LABEL,
    ENTITY_EMBEDDING_VECTOR_INDEX,
    HAS_ATTRIBUTE_TYPE,
)
from dynamiq.storages.graph.neo4j import Neo4jGraphStore


class FakeGraphStore:
    """Minimal write-graph store: records the resolved payload, no real DB."""

    def __init__(self):
        self.written = None

    def supports_write_graph(self) -> bool:
        return True

    def write_graph(self, nodes, relationships, database=None):
        self.written = (nodes, relationships)
        return {
            "nodes_created": len(nodes),
            "relationships_created": len(relationships),
            "properties_set": 0,
            "records": [],
            "keys": [],
        }

    def run_cypher(self, *args, **kwargs):
        return ([], None, [])

    def format_records(self, records):
        return []


class _ResolveStore:
    """Neo4j-typed stub for resolution: answers the two read queries resolution now issues.

    - The existence check (``_existing_ids``, contains no ``queryNodes``) → returns the requested ids that
      are in ``saved`` (the deterministic ids already "in the graph").
    - The fuzzy blocking query (``_blocking_candidates``, contains ``queryNodes``) → returns the configured
      near-dup ``(id, name)`` candidates for the queried label.
    """

    def __init__(self, existing_by_label=None, saved_ids=None):
        self.existing = existing_by_label or {}
        self.saved = set(saved_ids or [])
        self.database = None
        self.queries: list[str] = []

    def run_cypher(self, query, parameters=None, database=None, **kwargs):
        self.queries.append(query)
        params = parameters or {}
        if "queryNodes" in query:  # fuzzy blocking candidates for a label
            rows = [{"id": i, "name": n} for i, n in self.existing.get(params.get("label"), [])]
            return rows, None, []
        # existence check: which of the requested ids already exist
        return [{"id": i} for i in params.get("ids", []) if i in self.saved], None, []

    @staticmethod
    def format_records(records):
        return list(records)


def _neo4j_typed_store(existing_by_label=None, saved_ids=None) -> Neo4jGraphStore:
    """A ``_ResolveStore`` masquerading as a ``Neo4jGraphStore`` so resolution's ``isinstance`` gate passes."""
    store = Neo4jGraphStore.__new__(Neo4jGraphStore)
    stub = _ResolveStore(existing_by_label, saved_ids)
    store.database = None
    store.run_cypher = stub.run_cypher
    store.format_records = stub.format_records
    store._stub = stub  # expose for query assertions
    return store


def make_writer(existing: dict[str, list[tuple[str, str]]] | None = None, fuzzy: bool = True, saved_ids=None):
    """Writer over a Neo4j-typed stub store. ``existing`` are near-dup candidates the blocking query returns;
    ``saved_ids`` are deterministic ids treated as already in the graph (to exercise the idempotent gate)."""
    writer = KnowledgeGraphWriter(
        connection=Neo4j(uri="bolt://localhost:7687", username="neo4j", password="password"),
        is_postponed_component_init=True,
        fuzzy_matching=fuzzy,
    )
    writer._graph_store = _neo4j_typed_store(existing, saved_ids)
    return writer


def entity(label: str, entity_id: str, name: str) -> dict:
    return {"labels": [label], "id": entity_id, "name": name, "properties": {}}


def edge(rel_type: str, start_label: str, end_label: str, start_id: str, end_id: str, props: dict = None) -> dict:
    # src_name/dst_name/description are promoted to top-level edge fields (as the extractor emits them);
    # anything else stays in the ``properties`` bag.
    props = dict(props or {})
    promoted = {k: props.pop(k) for k in ("src_name", "dst_name", "description") if k in props}
    return {
        "type": rel_type,
        "start_label": start_label,
        "end_label": end_label,
        "start_identity": start_id,
        "end_identity": end_id,
        **promoted,
        "properties": props,
    }


def attr_value_node(owner_id: str, attr_key: str, doc_id: str | None, value: str) -> dict:
    """An AttributeValue node as the extractor emits it: composite id + transient `attr_ref` carrying
    the structured (owner, key, doc) parts the writer rebuilds the resolved id from."""
    value_id = f"{owner_id}::{attr_key}" + (f"::{doc_id}" if doc_id is not None else "")
    return {
        "labels": [ATTRIBUTE_VALUE_LABEL],
        "id": value_id,
        "properties": {"value": value},
        "attr_ref": {"owner": owner_id, "key": attr_key, "doc": doc_id},
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

        new_id = resolved_nodes[0]["id"]
        assert new_id != "acme" and is_uuid(new_id)
        # the edge endpoint followed the entity onto its new id
        assert resolved_rels[0]["start_identity"] == new_id

    def test_matches_existing_by_name_despite_different_id(self):
        writer = make_writer({"ORGANIZATION": [("uuid-existing", "Acme Capital")]})
        nodes = [entity("ORGANIZATION", "acme-capital-llc", "Acme Capital LLC")]
        rels = [edge("WORKS_AT", "PERSON", "ORGANIZATION", "jane", "acme-capital-llc")]

        resolved_nodes, resolved_rels = writer._resolve_against_graph(nodes, rels)

        assert resolved_nodes[0]["id"] == "uuid-existing"
        assert resolved_rels[0]["end_identity"] == "uuid-existing"

    def test_llm_id_never_participates_in_identity(self):
        # An existing node whose ID equals the LLM id but whose NAME is dissimilar must NOT capture
        # the new entity: identity is name-only.
        writer = make_writer({"ORGANIZATION": [("acme", "Globex Corporation")]})
        nodes = [entity("ORGANIZATION", "acme", "Acme Capital")]

        resolved_nodes, _ = writer._resolve_against_graph(nodes, [])

        new_id = resolved_nodes[0]["id"]
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
        node_id = resolved_nodes[0]["id"]
        assert {r["start_identity"] for r in resolved_rels} == {node_id}

    def test_dissimilar_names_stay_separate(self):
        writer = make_writer({"PERSON": []})
        nodes = [entity("PERSON", "jane", "Jane Doe"), entity("PERSON", "john", "John Smith")]

        resolved_nodes, _ = writer._resolve_against_graph(nodes, [])

        ids = {n["id"] for n in resolved_nodes}
        assert len(ids) == 2

    def test_attribute_value_ids_rederived_from_owner(self):
        # Value ids are "{wiring_id}::{key}::{doc_id}"; only the owner (wiring) segment is remapped,
        # so values from different documents keep distinct ids after resolution.
        writer = make_writer({"PERSON": [("uuid-jane", "Jane Doe")]})
        nodes = [
            entity("PERSON", "jane@d1", "Jane Doe"),
            attr_value_node("jane@d1", "salary", "d1", "$250,000"),
        ]
        rels = [
            edge(
                HAS_ATTRIBUTE_TYPE, "PERSON", ATTRIBUTE_VALUE_LABEL, "jane@d1", "jane@d1::salary::d1",
                {"key": "salary"},
            )
        ]

        resolved_nodes, resolved_rels = writer._resolve_against_graph(nodes, rels)

        value_node = next(n for n in resolved_nodes if n["labels"][0] == ATTRIBUTE_VALUE_LABEL)
        assert value_node["id"] == "uuid-jane::salary::d1"
        assert resolved_rels[0]["start_identity"] == "uuid-jane"
        assert resolved_rels[0]["end_identity"] == "uuid-jane::salary::d1"
        # The transient ref never leaks to the store.
        assert "attr_ref" not in value_node

    def test_resolution_does_not_mutate_input_payload(self):
        # Resolution must not touch the caller's dicts: the same payload may be written again (e.g. to a
        # second backend). A second resolve must rebuild attribute ids identically -- which is impossible
        # if the first run popped attr_ref or rewrote properties["id"] in place.
        writer = make_writer({"PERSON": [("uuid-jane", "Jane Doe")]})
        nodes = [
            entity("PERSON", "jane@d1", "Jane Doe"),
            attr_value_node("jane@d1", "salary", "d1", "$250,000"),
        ]
        rels = [
            edge(
                HAS_ATTRIBUTE_TYPE, "PERSON", ATTRIBUTE_VALUE_LABEL, "jane@d1", "jane@d1::salary::d1",
                {"key": "salary"},
            )
        ]

        first_nodes, first_rels = writer._resolve_against_graph(nodes, rels)

        # Input is untouched: wiring ids and the transient attr_ref survive on the caller's objects.
        assert nodes[0]["id"] == "jane@d1"
        assert nodes[1]["id"] == "jane@d1::salary::d1"
        assert nodes[1]["attr_ref"] == {"owner": "jane@d1", "key": "salary", "doc": "d1"}
        assert rels[0]["end_identity"] == "jane@d1::salary::d1"

        # Re-resolving the same payload yields the same resolved ids (repeatable, not corrupted).
        second_nodes, second_rels = writer._resolve_against_graph(nodes, rels)
        for resolved in (first_nodes, second_nodes):
            value_node = next(n for n in resolved if n["labels"][0] == ATTRIBUTE_VALUE_LABEL)
            assert value_node["id"] == "uuid-jane::salary::d1"
        assert first_rels[0]["end_identity"] == second_rels[0]["end_identity"] == "uuid-jane::salary::d1"

    def test_attribute_value_id_rederived_when_llm_id_contains_delimiter(self):
        # An LLM id can itself contain "::"; the value id is rebuilt from the carried (owner, key, doc)
        # parts, not by splitting the string, so the owner still resolves onto its UUID.
        writer = make_writer({"PERSON": [("uuid-jane", "Jane Doe")]})
        owner = "Person::42@d1"  # llm_id "Person::42" -> wiring id with embedded "::"
        nodes = [
            entity("PERSON", owner, "Jane Doe"),
            attr_value_node(owner, "salary", "d1", "$250,000"),
        ]
        rels = [
            edge(
                HAS_ATTRIBUTE_TYPE, "PERSON", ATTRIBUTE_VALUE_LABEL, owner, f"{owner}::salary::d1",
                {"key": "salary"},
            )
        ]

        resolved_nodes, resolved_rels = writer._resolve_against_graph(nodes, rels)

        value_node = next(n for n in resolved_nodes if n["labels"][0] == ATTRIBUTE_VALUE_LABEL)
        assert value_node["id"] == "uuid-jane::salary::d1"
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

        assert len({n["id"] for n in resolved_nodes}) == 2


class TestExecute:
    def test_execute_resolves_and_writes(self):
        # Full write path on a fake store: nodes get resolved ids and the store receives the resolved payload.
        writer = make_writer({})
        store = FakeGraphStore()
        writer._graph_store = store

        nodes = [entity("PERSON", "jane@d1", "Jane"), entity("ORGANIZATION", "acme@d1", "Acme")]
        rels = [edge("WORKS_AT", "PERSON", "ORGANIZATION", "jane@d1", "acme@d1", {"source_doc_id": "d1"})]

        result = writer.execute(KnowledgeGraphWriter.input_schema(nodes=nodes, relationships=rels))

        # The store was handed the RESOLVED payload (wiring ids replaced by UUIDs).
        assert result["nodes_created"] == 2 and result["relationships_created"] == 1
        written_ids = {n["id"] for n in store.written[0]}
        assert "jane@d1" not in written_ids and all(is_uuid(i) for i in written_ids)

    def test_execute_empty_payload_writes_nothing(self):
        writer = make_writer({})
        writer._graph_store = FakeGraphStore()

        result = writer.execute(KnowledgeGraphWriter.input_schema(nodes=[], relationships=[]))

        assert result["nodes_created"] == 0 and result["relationships_created"] == 0
        assert writer._graph_store.written is None  # write_graph never called

    def test_execute_skips_bare_nodes_with_no_relationship(self):
        # A bare mention (an entity with no relationship and no attribute edge) is never persisted:
        # provenance lives on edges, so a bare node could not be attributed to a document, reached by
        # retrieval, or removed by delete_documents. Its content-addressed id makes recreation free.
        writer = make_writer({})
        store = FakeGraphStore()
        writer._graph_store = store

        nodes = [
            entity("PERSON", "jane@d1", "Jane"),
            entity("ORGANIZATION", "acme@d1", "Acme"),
            entity("PERSON", "steve@d1", "Steve"),  # bare: no relationship references it
        ]
        rels = [edge("WORKS_AT", "PERSON", "ORGANIZATION", "jane@d1", "acme@d1", {"source_doc_id": "d1"})]

        result = writer.execute(KnowledgeGraphWriter.input_schema(nodes=nodes, relationships=rels))

        assert result["nodes_created"] == 2
        assert {n["name"] for n in store.written[0]} == {"Jane", "Acme"}  # Steve skipped

    def test_execute_all_nodes_bare_writes_nothing(self):
        writer = make_writer({})
        store = FakeGraphStore()
        writer._graph_store = store

        result = writer.execute(
            KnowledgeGraphWriter.input_schema(
                nodes=[entity("PERSON", "steve@d1", "Steve")], relationships=[]
            )
        )

        assert result["nodes_created"] == 0
        assert store.written is None  # write_graph never called

    def test_execute_rejects_relationship_endpoints_absent_from_nodes(self):
        # A relationship endpoint not present in `nodes` can never be resolved: it would write with an
        # ephemeral wiring id that MATCHes no graph node, leaving a dangling edge. The writer rejects such
        # payloads -- both the no-nodes case and a partially-dangling endpoint -- rather than silently
        # writing nothing useful.
        writer = make_writer({})
        store = FakeGraphStore()
        writer._graph_store = store
        rels = [edge("WORKS_AT", "PERSON", "ORGANIZATION", "jane@d1", "acme@d1", {"source_doc_id": "d1"})]

        # Only "jane@d1" is provided as a node -> "acme@d1" is dangling.
        nodes = [entity("PERSON", "jane@d1", "Jane")]
        with pytest.raises(ValueError, match="endpoints absent from"):
            writer.execute(KnowledgeGraphWriter.input_schema(nodes=nodes, relationships=rels))

        # And the no-nodes case (the reported scenario) is rejected too.
        with pytest.raises(ValueError, match="endpoints absent from"):
            writer.execute(KnowledgeGraphWriter.input_schema(nodes=[], relationships=rels))

        assert store.written is None  # nothing written in either case


class StubDocumentEmbedder(DocumentEmbedder):
    """Embeds each document to a fixed-dim vector derived from its content length (no real model)."""

    name: str = "stub-doc-embedder"
    dim: int = 3
    fail: bool = False

    def __init__(self, **kwargs):
        kwargs.setdefault("client", object())  # satisfy ConnectionNode's connection/client requirement
        super().__init__(**kwargs)

    def execute(self, input_data, config=None, **kwargs):
        if self.fail:
            raise RuntimeError("embed boom")
        for doc in input_data.documents:
            doc.embedding = [float(len(doc.content))] * self.dim
        return {"documents": input_data.documents, "meta": {}}


def _neo4j_conn() -> Neo4j:
    return Neo4j(uri="bolt://localhost:7687", username="neo4j", password="password")


def _recording_neo4j_store() -> tuple[Neo4jGraphStore, list[str]]:
    """A Neo4jGraphStore (bypassing __init__/DB) that records every Cypher statement it is asked to run."""
    store = Neo4jGraphStore.__new__(Neo4jGraphStore)
    store.database = None
    recorded: list[str] = []

    def run_cypher(query, parameters=None, database=None, **kwargs):
        recorded.append(query)
        return [], None, []

    store.run_cypher = run_cypher
    return store, recorded


def _entity(label: str, node_id: str, name: str) -> dict:
    return {"labels": [label, "Entity"], "id": node_id, "name": name, "properties": {}}


def _embedder_writer(**embedder_kwargs) -> KnowledgeGraphWriter:
    return KnowledgeGraphWriter(
        connection=_neo4j_conn(),
        is_postponed_component_init=True,
        entity_embedder=StubDocumentEmbedder(is_postponed_component_init=True, **embedder_kwargs),
    )


class TestEmbedEntityNames:
    def test_embeds_names_and_attaches_to_entities_not_attribute_values(self):
        writer = _embedder_writer(dim=3)
        writer._graph_store, _ = _recording_neo4j_store()
        nodes = [
            _entity("PERSON", "u1", "Jane Doe"),  # len("Jane Doe") == 8
            {"labels": [ATTRIBUTE_VALUE_LABEL], "id": "v1", "properties": {"value": "CTO"}},
        ]

        vectors = writer._embed_entity_names(nodes, config=None)
        assert vectors == {"Jane Doe": [8.0, 8.0, 8.0]}  # only the named entity, one embed call

        out = writer._attach_embeddings(nodes, vectors)
        assert out[0]["properties"]["embedding"] == [8.0, 8.0, 8.0]
        assert "embedding" not in out[1]["properties"]  # AttributeValue holders are not seeded on

    def test_empty_without_embedder(self):
        writer = make_writer()
        assert writer._embed_entity_names([_entity("PERSON", "u1", "Jane")], config=None) == {}

    def test_empty_on_non_neo4j_store(self):
        # Node embeddings are a Neo4j-only feature; on any other backend embedding is skipped so a raw
        # vector never lands as unqueryable dead weight (e.g. Postgres JSONB).
        writer = _embedder_writer()
        writer._graph_store = FakeGraphStore()  # not a Neo4jGraphStore
        assert writer._embed_entity_names([_entity("PERSON", "u1", "Jane")], config=None) == {}

    def test_degrades_without_raising_on_embedder_failure(self):
        writer = _embedder_writer(fail=True)
        writer._graph_store, _ = _recording_neo4j_store()
        assert writer._embed_entity_names([_entity("PERSON", "u1", "Jane")], config=None) == {}  # no raise

    def test_attach_embeddings_noop_without_vectors(self):
        writer = make_writer()
        nodes = [_entity("PERSON", "u1", "Jane")]
        assert "embedding" not in writer._attach_embeddings(nodes, {})[0]["properties"]

    def test_ensure_vector_index_uses_embedding_dimension(self):
        writer = KnowledgeGraphWriter(connection=_neo4j_conn(), is_postponed_component_init=True)
        store, recorded = _recording_neo4j_store()
        writer._graph_store = store

        writer._ensure_entity_vector_index(4)

        created = [q for q in recorded if "CREATE VECTOR INDEX" in q]
        assert created, "expected a CREATE VECTOR INDEX statement"
        assert ENTITY_EMBEDDING_VECTOR_INDEX in created[0]
        assert "`vector.dimensions`: 4" in created[0]
        assert "'cosine'" in created[0]
        assert ENTITY_EMBEDDING_VECTOR_INDEX in writer._vector_indexes_ready


class TestResolutionInternals:
    def test_deterministic_id_is_stable_and_label_scoped(self):
        w = make_writer()
        base = w._deterministic_id("ORGANIZATION", "Acme Capital")
        assert base == w._deterministic_id("ORGANIZATION", "acme   capital")  # case/whitespace-insensitive
        assert base != w._deterministic_id("PRODUCT", "Acme Capital")  # label-scoped
        assert is_uuid(base)

    def test_deterministic_id_is_the_base_even_with_fuzzy_off(self):
        # Deterministic id is ALWAYS assigned; fuzzy_matching only toggles the optional near-dup tier.
        w = make_writer(fuzzy=False)
        nodes = [entity("PERSON", "p@d1", "Jane Doe"), entity("PERSON", "p@d2", "Jane Doe")]
        resolved, _ = w._resolve_against_graph(nodes, [])
        assert resolved[0]["id"] == w._deterministic_id("PERSON", "Jane Doe")
        assert len({n["id"] for n in resolved}) == 1  # identical names collapse
        assert w._graph_store._stub.queries == []  # pure hash: no existence lookup, no fuzzy

    def test_existing_det_id_skips_fuzzy_and_is_not_re_merged(self):
        # Over-merge guard: an entity whose OWN deterministic node already exists is kept — a similar but
        # different candidate ("Acme Capital LLC") must not hijack an established node ("Acme LLC").
        w = make_writer()
        det = w._deterministic_id("ORGANIZATION", "Acme LLC")
        w._graph_store._stub.saved = {det}
        w._graph_store._stub.existing = {"ORGANIZATION": [("other-uuid", "Acme Capital LLC")]}

        resolved, _ = w._resolve_against_graph([entity("ORGANIZATION", "acme@d1", "Acme LLC")], [])

        assert resolved[0]["id"] == det  # kept, not merged into the near-dup
        assert not any("queryNodes" in q for q in w._graph_store._stub.queries)  # fuzzy never ran

    def test_fuzzy_off_issues_no_query_even_with_candidates(self):
        w = make_writer(fuzzy=False, existing={"ORGANIZATION": [("x", "Acme Capital")]})
        w._resolve_against_graph([entity("ORGANIZATION", "a@d1", "Acme Capital LLC")], [])
        assert w._graph_store._stub.queries == []  # no existence check, no blocking


class TestEmbedEdges:
    def test_attaches_triplet_embedding_to_edge_properties(self):
        writer = _embedder_writer(dim=3)
        writer._graph_store, _ = _recording_neo4j_store()
        rels = [edge("USES", "ORG", "SYS", "a", "b", {"src_name": "Acme", "dst_name": "Helios"})]

        writer._maybe_embed_edges(rels, config=None)

        # StubDocumentEmbedder embeds "{content}" -> [len(content)]*dim; "Acme USES Helios" has len 16.
        assert rels[0]["properties"]["embedding"] == [16.0, 16.0, 16.0]

    def test_noop_without_embedder(self):
        writer = make_writer()  # no entity_embedder
        rels = [edge("USES", "ORG", "SYS", "a", "b", {"src_name": "Acme", "dst_name": "Helios"})]
        writer._maybe_embed_edges(rels, config=None)
        assert "embedding" not in rels[0]["properties"]

    def test_edge_triplet_text_uses_attribute_key_for_has_attribute(self):
        rel = edge(
            HAS_ATTRIBUTE_TYPE,
            "PERSON",
            ATTRIBUTE_VALUE_LABEL,
            "p",
            "v",
            {"src_name": "Jane", "dst_name": "CTO", "key": "title"},
        )
        assert KnowledgeGraphWriter._edge_triplet_text(rel) == "Jane title CTO"

    def test_edge_triplet_text_includes_description(self):
        props = {"src_name": "Acme", "dst_name": "Helios", "description": "since 2020"}
        rel = edge("USES", "ORG", "SYS", "a", "b", props)
        assert KnowledgeGraphWriter._edge_triplet_text(rel) == "Acme USES Helios: since 2020"

    def test_edge_triplet_text_none_without_endpoint_names(self):
        assert KnowledgeGraphWriter._edge_triplet_text(edge("USES", "ORG", "SYS", "a", "b", {})) is None


class _DeleteStore:
    """Records delete_documents delegations; the store owns the deletion implementation."""

    def __init__(self, totals=None, error: Exception | None = None):
        self.totals = totals or {"relationships_deleted": 0, "nodes_deleted": 0}
        self.error = error
        self.calls: list[dict] = []

    def delete_documents(
        self,
        document_ids,
        *,
        doc_scoped_labels=None,
        orphan_labels=None,
        provenance_key="source_doc_id",
        database=None,
    ):
        self.calls.append(
            {
                "document_ids": document_ids,
                "doc_scoped_labels": doc_scoped_labels,
                "orphan_labels": orphan_labels,
                "provenance_key": provenance_key,
                "database": database,
            }
        )
        if self.error is not None:
            raise self.error
        return self.totals


class TestDeleteDocuments:
    def _writer_with(self, store):
        writer = KnowledgeGraphWriter(
            connection=Neo4j(uri="bolt://localhost:7687", username="neo4j", password="password"),
            is_postponed_component_init=True,
        )
        writer._graph_store = store
        return writer

    def test_delegates_to_the_store_with_attribute_value_scope(self):
        store = _DeleteStore(totals={"relationships_deleted": 3, "nodes_deleted": 2})
        writer = self._writer_with(store)
        out = writer.delete_documents(["docA", 7])
        assert out == {"relationships_deleted": 3, "nodes_deleted": 2}
        assert store.calls == [
            {
                "document_ids": ["docA", "7"],  # ids coerced to str, matching stored provenance values
                "doc_scoped_labels": ["AttributeValue"],
                "orphan_labels": ["Entity"],  # entities left edgeless by this delete are swept
                "provenance_key": "source_doc_id",
                "database": None,
            }
        ]

    def test_custom_provenance_key_passed_through(self):
        # e.g. catalyst deletes by knowledgebase file id: edges carry r.file_id via flattened chunk metadata
        store = _DeleteStore()
        writer = self._writer_with(store)
        writer.delete_documents(["file-1"], key="file_id")
        assert store.calls[0]["provenance_key"] == "file_id"

    def test_empty_ids_is_a_noop(self):
        store = _DeleteStore()
        writer = self._writer_with(store)
        assert writer.delete_documents([]) == {"relationships_deleted": 0, "nodes_deleted": 0}
        assert store.calls == []

    def test_unsupported_backend_error_propagates(self):
        # Non-writing backends raise from the store (base.delete_documents); the writer adds no gate.
        store = _DeleteStore(error=NotImplementedError("no writes"))
        writer = self._writer_with(store)
        with pytest.raises(NotImplementedError):
            writer.delete_documents(["docA"])
