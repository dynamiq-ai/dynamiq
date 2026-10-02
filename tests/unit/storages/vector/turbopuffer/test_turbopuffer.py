import base64
import struct
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import MagicMock

import httpx
import pytest
import turbopuffer
from turbopuffer.types import Row

from dynamiq.storages.vector.exceptions import (
    VectorStoreDuplicateDocumentException,
    VectorStoreException,
    VectorStoreFilterException,
)
from dynamiq.storages.vector.policies import DuplicatePolicy
from dynamiq.storages.vector.turbopuffer import TurbopufferVectorStore
from dynamiq.types import Document

SCHEMA = {
    "id": SimpleNamespace(model_dump=lambda exclude_none: {"type": "string"}),
    "vector": SimpleNamespace(model_dump=lambda exclude_none: {"type": "[3]f32"}),
    "content": SimpleNamespace(
        model_dump=lambda exclude_none: {"type": "string", "filterable": False, "full_text_search": {"k1": 1.2}}
    ),
    "title": SimpleNamespace(
        model_dump=lambda exclude_none: {"type": "string", "filterable": True, "full_text_search": {"k1": 1.2}}
    ),
    "file_id": SimpleNamespace(model_dump=lambda exclude_none: {"type": "string", "filterable": True}),
    "page_number": SimpleNamespace(model_dump=lambda exclude_none: {"type": "int", "filterable": True}),
    "acl": SimpleNamespace(model_dump=lambda exclude_none: {"type": "[]string", "filterable": True}),
}


def _error(cls, status):
    return cls("error", response=httpx.Response(status, request=httpx.Request("POST", "https://tpuf")), body=None)


def _row(id_, dist=None, **attributes):
    data = {"id": id_, **attributes}
    if dist is not None:
        data["$dist"] = dist
    return Row.model_validate(data)


@pytest.fixture
def namespace():
    ns = MagicMock()
    ns.schema.return_value = dict(SCHEMA)
    ns.write.return_value = SimpleNamespace(rows_affected=0)
    ns.query.return_value = SimpleNamespace(rows=[], aggregations=None)
    return ns


@pytest.fixture
def store(namespace):
    client = MagicMock()
    client.namespace.return_value = namespace
    return TurbopufferVectorStore(client=client, index_name="vs-1")


@pytest.mark.parametrize("name", ["", "has space", "a/b", "kb:1", "x" * 129])
def test_init_rejects_invalid_namespace_name(name):
    with pytest.raises(ValueError):
        TurbopufferVectorStore(client=MagicMock(), index_name=name)


def test_init_tracks_new_namespace_for_dry_run(namespace):
    client = MagicMock()
    client.namespace.return_value = namespace
    namespace.exists.return_value = False

    store = TurbopufferVectorStore(client=client, index_name="vs-1", create_if_not_exist=True)

    assert store._tracked_collection == "vs-1"


def test_write_documents_builds_rows_and_declares_new_attributes(store, namespace):
    namespace.write.return_value = SimpleNamespace(rows_affected=1)
    document = Document(
        id="d1",
        content="hello",
        embedding=[0.5, 0.25, 1.0],
        metadata={
            "file_id": "f1",
            "page_number": "2",
            "score": 3.0,
            "acl": [],
            "info": {"w": 612},
            "filename": "a.pdf",
            "none": None,
            "$reserved": 1,
            "vector": [1],
        },
    )

    assert store.write_documents([document]) == 1

    params = namespace.write.call_args.kwargs
    row = params["upsert_rows"][0]
    assert base64.b64decode(row.pop("vector")) == struct.pack("<3f", 0.5, 0.25, 1.0)
    assert row == {
        "id": "d1",
        "content": "hello",
        "file_id": "f1",
        "page_number": 2,
        "score": 3.0,
        "acl": [],
        "info": '{"w": 612}',
        "filename": "a.pdf",
    }
    assert params["distance_metric"] == "cosine_distance"
    assert params["schema"] == {
        "score": {"type": "float"},
        "info": {"type": "string"},
        "filename": {
            "type": "string",
            "filterable": True,
            "full_text_search": {
                "tokenizer": "word_v4",
                "language": "english",
                "stemming": False,
                "remove_stopwords": False,
                "case_sensitive": False,
            },
        },
    }
    assert "upsert_condition" not in params


def test_write_documents_into_new_namespace_declares_content_for_search(store, namespace):
    namespace.schema.side_effect = _error(turbopuffer.NotFoundError, 404)

    store.write_documents([Document(id="d1", content="hello", embedding=[0.1, 0.2, 0.3], metadata={"acl": []})])

    schema = namespace.write.call_args.kwargs["schema"]
    assert schema["content"]["type"] == "string"
    assert schema["content"]["full_text_search"]["tokenizer"] == "word_v4"
    assert schema["acl"] == {"type": "[]string"}


def test_write_documents_makes_long_strings_unfilterable(store, namespace):
    store.write_documents([Document(id="d1", content="c", metadata={"file_id": "x" * 5000, "notes": "y" * 5000})])

    schema = namespace.write.call_args.kwargs["schema"]
    assert schema["file_id"] == {"type": "string", "filterable": False}
    assert schema["notes"] == {"type": "string", "filterable": False}


def test_write_documents_drops_values_that_do_not_fit_the_schema(store, namespace):
    store.write_documents([Document(id="d1", content="c", metadata={"page_number": 7.5, "file_id": 42})])

    row = namespace.write.call_args.kwargs["upsert_rows"][0]
    assert "page_number" not in row
    assert row["file_id"] == "42"


def test_write_documents_keeps_last_copy_of_duplicate_ids_and_batches(store, namespace):
    store.batch_size = 2
    documents = [Document(id=f"d{i}", content=f"c{i}") for i in range(3)] + [Document(id="d0", content="latest")]

    store.write_documents(documents)

    batches = [call.kwargs["upsert_rows"] for call in namespace.write.call_args_list]
    assert [[row["id"] for row in batch] for batch in batches] == [["d0", "d1"], ["d2"]]
    assert batches[0][0]["content"] == "latest"
    assert "distance_metric" not in namespace.write.call_args_list[0].kwargs


def test_write_documents_skip_policy_inserts_only_missing_ids(store, namespace):
    store.write_documents([Document(id="d1", content="c")], policy=DuplicatePolicy.SKIP)

    assert namespace.write.call_args.kwargs["upsert_condition"] == ["id", "Eq", None]


def test_write_documents_fail_policy_raises_for_existing_ids(store, namespace):
    namespace.query.return_value = SimpleNamespace(rows=[_row("d1")])

    with pytest.raises(VectorStoreDuplicateDocumentException):
        store.write_documents([Document(id="d1", content="c")], policy=DuplicatePolicy.FAIL)

    namespace.write.assert_not_called()


def test_write_documents_retries_once_with_a_fresh_schema(store, namespace):
    namespace.write.side_effect = [_error(turbopuffer.BadRequestError, 400), SimpleNamespace(rows_affected=1)]

    assert store.write_documents([Document(id="d1", content="c")]) == 1
    assert namespace.schema.call_count == 2


def test_write_documents_raises_when_turbopuffer_keeps_rejecting(store, namespace):
    namespace.write.side_effect = _error(turbopuffer.BadRequestError, 400)

    with pytest.raises(VectorStoreException):
        store.write_documents([Document(id="d1", content="c")])


def test_write_documents_rejects_long_ids(store):
    with pytest.raises(ValueError):
        store.write_documents([Document(id="x" * 65, content="c")])


def test_embedding_retrieval_scores_and_limits(store, namespace):
    namespace.query.return_value = SimpleNamespace(
        rows=[_row("d1", dist=0.0, content="one", file_id="f1"), _row("d2", dist=1.0, content="two")]
    )

    documents = store._embedding_retrieval(
        [0.1, 0.2, 0.3], filters={"file_id": ["f1", "f2"]}, top_k=2, max_distance=0.5
    )

    params = namespace.query.call_args.kwargs
    assert params["rank_by"] == ["vector", "ANN", [0.1, 0.2, 0.3]]
    assert params["top_k"] == 2
    assert params["filters"] == ["file_id", "In", ["f1", "f2"]]
    assert params["exclude_attributes"] == ["vector"]
    assert [(d.id, d.score, d.content, d.metadata) for d in documents] == [("d1", 1.0, "one", {"file_id": "f1"})]


def test_embedding_retrieval_skips_query_when_filters_match_nothing(store, namespace):
    assert store._embedding_retrieval([0.1, 0.2, 0.3], filters={"missing": "x"}) == []
    namespace.query.assert_not_called()


def test_retrieval_on_missing_namespace_returns_nothing(store, namespace):
    namespace.query.side_effect = _error(turbopuffer.NotFoundError, 404)
    namespace.multi_query.side_effect = _error(turbopuffer.NotFoundError, 404)

    assert store._embedding_retrieval([0.1, 0.2, 0.3]) == []
    assert store._keyword_retrieval("query") == []
    assert store._hybrid_retrieval([0.1, 0.2, 0.3], "query") == []
    assert store.get_documents_by_id(["d1"]) == []
    assert store.count_documents() == 0


def test_keyword_retrieval_sums_searchable_attributes(store, namespace):
    namespace.query.return_value = SimpleNamespace(rows=[_row("d1", dist=2.5, content="fox")])

    documents = store._keyword_retrieval("fox", top_k=3)

    assert namespace.query.call_args.kwargs["rank_by"] == [
        "Sum",
        [["content", "BM25", "fox"], ["title", "BM25", "fox"]],
    ]
    assert [(d.id, d.score) for d in documents] == [("d1", 2.5)]


def test_keyword_retrieval_ignores_blank_query(store, namespace):
    assert store._keyword_retrieval("  ") == []
    namespace.query.assert_not_called()


def test_hybrid_retrieval_fuses_relative_scores(store, namespace):
    namespace.multi_query.return_value = SimpleNamespace(
        results=[
            SimpleNamespace(rows=[_row("a", dist=0.1), _row("b", dist=0.3), _row("c", dist=0.5)]),
            SimpleNamespace(rows=[_row("c", dist=4.0), _row("d", dist=2.0)]),
        ]
    )

    documents = store._hybrid_retrieval([0.1, 0.2, 0.3], "fox", top_k=3, alpha=0.75)

    queries = namespace.multi_query.call_args.kwargs["queries"]
    assert queries[0]["rank_by"][:2] == ["vector", "ANN"]
    assert queries[1]["rank_by"][0] == "Sum"
    assert [(d.id, round(d.score, 4)) for d in documents] == [("a", 0.75), ("b", 0.375), ("c", 0.25)]


def test_hybrid_retrieval_drops_distant_vector_matches(store, namespace):
    namespace.multi_query.return_value = SimpleNamespace(
        results=[SimpleNamespace(rows=[_row("a", dist=0.1), _row("b", dist=0.9)]), SimpleNamespace(rows=[])]
    )

    documents = store._hybrid_retrieval([0.1, 0.2, 0.3], "fox", alpha=1.0, max_vector_distance=0.5)

    assert [(d.id, d.score) for d in documents] == [("a", 1.0)]


def test_get_documents_by_id(store, namespace):
    namespace.query.return_value = SimpleNamespace(rows=[_row("d1", content="one", vector=[0.1, 0.2, 0.3])])

    documents = store.get_documents_by_id(["d1", "d1", "d2"], include_embeddings=True)

    params = namespace.query.call_args.kwargs
    assert params["filters"] == ["id", "In", ["d1", "d2"]]
    assert params["include_attributes"] is True
    assert [(d.id, d.content, d.embedding) for d in documents] == [("d1", "one", [0.1, 0.2, 0.3])]


def test_filter_documents_pages_through_ids(store, namespace, monkeypatch):
    monkeypatch.setattr("dynamiq.storages.vector.turbopuffer.turbopuffer.MAX_TOP_K", 2)
    namespace.query.side_effect = [
        SimpleNamespace(rows=[_row("a"), _row("b")]),
        SimpleNamespace(rows=[_row("c")]),
    ]

    documents = store.filter_documents({"file_id": "f1"})

    assert [d.id for d in documents] == ["a", "b", "c"]
    second = namespace.query.call_args_list[1].kwargs
    assert second["filters"] == ["And", [["file_id", "Eq", "f1"], ["id", "Gt", "b"]]]


def test_count_documents(store, namespace):
    namespace.query.return_value = SimpleNamespace(aggregations={"count": 7})

    assert store.count_documents() == 7


def test_delete_documents_by_filters(store, namespace):
    store.delete_documents_by_file_ids(["f1", "f2"])

    assert namespace.write.call_args.kwargs == {"delete_by_filter": ["file_id", "In", ["f1", "f2"]]}


def test_delete_documents_by_filters_skips_unmatchable_filters(store, namespace):
    store.delete_documents_by_filters({"missing": "x"})

    namespace.write.assert_not_called()


def test_delete_documents_by_filters_requires_filters(store):
    with pytest.raises(ValueError):
        store.delete_documents_by_filters({})


def test_delete_documents(store, namespace):
    store.delete_documents(["d1", "d2", "d1"])

    assert namespace.write.call_args.kwargs == {"deletes": ["d1", "d2"]}


def test_delete_all_documents_deletes_namespace(store, namespace):
    store.delete_documents(delete_all=True)

    namespace.delete_all.assert_called_once()


def test_delete_collection_tolerates_missing_namespace(store, namespace):
    namespace.delete_all.side_effect = _error(turbopuffer.NotFoundError, 404)

    store.delete_collection()


def test_replace_document_metadata(store, namespace):
    namespace.query.return_value = SimpleNamespace(rows=[_row("d1", content="one", vector=[0.1, 0.2, 0.3])])

    store.replace_document_metadata("d1", {"file_id": "f9"})

    row = namespace.write.call_args.kwargs["upsert_rows"][0]
    assert (row["id"], row["content"], row["file_id"]) == ("d1", "one", "f9")
    assert "vector" in row


def test_replace_document_metadata_raises_for_missing_ids(store, namespace):
    with pytest.raises(VectorStoreException):
        store.replace_document_metadata(["d1"], {"file_id": "f9"})


def _schema(**attributes):
    return {name: SimpleNamespace(model_dump=lambda exclude_none, c=config: c) for name, config in attributes.items()}


def test_filters_refresh_a_schema_missing_the_attribute(store, namespace):
    namespace.schema.side_effect = [{}, _schema(file_id={"type": "string", "filterable": True})]
    namespace.query.return_value = SimpleNamespace(rows=[_row("d1", dist=0.0)])

    documents = store._embedding_retrieval([0.1, 0.2, 0.3], filters={"file_id": "f1"})

    assert namespace.schema.call_count == 2
    assert namespace.query.call_args.kwargs["filters"] == ["file_id", "Eq", "f1"]
    assert [d.id for d in documents] == ["d1"]


def test_keyword_retrieval_refreshes_a_schema_without_searchable_content(store, namespace):
    namespace.schema.side_effect = [{}, _schema(content={"type": "string", "full_text_search": {"k1": 1.2}})]
    namespace.query.return_value = SimpleNamespace(rows=[_row("d1", dist=1.5)])

    documents = store._keyword_retrieval("fox")

    assert namespace.query.call_args.kwargs["rank_by"] == ["content", "BM25", "fox"]
    assert [d.id for d in documents] == ["d1"]


def test_rejected_read_is_rebuilt_from_a_fresh_schema(store, namespace):
    stale = _schema(file_id={"type": "string", "filterable": True})
    fresh = _schema(file_id={"type": "string", "filterable": False})
    namespace.schema.side_effect = [stale, fresh]
    namespace.query.side_effect = _error(turbopuffer.BadRequestError, 400)

    with pytest.raises(VectorStoreFilterException, match="not filterable"):
        store._embedding_retrieval([0.1, 0.2, 0.3], filters={"file_id": "f1"})

    assert namespace.query.call_count == 1


def test_read_rejected_twice_raises(store, namespace):
    namespace.query.side_effect = _error(turbopuffer.BadRequestError, 400)

    with pytest.raises(VectorStoreException):
        store._embedding_retrieval([0.1, 0.2, 0.3])

    assert namespace.query.call_count == 2


def test_delete_by_unfilterable_attribute_raises_instead_of_deleting(store, namespace):
    namespace.schema.return_value = _schema(notes={"type": "string", "filterable": False})

    with pytest.raises(VectorStoreFilterException):
        store.delete_documents_by_filters({"field": "notes", "operator": "!=", "value": "x"})

    namespace.write.assert_not_called()


def test_delete_by_filters_matching_every_document_raises(store, namespace):
    with pytest.raises(VectorStoreFilterException, match="match every document"):
        store.delete_documents_by_filters({"field": "page_number", "operator": "!=", "value": "five"})

    namespace.write.assert_not_called()


def test_delete_by_filters_normalizes_values_like_writes(store, namespace):
    store.delete_documents_by_filters({"field": "page_number", "operator": "!=", "value": Decimal(3)})

    assert namespace.write.call_args.kwargs == {"delete_by_filter": ["page_number", "NotEq", 3]}
