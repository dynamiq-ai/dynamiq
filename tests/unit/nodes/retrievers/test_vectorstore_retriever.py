from dynamiq.nodes.retrievers.retriever import VectorStoreRetriever, VectorStoreRetrieverInputSchema
from dynamiq.runnables import RunnableConfig, RunnableStatus
from dynamiq.types import Document


def _retriever(**kwargs) -> VectorStoreRetriever:
    defaults = {
        "skip_empty_metadata": True,
        "is_optimized_for_agents": False,
    }
    return VectorStoreRetriever.model_construct(**(defaults | kwargs))


def test_format_content_skips_empty_metadata_and_rounds_score():
    retriever = _retriever()
    document = Document(
        content="Relevant chunk text.",
        score=0.756173968315,
        metadata={
            "doi": None,
            "file_path": "kb-uae-adgm-payroll-and-taxation.pdf",
            "title": "Payroll and Taxation",
            "city": "",
            "property": "Payroll",
        },
    )

    content = retriever.format_content([document])

    assert "Score: 0.756" in content
    assert "Doi:" not in content
    assert "City:" not in content
    assert content.index("File Path:") < content.index("Property:")


def test_format_content_metadata_fields_are_an_allowlist():
    retriever = _retriever()
    document = Document(
        content="Relevant chunk text.",
        score=0.9,
        metadata={
            "file_path": "source.pdf",
            "title": "Allowed title",
            "property": "Hidden property",
        },
    )

    content = retriever.format_content([document], metadata_fields=["title"])

    assert "Title: Allowed title" in content
    assert "File Path:" not in content
    assert "Property:" not in content
    assert "Score:" not in content


def test_format_content_empty_metadata_fields_does_not_fallback_to_all_metadata():
    retriever = _retriever()
    document = Document(
        content="Relevant chunk text.",
        score=0.9,
        metadata={
            "title": "Hidden title",
            "property": "Hidden property",
        },
    )

    content = retriever.format_content([document], metadata_fields=["", "missing"])

    assert "No metadata available." in content
    assert "Score:" not in content
    assert "Hidden title" not in content
    assert "Hidden property" not in content


def test_format_content_uses_retrieved_source_delimiters():
    retriever = _retriever(is_optimized_for_agents=True)
    document = Document(
        content="Agent-visible chunk.",
        score=0.91234,
        metadata={
            "file_path": "source.pdf",
            "title": None,
            "property": "Hidden property",
        },
    )

    content = retriever.format_content(
        [document],
        metadata_fields=["score", "file_path"],
    )

    assert content.startswith("--- Retrieved Source 1 ---\n")
    assert "Metadata:\nScore: 0.912" in content
    assert "\n\nContent:\nAgent-visible chunk." in content
    assert content.endswith("--- End Source 1 ---")
    assert "Score: 0.912" in content
    assert "File Path: source.pdf" in content
    assert "Hidden property" not in content
    assert "==" not in content


def test_agent_metadata_defaults_exclude_file_paths_and_ignore_missing_fields():
    retriever = _retriever(is_optimized_for_agents=True)
    document = Document(
        content="Agent-visible chunk.",
        score=0.91,
        metadata={
            "title": "Source Title",
            "file_path": "internal/path/source.pdf",
            "url": "https://example.com/source",
        },
    )

    content = retriever.format_content(
        [document],
        metadata_fields=retriever._resolve_formatted_metadata_fields(),
    )

    assert "Title: Source Title" in content
    assert "Url: https://example.com/source" in content
    assert "File Path:" not in content
    assert "Source Url:" not in content


def test_format_content_strips_outer_content_whitespace_only():
    retriever = _retriever(is_optimized_for_agents=True)
    document = Document(
        content="\n\nFirst paragraph.\n\nSecond paragraph.\n\n",
        score=0.5,
        metadata={"title": "Whitespace Example"},
    )

    content = retriever.format_content(
        [document],
        metadata_fields=retriever._resolve_formatted_metadata_fields(),
    )

    assert "Content:\nFirst paragraph.\n\nSecond paragraph.\n--- End Source 1 ---" in content


# --- locked_filters: a server-side ACL filter the runtime/LLM cannot drop ---

_LOCKED_ACL = {
    "operator": "AND",
    "conditions": [
        {"field": "metadata.acl_workspace_id", "operator": "==", "value": "ws:abc"},
        {"field": "metadata.allowed_principals", "operator": "in", "value": ["ws:abc:public"]},
    ],
}


def test_runtime_filters_cannot_drop_locked_filters():
    # Security contract (RFC README §9, 06 §10.2): a runtime filter supplied by the
    # prompt-injectable LLM must be AND-merged with the locked ACL filter, never
    # substituted for it. If this assertion can be made to fail, the ACL is bypassable.
    retriever = _retriever(filters={}, locked_filters=_LOCKED_ACL)
    llm_supplied = {"field": "metadata.source", "operator": "==", "value": "confluence"}

    effective = retriever._resolve_filters(llm_supplied)

    assert effective["operator"] == "AND"
    assert _LOCKED_ACL in effective["conditions"]
    assert llm_supplied in effective["conditions"]


def test_without_locked_filters_runtime_behaviour_is_unchanged():
    # Backward compatibility: with no locked_filters (the default for every existing KB),
    # the node must behave exactly as before — runtime filters win, self.filters is the fallback.
    retriever = _retriever(filters={"field": "metadata.lang", "operator": "==", "value": "en"})
    runtime = {"field": "metadata.source", "operator": "==", "value": "drive"}

    assert retriever._resolve_filters(runtime) == runtime
    assert retriever._resolve_filters({}) == retriever.filters


def test_locked_filters_apply_when_no_runtime_filters():
    # With a lock but no runtime/node filters, the lock alone is the effective filter.
    retriever = _retriever(filters={}, locked_filters=_LOCKED_ACL)

    assert retriever._resolve_filters({}) == _LOCKED_ACL
    assert retriever._resolve_filters(None) == _LOCKED_ACL


def test_no_filters_at_all_returns_empty():
    # No lock, no runtime, no node filters → empty (legacy: `{} or {}` == `{}`).
    retriever = _retriever(filters={})

    assert retriever._resolve_filters({}) == {}


def test_execute_passes_anded_filters_to_document_retriever(mocker):
    # The contract proven at the real execute() boundary: the filters that reach the
    # document retriever are AND(locked, llm_supplied) — the LLM cannot drop the ACL.
    retriever = _retriever(filters={}, locked_filters=_LOCKED_ACL, top_k=None, alpha=0.5)

    # Neutralize tracing/callbacks/cancellation that need real Node wiring; they're not under test.
    mocker.patch.object(VectorStoreRetriever, "run_on_node_execute_run")
    mocker.patch("dynamiq.nodes.retrievers.retriever.NodeDependency")
    mocker.patch("dynamiq.nodes.retrievers.retriever.check_cancellation")

    embedder = mocker.MagicMock()
    embedder.run.return_value = mocker.MagicMock(status=RunnableStatus.SUCCESS, output={"embedding": [0.1, 0.2]})
    retriever.text_embedder = embedder

    doc_retriever = mocker.MagicMock()
    doc_retriever.run.return_value = mocker.MagicMock(status=RunnableStatus.SUCCESS, output={"documents": []})
    retriever.document_retriever = doc_retriever
    retriever.document_reranker = None

    llm_supplied = {"field": "metadata.source", "operator": "==", "value": "confluence"}
    retriever.execute(
        VectorStoreRetrieverInputSchema(query="q", filters=llm_supplied),
        RunnableConfig(callbacks=[]),
    )

    passed_filters = doc_retriever.run.call_args.kwargs["input_data"]["filters"]
    assert passed_filters == {"operator": "AND", "conditions": [_LOCKED_ACL, llm_supplied]}
