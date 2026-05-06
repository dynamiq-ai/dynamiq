from dynamiq.components.splitters.contextual import ContextualSplitterComponent
from dynamiq.nodes.splitters.contextual import ContextualSplitter
from dynamiq.runnables import RunnableConfig, RunnableResult, RunnableStatus
from dynamiq.types import Document


class _FakeInnerSplitter:
    def run(self, documents):
        chunks = []
        for doc in documents:
            for piece in doc.content.split("|"):
                chunks.append(Document(content=piece, metadata={"source_id": doc.id}))
        return {"documents": chunks}


class _FakeRunnableInnerSplitter:
    input_schema = object

    def __init__(self):
        self.run_calls = []
        self.execute_calls = 0

    def run(self, input_data, config=None, **kwargs):
        self.run_calls.append((input_data, config, kwargs))
        return RunnableResult(
            status=RunnableStatus.SUCCESS,
            output={"documents": [Document(content="child", metadata={"source_id": "source"})]},
        )

    def execute(self, input_data, config=None, **kwargs):
        self.execute_calls += 1
        raise AssertionError("execute should not be called directly")


class _FakeLLM:
    def __init__(self):
        self.run_calls = []
        self.execute_calls = 0

    def run(self, input_data, prompt, config=None, **kwargs):
        self.run_calls.append((input_data, prompt, config, kwargs))
        return RunnableResult(status=RunnableStatus.SUCCESS, output={"content": "ctx"})

    def execute(self, input_data, config=None, **kwargs):
        self.execute_calls += 1
        raise AssertionError("execute should not be called directly")

    def to_dict(self, **kwargs):
        return {"type": "fake-llm"}


def test_contextual_splitter_prepends_llm_context():
    calls = []

    def fake_llm(prompt: str) -> str:
        calls.append(prompt)
        return "Doc-level context."

    splitter = ContextualSplitterComponent(
        inner_splitter=_FakeInnerSplitter(),
        llm_fn=fake_llm,
        prepend=True,
    )
    output = splitter.run([Document(content="part-a|part-b")])
    assert len(output["documents"]) == 2
    for chunk in output["documents"]:
        assert chunk.content.startswith("Doc-level context.")
        assert chunk.metadata["context"] == "Doc-level context."
    assert len(calls) == 2


def test_contextual_splitter_caches_repeated_splits():
    call_count = {"n": 0}

    def fake_llm(prompt: str) -> str:
        call_count["n"] += 1
        return "ctx"

    splitter = ContextualSplitterComponent(
        inner_splitter=_FakeInnerSplitter(),
        llm_fn=fake_llm,
        cache_context=True,
    )
    splitter.run([Document(content="same|same")])
    assert call_count["n"] == 1


def test_contextual_splitter_cache_key_separates_document_and_chunk_text():
    calls = []

    def fake_llm(prompt: str) -> str:
        calls.append(prompt)
        return f"ctx-{len(calls)}"

    splitter = ContextualSplitterComponent(
        inner_splitter=_FakeInnerSplitter(),
        llm_fn=fake_llm,
        cache_context=True,
    )

    first = splitter._get_context("a:", "b")
    second = splitter._get_context("a", ":b")

    assert first == "ctx-1"
    assert second == "ctx-2"
    assert len(calls) == 2


def test_contextual_splitter_uses_run_lifecycle_for_node_like_inner_splitter():
    inner = _FakeRunnableInnerSplitter()
    config = RunnableConfig()
    splitter = ContextualSplitterComponent(
        inner_splitter=inner,
        llm_fn=lambda prompt: "ctx",
    )

    output = splitter.run([Document(id="source", content="parent")], config=config)

    assert len(output["documents"]) == 1
    assert inner.execute_calls == 0
    assert len(inner.run_calls) == 1
    assert inner.run_calls[0][0]["documents"][0].content == "parent"
    assert inner.run_calls[0][1] is config


def test_contextual_splitter_node_uses_run_lifecycle_for_llm():
    llm = _FakeLLM()
    splitter = ContextualSplitter.model_construct(llm=llm)

    context = splitter._call_llm("Prompt text", config=RunnableConfig())

    assert context == "ctx"
    assert llm.execute_calls == 0
    assert len(llm.run_calls) == 1


def test_contextual_splitter_llm_call_removes_duplicate_run_kwargs():
    llm = _FakeLLM()
    config = RunnableConfig()
    splitter = ContextualSplitter.model_construct(llm=llm)

    context = splitter._call_llm(
        "Prompt text",
        config=config,
        input_data={"ignored": True},
        prompt="ignored",
        run_depends=[{"ignored": True}],
        parent_run_id="parent",
        run_id="run",
    )

    assert context == "ctx"
    _, _, call_config, call_kwargs = llm.run_calls[0]
    assert call_config is config
    assert call_kwargs["run_depends"] == []
    assert call_kwargs["parent_run_id"] == "parent"
    assert "run_id" not in call_kwargs
    assert "input_data" not in call_kwargs
    assert "prompt" not in call_kwargs


def test_contextual_splitter_llm_call_converts_run_id_to_parent_run_id():
    llm = _FakeLLM()
    splitter = ContextualSplitter.model_construct(llm=llm)

    context = splitter._call_llm(
        "Prompt text",
        config=RunnableConfig(),
        parent_run_id=None,
        run_id="parent-run",
    )

    assert context == "ctx"
    _, _, _, call_kwargs = llm.run_calls[0]
    assert call_kwargs["parent_run_id"] == "parent-run"
    assert "run_id" not in call_kwargs
