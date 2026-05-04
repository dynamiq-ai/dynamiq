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
