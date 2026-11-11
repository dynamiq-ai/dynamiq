from unittest.mock import patch

from dynamiq import connections
from dynamiq.nodes.llms import OpenAI
from dynamiq.nodes.rankers import LLMDocumentRanker
from dynamiq.runnables import RunnableConfig
from dynamiq.types import Document


def _ranker(top_k: int) -> LLMDocumentRanker:
    llm = OpenAI(model="gpt-4o-mini", connection=connections.OpenAI(api_key="test-key"))
    return LLMDocumentRanker(llm=llm, top_k=top_k)


def test_llm_ranker_respects_top_k():
    documents = [Document(content=f"doc {i}") for i in range(6)]
    ranker = _ranker(top_k=2)

    with patch.object(LLMDocumentRanker, "call_llm", return_value="Yes"):
        result = ranker.perform_llm_ranking("query", documents, RunnableConfig())

    assert [d.content for d in result] == ["doc 0", "doc 1"]


def test_llm_ranker_returns_fewer_than_top_k_when_few_selected():
    documents = [Document(content=f"doc {i}") for i in range(4)]
    ranker = _ranker(top_k=3)

    # call_llm runs in a thread pool, so answer by passage rather than by call order.
    def answer(input_data, *args, **kwargs):
        return "Yes" if input_data["passage"] == "doc 1" else "No"

    with patch.object(LLMDocumentRanker, "call_llm", side_effect=answer):
        result = ranker.perform_llm_ranking("query", documents, RunnableConfig())

    assert [d.content for d in result] == ["doc 1"]
