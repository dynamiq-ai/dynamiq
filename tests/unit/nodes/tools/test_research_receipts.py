from unittest.mock import MagicMock

import pytest

from dynamiq.callbacks.base import BaseCallbackHandler
from dynamiq.connections import Exa, Firecrawl, Jina
from dynamiq.nodes.tools import ExaTool, FirecrawlTool, JinaScrapeTool
from dynamiq.runnables import RunnableConfig


class ReceiptRecorder(BaseCallbackHandler):
    def __init__(self):
        self.receipts = []

    def on_node_execute_run(self, serialized, **kwargs):
        if receipt := kwargs.get("research_receipt"):
            self.receipts.append(receipt)


@pytest.mark.parametrize("optimized", [False, True])
@pytest.mark.parametrize("provider", ["exa", "jina", "firecrawl"])
def test_receipts_come_from_transport_response_independently_of_rendering(mocker, provider, optimized):
    response = MagicMock(status_code=200)
    configurations = {
        "exa": (
            ExaTool(connection=Exa(api_key="test"), is_optimized_for_agents=optimized),
            {"query": "observed search"},
            {"results": [{"url": "https://authority.example/document", "text": "private-document"}]},
        ),
        "jina": (
            JinaScrapeTool(connection=Jina(api_key="test"), is_optimized_for_agents=optimized),
            {"url": "https://authority.example/document"},
            {"data": {"content": "private-document"}},
        ),
        "firecrawl": (
            FirecrawlTool(connection=Firecrawl(api_key="test"), is_optimized_for_agents=optimized),
            {"url": "https://authority.example/document"},
            {"success": True, "data": {"markdown": "private-document"}},
        ),
    }
    tool, arguments, payload = configurations[provider]
    response.json.return_value = payload
    mocker.patch("requests.request", return_value=response)
    recorder = ReceiptRecorder()
    tool.run(arguments, config=RunnableConfig(callbacks=[recorder]))
    assert len(recorder.receipts) == 1
    receipt = recorder.receipts[0]
    assert receipt["status"] == "succeeded"
    assert receipt["provider"] == provider
    assert "private-document" not in str(receipt)
    assert receipt.get("query") == arguments.get("query")
    assert receipt.get("url") == arguments.get("url")
    if provider == "exa":
        assert receipt["result_urls"] == ["https://authority.example/document"]
    assert "cost_usd" not in receipt


@pytest.mark.parametrize("provider", ["jina", "firecrawl"])
def test_http_success_with_empty_body_is_not_a_successful_fetch(mocker, provider):
    response = MagicMock(status_code=200)
    response.json.return_value = {"success": True, "data": {"content": "", "markdown": ""}}
    mocker.patch("requests.request", return_value=response)
    tool = (
        JinaScrapeTool(connection=Jina(api_key="test"))
        if provider == "jina"
        else FirecrawlTool(connection=Firecrawl(api_key="test"))
    )
    recorder = ReceiptRecorder()
    tool.run({"url": "https://authority.example"}, config=RunnableConfig(callbacks=[recorder]))
    assert recorder.receipts[0]["status"] == "failed"
    assert recorder.receipts[0]["error_code"] == "empty_content"


def test_transport_failure_receipt_excludes_exception_text(mocker):
    mocker.patch("requests.request", side_effect=RuntimeError("private-secret-failure"))
    recorder = ReceiptRecorder()
    ExaTool(connection=Exa(api_key="test")).run(
        {"query": "observed search"}, config=RunnableConfig(callbacks=[recorder])
    )
    assert recorder.receipts[0]["status"] == "failed"
    assert recorder.receipts[0]["error_code"] == "transport_error"
    assert "private-secret-failure" not in str(recorder.receipts)
