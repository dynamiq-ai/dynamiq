from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import ValidationError

from dynamiq import Workflow
from dynamiq.callbacks.base import BaseCallbackHandler
from dynamiq.connections import Exa, Firecrawl, Jina
from dynamiq.flows import Flow
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


@pytest.mark.asyncio
@pytest.mark.parametrize("async_mode", [False, True])
@pytest.mark.parametrize(
    "format_,data,success",
    [
        ("screenshot", {"screenshotUrl": "https://asset.example/private-image.png"}, True),
        ("pageshot", {"pageshotUrl": "https://asset.example/private-image.png"}, True),
        ("screenshot", {"screenshotUrl": " "}, False),
        ("screenshot", {"content": "unexpected text"}, False),
        ("markdown", {"screenshotUrl": "https://asset.example/private-image.png"}, False),
        ("html", {"html": "<p>document text</p>"}, True),
        ("text", {"text": "document text"}, True),
    ],
)
async def test_jina_receipt_tracks_its_requested_representation_without_retaining_assets(
    mocker, async_mode, format_, data, success
):
    response = MagicMock(status_code=200, headers={"X-Usage-Tokens": "7"})
    response.json.return_value = {"data": data}
    tool = JinaScrapeTool(connection=Jina(api_key="test"), response_format=format_, unit_price_usd=0.01)
    recorder = ReceiptRecorder()
    config = RunnableConfig(callbacks=[recorder])
    if async_mode:
        request = AsyncMock(return_value=response)
        mocker.patch.object(type(tool), "get_async_client", AsyncMock(return_value=MagicMock(request=request)))
        result = await tool.run_async({"url": "https://authority.example"}, config=config)
    else:
        request = mocker.patch("requests.request", return_value=response)
        result = tool.run_sync({"url": "https://authority.example"}, config=config)
    assert result.status.value == "success"
    request.assert_called_once()
    (receipt,) = recorder.receipts
    assert receipt["status"] == ("succeeded" if success else "failed")
    assert receipt["usage_units"] == 7
    assert Decimal(receipt["cost_usd"]) == Decimal("0.07")
    assert receipt["url"] == "https://authority.example"
    assert "private-image" not in str(receipt)
    assert "document text" not in str(receipt)


@pytest.mark.asyncio
@pytest.mark.parametrize("async_mode", [False, True])
@pytest.mark.parametrize(
    "formats,data,success",
    [
        (["links"], {"links": ["https://asset.example/private-link"]}, True),
        (["links"], {"links": []}, True),
        (["links"], {"links": {}}, False),
        (["summary"], {"summary": "private summary"}, True),
        (["summary"], {"summary": " "}, False),
        (["screenshot"], {"screenshot": "https://asset.example/private-image.png"}, True),
        ([{"type": "screenshot", "fullPage": True}], {"screenshot": "https://asset.example/private-image.png"}, True),
        (["screenshot"], {"screenshot": " "}, False),
        ([{"type": "json"}], {"json": {}}, True),
        ([{"type": "json"}], {"json": False}, True),
        ([{"type": "json"}], {"json": 0}, True),
        ([{"type": "json", "schema": {"type": ["object", "null"]}}], {"json": None}, True),
        ([{"type": "json"}], {}, False),
        ([{"type": []}], {"json": {"private": "unrecognized representation"}}, False),
        (["branding"], {"branding": {}}, True),
        (["changeTracking"], {"changeTracking": {}}, True),
        (["markdown", "links"], {"markdown": "", "links": ["https://asset.example/private-link"]}, False),
        (["markdown", "screenshot"], {"markdown": "", "screenshot": "https://asset.example/private-image.png"}, False),
        (["html", "json"], {"html": "", "json": {"private": "extracted data"}}, False),
        (["markdown", "links"], {"markdown": "private document", "links": []}, True),
        (["markdown", "html"], {"markdown": "", "html": "<p>private document</p>"}, True),
        (["html"], {"html": "<p>access denied</p>", "metadata": {"statusCode": 403}}, False),
        (["markdown"], {"markdown": "private document", "metadata": {"statusCode": 404}}, False),
        (["html"], {"html": "<p>private document</p>", "metadata": {"statusCode": 200}}, True),
        (["markdown"], {"markdown": "private cached document", "metadata": {"statusCode": 304}}, True),
    ],
)
async def test_firecrawl_requested_formats_keep_text_read_grounding_separate_from_nontext_success(
    mocker, async_mode, formats, data, success
):
    response = MagicMock(status_code=200, headers={})
    response.json.return_value = {
        "success": True,
        "data": {**data, "metadata": {"creditsUsed": 7, **data.get("metadata", {})}},
    }
    tool = FirecrawlTool(connection=Firecrawl(api_key="test"), formats=formats, unit_price_usd=0.01)
    recorder = ReceiptRecorder()
    config = RunnableConfig(callbacks=[recorder])
    if async_mode:
        request = AsyncMock(return_value=response)
        mocker.patch.object(type(tool), "get_async_client", AsyncMock(return_value=MagicMock(request=request)))
        result = await tool.run_async({"url": "https://authority.example/notice.pdf"}, config=config)
    else:
        request = mocker.patch("requests.request", return_value=response)
        result = tool.run_sync({"url": "https://authority.example/notice.pdf"}, config=config)
    assert result.status.value == "success"
    request.assert_called_once()
    (receipt,) = recorder.receipts
    assert receipt["status"] == ("succeeded" if success else "failed")
    assert receipt["usage_units"] == 7
    assert Decimal(receipt["cost_usd"]) == Decimal("0.07")
    assert "private" not in str(receipt)
    assert receipt["url"] == "https://authority.example/notice.pdf"


@pytest.mark.asyncio
@pytest.mark.parametrize("async_mode", [False, True])
async def test_firecrawl_uses_per_call_hidden_formats_without_changing_the_shared_configuration(mocker, async_mode):
    tool = FirecrawlTool(connection=Firecrawl(api_key="test"), formats=["markdown", "links"])
    response = MagicMock(status_code=200, headers={})
    response.json.return_value = {"success": True, "data": {"links": []}}
    recorder = ReceiptRecorder()
    config = RunnableConfig(callbacks=[recorder])
    if async_mode:
        request = AsyncMock(return_value=response)
        mocker.patch.object(type(tool), "get_async_client", AsyncMock(return_value=MagicMock(request=request)))
        await tool.run_async({"url": "https://authority.example", "formats": ["links"]}, config=config)
        await tool.run_async({"url": "https://authority.example"}, config=config)
    else:
        request = mocker.patch("requests.request", return_value=response)
        tool.run_sync({"url": "https://authority.example", "formats": ["links"]}, config=config)
        tool.run_sync({"url": "https://authority.example"}, config=config)
    assert [item["status"] for item in recorder.receipts] == ["succeeded", "failed"]
    assert [call.kwargs["json"]["formats"] for call in request.call_args_list] == [["links"], ["markdown", "links"]]
    assert tool.formats == ["markdown", "links"]


@pytest.mark.asyncio
@pytest.mark.parametrize("async_mode", [False, True])
async def test_jina_unexpected_binary_response_stays_failed_and_never_enters_receipt(mocker, async_mode):
    response = MagicMock(status_code=200, headers={"Content-Type": "image/png", "X-Usage-Tokens": "7"})
    response.content = b"private-image-bytes"
    response.json.side_effect = ValueError("not JSON")
    tool = JinaScrapeTool(connection=Jina(api_key="test"), response_format="screenshot", unit_price_usd=0.01)
    recorder = ReceiptRecorder()
    config = RunnableConfig(callbacks=[recorder])
    if async_mode:
        request = AsyncMock(return_value=response)
        mocker.patch.object(type(tool), "get_async_client", AsyncMock(return_value=MagicMock(request=request)))
        result = await tool.run_async({"url": "https://authority.example"}, config=config)
    else:
        request = mocker.patch("requests.request", return_value=response)
        result = tool.run_sync({"url": "https://authority.example"}, config=config)
    assert result.status.value == "failure"
    request.assert_called_once()
    (receipt,) = recorder.receipts
    assert receipt["status"] == "failed"
    assert "cost_usd" not in receipt
    assert "usage_units" not in receipt
    assert "private-image-bytes" not in str(receipt)


def test_transport_failure_receipt_excludes_exception_text(mocker):
    mocker.patch("requests.request", side_effect=RuntimeError("private-secret-failure"))
    recorder = ReceiptRecorder()
    ExaTool(connection=Exa(api_key="test")).run(
        {"query": "observed search"}, config=RunnableConfig(callbacks=[recorder])
    )
    assert recorder.receipts[0]["status"] == "failed"
    assert recorder.receipts[0]["error_code"] == "transport_error"
    assert "private-secret-failure" not in str(recorder.receipts)


@pytest.mark.asyncio
@pytest.mark.parametrize("async_mode", [False, True])
@pytest.mark.parametrize("provider", ["exa", "jina", "firecrawl"])
@pytest.mark.parametrize(
    "value,known_cost,known_units",
    [
        (0, "0", 0),
        (17, "17", 17),
        (2**63, str(2**63), None),
        (True, None, None),
        (-1, None, None),
        (1.5, "1.5", None),
        (float("nan"), None, None),
        (float("inf"), None, None),
        ("23", "23", None),
        ("invalid", None, None),
        (None, None, None),
        ("1e999999999", None, None),
        ("1e-999999999", None, None),
        ("9" * 65, None, None),
        ("1e64", "1e64", None),
        ("1e-64", "1e-64", None),
    ],
)
async def test_reported_accounting_is_preserved_without_assuming_unknown_prices(
    mocker, provider, value, known_cost, known_units, async_mode
):
    response = MagicMock(status_code=200, headers={})
    if provider == "exa":
        tool = ExaTool(connection=Exa(api_key="test"))
        arguments = {"query": "observed search"}
        payload = {"results": [], "costDollars": {"total": value}}
    elif provider == "jina":
        tool = JinaScrapeTool(connection=Jina(api_key="test"))
        arguments = {"url": "https://authority.example/document"}
        payload = {"data": {"content": "private-document", "usage": {"tokens": value}}}
    else:
        tool = FirecrawlTool(connection=Firecrawl(api_key="test"))
        arguments = {"url": "https://authority.example/document"}
        payload = {"success": True, "data": {"markdown": "private-document", "metadata": {"creditsUsed": value}}}
    response.json.return_value = payload
    recorder = ReceiptRecorder()
    config = RunnableConfig(callbacks=[recorder])
    if async_mode:
        client = MagicMock(request=AsyncMock(return_value=response))
        mocker.patch.object(type(tool), "get_async_client", AsyncMock(return_value=client))
        result = await tool.run_async(arguments, config=config)
    else:
        mocker.patch("requests.request", return_value=response)
        result = tool.run_sync(arguments, config=config)
    assert result.status.value == "success"
    (receipt,) = recorder.receipts
    if provider == "exa":
        assert (Decimal(receipt["cost_usd"]) if "cost_usd" in receipt else None) == (
            Decimal(known_cost) if known_cost is not None else None
        )
        assert "usage_units" not in receipt
    else:
        assert receipt.get("usage_units") == known_units
        assert receipt.get("usage_unit") == (
            ("tokens" if provider == "jina" else "credits") if known_units is not None else None
        )
        assert "cost_usd" not in receipt
    assert "private-document" not in str(receipt)


@pytest.mark.asyncio
@pytest.mark.parametrize("async_mode", [False, True])
@pytest.mark.parametrize("header,expected", [("0", 0), ("23", 23), (" 0023 ", 23), ("1.5", 7), ("NaN", 7), (True, 7)])
async def test_jina_prefers_valid_integer_header_and_falls_back_to_json(mocker, async_mode, header, expected):
    response = MagicMock(status_code=200, headers={"X-Usage-Tokens": header})
    response.json.return_value = {"data": {"content": "", "usage": {"tokens": 7}}}
    tool = JinaScrapeTool(connection=Jina(api_key="test"))
    recorder = ReceiptRecorder()
    config = RunnableConfig(callbacks=[recorder])
    arguments = {"url": "https://authority.example/document"}
    if async_mode:
        mocker.patch.object(
            JinaScrapeTool,
            "get_async_client",
            AsyncMock(return_value=MagicMock(request=AsyncMock(return_value=response))),
        )
        await tool.run_async(arguments, config=config)
    else:
        mocker.patch("requests.request", return_value=response)
        tool.run_sync(arguments, config=config)
    (receipt,) = recorder.receipts
    assert receipt["usage_units"] == expected
    assert receipt["usage_unit"] == "tokens"
    assert receipt["status"] == "failed"
    assert receipt["error_code"] == "empty_content"
    assert "cost_usd" not in receipt


@pytest.mark.asyncio
@pytest.mark.parametrize("async_mode", [False, True])
@pytest.mark.parametrize("provider", ["jina", "firecrawl"])
@pytest.mark.parametrize(
    "price,units,expected",
    [(None, 3, None), (0, 3, "0"), (0.125, 3, "0.375"), (0.125, 0, "0"), (0.125, None, None), (0.125, True, None)],
)
async def test_only_declared_static_price_and_observed_units_estimate_cost(
    mocker, async_mode, provider, price, units, expected
):
    response = MagicMock(status_code=200, headers={})
    if provider == "jina":
        tool = JinaScrapeTool(connection=Jina(api_key="test"), unit_price_usd=price)
        response.json.return_value = {"data": {"content": "page", "usage": {"tokens": units}}}
    else:
        tool = FirecrawlTool(connection=Firecrawl(api_key="test"), unit_price_usd=price)
        response.json.return_value = {"success": True, "data": {"markdown": "page", "metadata": {"creditsUsed": units}}}
    recorder = ReceiptRecorder()
    arguments = {"url": "https://authority.example", "unit_price_usd": 999}
    assert "unit_price_usd" not in tool.input_schema.model_fields
    if async_mode:
        request = AsyncMock(return_value=response)
        mocker.patch.object(type(tool), "get_async_client", AsyncMock(return_value=MagicMock(request=request)))
        result = await tool.run_async(arguments, config=RunnableConfig(callbacks=[recorder]))
    else:
        request = mocker.patch("requests.request", return_value=response)
        result = tool.run_sync(arguments, config=RunnableConfig(callbacks=[recorder]))
    assert result.status.value == "success"
    assert "unit_price_usd" not in request.call_args.kwargs["json"]
    (receipt,) = recorder.receipts
    assert (Decimal(receipt["cost_usd"]) if "cost_usd" in receipt else None) == (
        Decimal(expected) if expected is not None else None
    )


@pytest.mark.parametrize("tool_type,connection_type", [(JinaScrapeTool, Jina), (FirecrawlTool, Firecrawl)])
@pytest.mark.parametrize("price", [True, float("nan"), float("inf"), -1, 1e65, 1e-65, "0.125"])
def test_declared_tool_price_rejects_unsafe_or_non_numeric_configuration(tool_type, connection_type, price):
    with pytest.raises(ValidationError):
        tool_type(connection=connection_type(api_key="test"), unit_price_usd=price)


@pytest.mark.parametrize("tool_type,connection_type", [(JinaScrapeTool, Jina), (FirecrawlTool, Firecrawl)])
@pytest.mark.parametrize("price", [None, 0, 0.125])
def test_declared_tool_price_survives_initialized_yaml_roundtrip(tmp_path, tool_type, connection_type, price):
    workflow = Workflow(flow=Flow(nodes=[tool_type(connection=connection_type(api_key="test"), unit_price_usd=price)]))
    first, second = tmp_path / "first.yaml", tmp_path / "second.yaml"
    workflow.to_yaml_file(first)
    loaded = Workflow.from_yaml_file(first, init_components=True)
    assert loaded.flow.nodes[0].unit_price_usd == price
    loaded.to_yaml_file(second)
    reloaded = Workflow.from_yaml_file(second, init_components=True)
    assert reloaded.flow.nodes[0].unit_price_usd == price
