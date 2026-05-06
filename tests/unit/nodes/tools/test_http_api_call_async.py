from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from dynamiq import connections
from dynamiq.nodes.node import Node
from dynamiq.nodes.tools import HttpApiCall, ResponseType
from dynamiq.nodes.tools.http_api_call import HttpApiCallInputSchema


def _build_node(response_type=ResponseType.JSON):
    connection = connections.Http(
        method=connections.HTTPMethod.GET,
        url="https://api.example.com/data",
        headers={"x-api-key": "k"},
    )
    return HttpApiCall(
        connection=connection,
        success_codes=[200, 201],
        timeout=5,
        response_type=response_type,
    )


def _mock_response(status_code=200, json_payload=None, text="", content=b"", headers=None):
    resp = MagicMock()
    resp.status_code = status_code
    resp.text = text
    resp.content = content
    resp.json = MagicMock(return_value=json_payload)
    resp.headers = headers or {"content-type": "application/json"}
    return resp


def test_http_api_call_has_native_async():
    assert HttpApiCall.execute_async is not Node.execute_async


@pytest.mark.asyncio
async def test_http_api_call_execute_async_json_response():
    node = _build_node(ResponseType.JSON)
    payload = {"a": "1"}
    mock_client = MagicMock()
    mock_client.request = AsyncMock(return_value=_mock_response(json_payload=payload))

    with patch.object(HttpApiCall, "get_async_client", AsyncMock(return_value=mock_client)):
        result = await node.run_async(input_data={})

    assert result.status.value == "success"
    assert result.output["content"] == payload
    assert result.output["status_code"] == 200
    mock_client.request.assert_awaited_once()
    call_kwargs = mock_client.request.await_args.kwargs
    assert call_kwargs["url"] == "https://api.example.com/data"
    assert call_kwargs["headers"]["x-api-key"] == "k"


@pytest.mark.asyncio
async def test_http_api_call_execute_async_text_response():
    node = _build_node(ResponseType.TEXT)
    mock_client = MagicMock()
    mock_client.request = AsyncMock(
        return_value=_mock_response(text="hello", headers={"content-type": "text/plain"})
    )

    with patch.object(HttpApiCall, "get_async_client", AsyncMock(return_value=mock_client)):
        result = await node.run_async(input_data={})

    assert result.output["content"] == "hello"
    assert result.output["status_code"] == 200


@pytest.mark.asyncio
async def test_http_api_call_execute_async_raw_response():
    node = _build_node(ResponseType.RAW)
    mock_client = MagicMock()
    mock_client.request = AsyncMock(
        return_value=_mock_response(content=b"raw-bytes", headers={"content-type": "application/octet-stream"})
    )

    with patch.object(HttpApiCall, "get_async_client", AsyncMock(return_value=mock_client)):
        result = await node.run_async(input_data={})

    assert result.output["content"] == b"raw-bytes"


@pytest.mark.asyncio
async def test_http_api_call_execute_async_failed_status():
    node = _build_node(ResponseType.JSON)
    mock_client = MagicMock()
    mock_client.request = AsyncMock(return_value=_mock_response(status_code=500, text="boom"))

    with patch.object(HttpApiCall, "get_async_client", AsyncMock(return_value=mock_client)):
        result = await node.run_async(input_data={})

    assert result.status.value == "failure"


@pytest.mark.asyncio
async def test_http_api_call_execute_async_request_exception():
    node = _build_node(ResponseType.JSON)
    mock_client = MagicMock()
    mock_client.request = AsyncMock(side_effect=RuntimeError("network down"))

    with patch.object(HttpApiCall, "get_async_client", AsyncMock(return_value=mock_client)):
        result = await node.run_async(input_data={})

    assert result.status.value == "failure"


@pytest.mark.asyncio
async def test_http_api_call_execute_async_input_overrides():
    node = _build_node(ResponseType.JSON)
    mock_client = MagicMock()
    mock_client.request = AsyncMock(return_value=_mock_response(json_payload={"ok": True}))

    with patch.object(HttpApiCall, "get_async_client", AsyncMock(return_value=mock_client)):
        await node.run_async(
            input_data={
                "url": "https://override.example.com/x",
                "headers": {"extra": "v"},
                "params": {"q": "1"},
                "data": {"k": "v"},
            }
        )

    call_kwargs = mock_client.request.await_args.kwargs
    assert call_kwargs["url"] == "https://override.example.com/x"
    assert call_kwargs["headers"]["extra"] == "v"
    assert call_kwargs["params"] == {"q": "1"}
    assert call_kwargs["data"] == {"k": "v"}


@pytest.mark.asyncio
async def test_http_api_call_execute_async_payload_json_mode():
    node = _build_node(ResponseType.JSON)
    mock_client = MagicMock()
    mock_client.request = AsyncMock(return_value=_mock_response(json_payload={"ok": True}))

    schema = HttpApiCallInputSchema(
        data={"k": "v"},
        payload_type="json",
        url="https://api.example.com/post",
    )

    with patch.object(HttpApiCall, "get_async_client", AsyncMock(return_value=mock_client)):
        await node.execute_async(schema)

    call_kwargs = mock_client.request.await_args.kwargs
    assert "data" not in call_kwargs
    assert call_kwargs["json"] == {"k": "v"}


@pytest.mark.asyncio
async def test_http_api_call_default_get_omits_empty_data_kwarg():
    """Default GET (RAW payload, no files, no data) must not send ``data={}``.

    Sync ``requests`` treats ``data={}`` as no body; older or future httpx versions
    could encode it as an empty form payload with a content-type header. Omitting the
    kwarg keeps both paths byte-equivalent.
    """
    node = _build_node(ResponseType.JSON)
    node.payload_type = "raw"
    mock_client = MagicMock()
    mock_client.request = AsyncMock(return_value=_mock_response(json_payload={"ok": True}))

    with patch.object(HttpApiCall, "get_async_client", AsyncMock(return_value=mock_client)):
        await node.run_async(input_data={})

    call_kwargs = mock_client.request.await_args.kwargs
    assert "data" not in call_kwargs
    assert "json" not in call_kwargs
    assert "files" not in call_kwargs


@pytest.mark.asyncio
async def test_http_api_call_raw_with_data_includes_data():
    """When RAW data is non-empty, it still travels in the request kwargs."""
    node = _build_node(ResponseType.JSON)
    node.payload_type = "raw"
    mock_client = MagicMock()
    mock_client.request = AsyncMock(return_value=_mock_response(json_payload={"ok": True}))

    with patch.object(HttpApiCall, "get_async_client", AsyncMock(return_value=mock_client)):
        await node.run_async(input_data={"data": {"k": "v"}})

    call_kwargs = mock_client.request.await_args.kwargs
    assert call_kwargs["data"] == {"k": "v"}


@pytest.mark.asyncio
async def test_http_api_call_json_payload_dropped_when_files_present():
    """``json`` and ``files`` are mutually exclusive on the wire.

    Sync ``requests`` silently overwrites a json body with multipart when ``files`` is set;
    we mirror that here so the async path doesn't ship an ambiguous request.
    """
    import io

    node = _build_node(ResponseType.JSON)
    mock_client = MagicMock()
    mock_client.request = AsyncMock(return_value=_mock_response(json_payload={"ok": True}))

    schema = HttpApiCallInputSchema(
        data={"k": "v"},
        payload_type="json",
        url="https://api.example.com/post",
        files={"upload": io.BytesIO(b"hello")},
    )

    with patch.object(HttpApiCall, "get_async_client", AsyncMock(return_value=mock_client)):
        await node.execute_async(schema)

    call_kwargs = mock_client.request.await_args.kwargs
    assert "json" not in call_kwargs
    assert "files" in call_kwargs
    assert call_kwargs["files"]["upload"] == b"hello"


@pytest.mark.asyncio
async def test_http_api_call_raw_data_kept_alongside_files():
    """RAW payload data must travel as multipart form fields when files are present."""
    import io

    node = _build_node(ResponseType.JSON)
    mock_client = MagicMock()
    mock_client.request = AsyncMock(return_value=_mock_response(json_payload={"ok": True}))

    schema = HttpApiCallInputSchema(
        data={"field1": "value1"},
        payload_type="raw",
        url="https://api.example.com/post",
        files={"upload": io.BytesIO(b"hello")},
    )

    with patch.object(HttpApiCall, "get_async_client", AsyncMock(return_value=mock_client)):
        await node.execute_async(schema)

    call_kwargs = mock_client.request.await_args.kwargs
    assert call_kwargs["data"] == {"field1": "value1"}
    assert call_kwargs["files"]["upload"] == b"hello"


@pytest.mark.asyncio
async def test_http_api_call_json_payload_omits_empty_files_kwarg():
    """When no files are uploaded, ``files`` must not appear in request kwargs.

    Regression: previously ``_build_request_kwargs`` always set ``files={}``, which left
    the async request with both ``files`` and ``json`` arguments — currently tolerated by
    httpx 0.28 but ambiguous and not guaranteed across versions.
    """
    node = _build_node(ResponseType.JSON)
    mock_client = MagicMock()
    mock_client.request = AsyncMock(return_value=_mock_response(json_payload={"ok": True}))

    schema = HttpApiCallInputSchema(
        data={"k": "v"},
        payload_type="json",
        url="https://api.example.com/post",
    )

    with patch.object(HttpApiCall, "get_async_client", AsyncMock(return_value=mock_client)):
        await node.execute_async(schema)

    call_kwargs = mock_client.request.await_args.kwargs
    assert "files" not in call_kwargs
    assert call_kwargs["json"] == {"k": "v"}


@pytest.mark.asyncio
async def test_http_api_call_real_httpx_json_payload():
    """End-to-end against a real ``httpx.AsyncClient`` to catch httpx behavior shifts.

    Uses ``httpx.MockTransport`` so no network is required. If a future httpx version
    starts rejecting ``files=`` alongside ``json=``, or the request body fails to be
    sent as JSON, this test fails before the bug reaches users. The body assertion
    parses JSON rather than byte-comparing because httpx 0.28 switched to compact
    separators (``{"k":"v"}``) where 0.27 used spaced ones (``{"k": "v"}``).
    """
    import json

    import httpx

    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body_bytes"] = request.content
        captured["content_type"] = request.headers.get("content-type", "")
        return httpx.Response(200, json={"ok": True})

    transport = httpx.MockTransport(handler)
    real_client = httpx.AsyncClient(transport=transport, follow_redirects=True)

    node = _build_node(ResponseType.JSON)
    schema = HttpApiCallInputSchema(
        data={"k": "v"},
        payload_type="json",
        url="https://api.example.com/post",
    )

    try:
        with patch.object(HttpApiCall, "get_async_client", AsyncMock(return_value=real_client)):
            result = await node.execute_async(schema)
    finally:
        await real_client.aclose()

    assert result["status_code"] == 200
    assert result["content"] == {"ok": True}
    assert captured["content_type"].startswith("application/json")
    assert json.loads(captured["body_bytes"]) == {"k": "v"}


@pytest.mark.asyncio
async def test_http_api_call_concurrency_overlap():
    """Verify gather'd run_async calls overlap rather than serialize."""
    from tests.helpers.async_node import assert_concurrent_execution

    node = _build_node(ResponseType.JSON)

    sleep_s = 0.05

    async def slow_request(**kwargs):
        import asyncio
        await asyncio.sleep(sleep_s)
        return _mock_response(json_payload={"ok": True})

    mock_client = MagicMock()
    mock_client.request = AsyncMock(side_effect=slow_request)

    with patch.object(HttpApiCall, "get_async_client", AsyncMock(return_value=mock_client)):
        results = await assert_concurrent_execution(
            node,
            payloads=[{} for _ in range(10)],
            expected_single_call_s=sleep_s,
        )

    assert len(results) == 10
    for r in results:
        assert r.status.value == "success"
        assert r.output["content"] == {"ok": True}
