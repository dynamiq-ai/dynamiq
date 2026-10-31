"""Request-local forwarding of MiniMax's native Anthropic-compatible controls.

LiteLLM's Anthropic adapter validates controls against Claude model metadata.
Keep its message/response/stream conversion, but forward MiniMax controls at the
HTTP boundary without registering global aliases or changing shared clients.
"""

import json
from copy import deepcopy

from litellm.llms.custom_httpx.http_handler import (
    AsyncHTTPHandler,
    HTTPHandler,
    _get_httpx_client,
    get_async_httpx_client,
)


class _MiniMaxControls:
    def __init__(self, delegate, controls):
        self._delegate = delegate
        self._controls = deepcopy(controls)

    def _request_body(self, data):
        body = json.loads(data) if isinstance(data, (str, bytes)) else dict(data)
        body.update(deepcopy(self._controls))
        return json.dumps(body)

    def __del__(self):
        # The delegated pool belongs to LiteLLM; this request wrapper owns no sockets.
        pass


class _MiniMaxHTTPHandler(_MiniMaxControls, HTTPHandler):
    def post(self, url, data=None, **kwargs):
        return self._delegate.post(url, data=self._request_body(kwargs.pop("json", data)), **kwargs)


class _MiniMaxAsyncHTTPHandler(_MiniMaxControls, AsyncHTTPHandler):
    async def post(self, url, data=None, **kwargs):
        return await self._delegate.post(url, data=self._request_body(kwargs.pop("json", data)), **kwargs)


def minimax_anthropic_client(controls: dict, *, asynchronous: bool):
    if asynchronous:
        return _MiniMaxAsyncHTTPHandler(get_async_httpx_client(llm_provider="anthropic"), controls)
    return _MiniMaxHTTPHandler(_get_httpx_client(), controls)
