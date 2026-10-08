import json
from types import SimpleNamespace

import pytest

from dynamiq.cli.commands.context import DynamiqCtx
from dynamiq.cli.config import Settings


class RecordingApi:
    """Stands in for ApiClient: records every call and answers from a per-(method, path) table."""

    def __init__(self, responses: dict[tuple[str, str], dict] | None = None):
        self.calls: list[tuple[str, str, dict]] = []
        self.responses = responses or {}

    def _answer(self, method: str, path: str, **kwargs):
        self.calls.append((method, path, kwargs))
        body = self.responses.get((method, path), {"data": {}})
        return SimpleNamespace(status_code=200, text=json.dumps(body), json=lambda: body)

    def get(self, path, **kwargs):
        return self._answer("GET", path, **kwargs)

    def post(self, path, **kwargs):
        return self._answer("POST", path, **kwargs)

    def put(self, path, **kwargs):
        return self._answer("PUT", path, **kwargs)

    def patch(self, path, **kwargs):
        return self._answer("PATCH", path, **kwargs)

    def delete(self, path, **kwargs):
        return self._answer("DELETE", path, **kwargs)


@pytest.fixture
def recording_api():
    return RecordingApi()


@pytest.fixture
def cli_ctx(recording_api):
    dctx = DynamiqCtx()
    dctx.settings = Settings(project_id="proj-1")
    dctx.api = recording_api
    return dctx
