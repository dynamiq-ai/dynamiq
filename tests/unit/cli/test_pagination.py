"""Termination of the list-endpoint page walk."""

from types import SimpleNamespace

import click
import pytest

from dynamiq.cli.commands.workflow import MAX_PAGES, iter_pages


class CountingApi:
    """Answers every page the same way, like an endpoint that ignores `page`."""

    def __init__(self, body: dict, status_code: int = 200):
        self.body = body
        self.status_code = status_code
        self.calls: list[dict] = []

    def get(self, path, params=None, **kwargs):
        self.calls.append(dict(params or {}))
        return SimpleNamespace(
            status_code=self.status_code,
            text="",
            json=lambda: self.body,
            content=b"",
        )


def test_walk_stops_when_the_endpoint_sends_no_pagination():
    """The /v1/agent-builder/nodes shape: full page, no pagination block, `page` ignored."""
    api = CountingApi({"data": [{"type": "dynamiq.nodes.agents.Agent"}]})

    pages = list(iter_pages(api, "/v1/agent-builder/nodes"))

    assert len(pages) == 1
    assert len(api.calls) == 1


def test_walk_follows_page_count_when_the_endpoint_reports_it():
    bodies = [
        {"data": [1, 2], "pagination": {"page_count": 2, "total_count": 4}},
        {"data": [3, 4], "pagination": {"page_count": 2, "total_count": 4}},
    ]

    class Paged(CountingApi):
        def get(self, path, params=None, **kwargs):
            self.calls.append(dict(params or {}))
            return SimpleNamespace(status_code=200, text="", content=b"", json=lambda: bodies[len(self.calls) - 1])

    api = Paged({})
    pages = list(iter_pages(api, "/v1/workflows"))

    assert [len(p) for p in pages] == [2, 2]
    assert [c["page"] for c in api.calls] == [1, 2]


def test_walk_refuses_to_page_forever_when_metadata_never_ends():
    """page_count missing but a pagination block present: cap it, loudly."""
    api = CountingApi({"data": [1], "pagination": {"total_count": None}})

    with pytest.raises(click.ClickException) as excinfo:
        list(iter_pages(api, "/v1/something"))

    assert "still paging" in str(excinfo.value)
    assert len(api.calls) == MAX_PAGES
