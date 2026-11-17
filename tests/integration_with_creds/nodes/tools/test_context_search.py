"""Live Context.dev searches; requires CONTEXT_DEV_API_KEY and consumes API credits."""

import os

import pytest

from dynamiq.connections import Context
from dynamiq.nodes.tools import ContextSearchTool
from dynamiq.runnables import RunnableStatus


@pytest.fixture
def context_search_tool():
    if not os.getenv("CONTEXT_DEV_API_KEY"):
        pytest.skip("CONTEXT_DEV_API_KEY is not set")
    return ContextSearchTool(connection=Context(), max_results=3, is_optimized_for_agents=True)


def assert_cited_highlights(result):
    assert result.status == RunnableStatus.SUCCESS, result.error
    assert result.output["urls"]
    assert len(result.output["urls"]) <= 2
    assert all(url.startswith("https://") for url in result.output["urls"])
    assert any(source["highlights_status"] == "SUCCESS" and source["content"] for source in result.output["sources"])
    assert result.output["content"]


def test_context_search_returns_cited_highlights(context_search_tool):
    result = context_search_tool.run({"query": "Python standard library documentation", "max_results": 2})

    assert_cited_highlights(result)


@pytest.mark.asyncio
async def test_context_async_search_returns_cited_highlights(context_search_tool):
    result = await context_search_tool.run_async({"query": "Python standard library documentation", "max_results": 2})

    assert_cited_highlights(result)
