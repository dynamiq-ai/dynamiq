"""Integration tests for BedrockAgentCoreWebSearchTool (live AWS AgentCore Gateway).

Requires AWS credentials allowed to call bedrock-agentcore:InvokeGateway, and AGENTCORE_WEB_SEARCH_GATEWAY_URL set to
the MCP endpoint of a gateway with a Web Search target on connector version 1.2.0 or later.
"""

import os

import pytest

from dynamiq.connections import AWS as AWSConnection
from dynamiq.nodes.tools import BedrockAgentCoreWebSearchTool
from dynamiq.runnables import RunnableStatus

GATEWAY_URL_ENV = "AGENTCORE_WEB_SEARCH_GATEWAY_URL"


@pytest.fixture
def web_search_tool():
    gateway_url = os.getenv(GATEWAY_URL_ENV)
    if not gateway_url:
        pytest.skip(f"{GATEWAY_URL_ENV} is not set")
    return BedrockAgentCoreWebSearchTool(
        connection=AWSConnection(), gateway_url=gateway_url, is_optimized_for_agents=True
    )


def test_search_returns_cited_results(web_search_tool):
    result = web_search_tool.run({"query": "Amazon Bedrock AgentCore announcements", "max_results": 3})

    assert result.status == RunnableStatus.SUCCESS, result.error
    assert result.output["urls"]
    assert "- URL: [https://" in result.output["content"]


@pytest.mark.asyncio
async def test_async_search_keeps_to_included_domains(web_search_tool):
    result = await web_search_tool.run_async(
        {"query": "Bedrock AgentCore web search", "include_domains": ["aws.amazon.com"], "max_results": 3}
    )

    assert result.status == RunnableStatus.SUCCESS, result.error
    assert result.output["urls"]
    assert all("aws.amazon.com" in url for url in result.output["urls"])
