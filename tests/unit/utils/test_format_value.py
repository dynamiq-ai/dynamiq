from dynamiq.nodes.agents.base import ToolParams
from dynamiq.runnables import RunnableResult, RunnableStatus
from dynamiq.utils.utils import TRACING_REDACTED_PLACEHOLDER, format_value


def test_format_value_redacts_mcp_http_headers_for_tracing():
    payload = {
        "title": "bug",
        "mcp_http_headers": {"Authorization": "Bearer user-token"},
        "tool_params": {"by_name": {"github-mcp": {"mcp_http_headers": {"Authorization": "Bearer user-token"}}}},
    }

    traced = format_value(payload, for_tracing=True)
    assert traced["mcp_http_headers"] == TRACING_REDACTED_PLACEHOLDER
    assert traced["tool_params"]["by_name"]["github-mcp"]["mcp_http_headers"] == TRACING_REDACTED_PLACEHOLDER
    assert traced["title"] == "bug"

    raw = format_value(payload, for_tracing=False)
    assert raw["mcp_http_headers"] == {"Authorization": "Bearer user-token"}


def test_format_value_redacts_mcp_http_headers_nested_in_a_model():
    """The agent's own input takes the model branch: `tool_params` is a `ToolParams` instance."""
    tool_params = ToolParams.model_validate(
        {"by_name": {"github-mcp": {"mcp_http_headers": {"Authorization": "Bearer user-token"}}}}
    )

    traced = format_value({"input": "hi", "tool_params": tool_params}, for_tracing=True)

    assert traced["tool_params"]["by_name_params"]["github-mcp"]["mcp_http_headers"] == TRACING_REDACTED_PLACEHOLDER
    assert traced["input"] == "hi"


def test_format_value_redacts_mcp_http_headers_carried_by_a_runnable_result():
    result = RunnableResult(
        status=RunnableStatus.SUCCESS,
        input={"q": "x", "mcp_http_headers": {"Authorization": "Bearer user-token"}},
        output={"content": "ok"},
    )

    traced = format_value(result, for_tracing=True)

    assert traced["input"]["mcp_http_headers"] == TRACING_REDACTED_PLACEHOLDER
    assert traced["input"]["q"] == "x"
    assert traced["output"] == {"content": "ok"}
