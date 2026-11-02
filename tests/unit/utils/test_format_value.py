import enum

from dynamiq.nodes.agents.base import ToolParams
from dynamiq.nodes.tools.mcp import MCPTool
from dynamiq.runnables import RunnableResult, RunnableStatus
from dynamiq.utils.utils import TRACING_REDACTED_PLACEHOLDER, format_value, format_value_for_log


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


def test_format_value_for_log_redacts_mcp_http_headers():
    """The node lifecycle logs render the validated input instance through this at DEBUG."""
    schema = MCPTool.get_input_schema({"type": "object", "properties": {"query": {"type": "string"}}})
    instance = schema(query="hi", mcp_http_headers={"Authorization": "Bearer user-token"})

    logged = format_value_for_log(instance)
    assert logged["mcp_http_headers"] == TRACING_REDACTED_PLACEHOLDER
    assert logged["query"] == "hi"

    nested = format_value_for_log({"tool_params": {"github-mcp": {"mcp_http_headers": {"Authorization": "Bearer t"}}}})
    assert nested["tool_params"]["github-mcp"]["mcp_http_headers"] == TRACING_REDACTED_PLACEHOLDER


def test_format_value_returns_json_scalars_as_they_are():
    for value in ("text", 7, 2.5, True, None):
        assert format_value(value) is value
        assert format_value(value, for_tracing=True) is value
        assert format_value(value, force_format_types={str, int, float, bool}) is value


def test_format_value_keeps_a_stream_chunk_as_it_is():
    chunk = {
        "id": "chatcmpl-1",
        "created": 1,
        "choices": [
            {
                "index": 0,
                "delta": {"content": "Hel", "role": None, "tool_calls": [{"index": 0, "function": {"arguments": "{"}}]},
                "finish_reason": None,
            }
        ],
        "usage": None,
    }

    assert format_value(chunk) == chunk


def test_format_value_still_formats_scalar_subclasses():
    class Level(str, enum.Enum):
        LOW = "low"

    assert format_value(Level.LOW) == Level.LOW
    assert format_value({"level": Level.LOW, "raw": b"hi"}) == {"level": Level.LOW, "raw": "hi"}
