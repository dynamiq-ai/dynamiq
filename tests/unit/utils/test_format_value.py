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
