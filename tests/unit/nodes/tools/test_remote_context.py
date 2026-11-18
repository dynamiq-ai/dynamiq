import json
from unittest.mock import MagicMock

import pytest

from dynamiq.connections import Http, HTTPMethod
from dynamiq.flows import Flow
from dynamiq.nodes.tools import RemoteContextTool
from dynamiq.runnables import RunnableConfig, RunnableStatus
from dynamiq.workflow import Workflow


def make_tool():
    tool = RemoteContextTool(
        name="tenant_query",
        description="Query authorized resources using the requested resource kind.",
        parameters={
            "type": "object",
            "properties": {"kind": {"type": "string", "enum": ["alerts", "watchlists"]}},
            "required": ["kind"],
        },
        connection=Http(url="https://control.example/query", method=HTTPMethod.POST),
    )
    tool.init_components()
    response = MagicMock()
    response.status_code = 200
    response.iter_content.return_value = [b'{"items":[]}']
    response.__enter__.return_value = response
    tool.client = MagicMock()
    tool.client.closed = False
    tool.client.is_closed.return_value = False
    tool.client.post.return_value = response
    return tool, response


def trusted_config():
    return RunnableConfig(
        run_id="platform-run",
        trusted_context={
            "app_id": "app",
            "deployment_id": "release",
            "control_context": {"token": "private-control-token"},
        },
    )


def test_remote_tool_forwards_host_context_separately_and_validates_arguments():
    tool, _ = make_tool()
    result = tool.run(
        {"kind": "alerts", "control_context": {"token": "forged"}, "url": "https://attacker.example"},
        config=trusted_config(),
    )
    assert result.status == RunnableStatus.SUCCESS
    assert result.output == {"content": {"items": []}}
    request = tool.client.post.call_args
    assert request.args == ("https://control.example/query",)
    assert request.kwargs["json"] == {
        "run_id": "platform-run",
        "app_id": "app",
        "deployment_id": "release",
        "control_context": {"token": "private-control-token"},
        "input": {"kind": "alerts"},
    }
    assert "private-control-token" not in json.dumps(tool.to_dict(for_tracing=True))
    tool.client.post.reset_mock()
    invalid = tool.run({"kind": "unauthorized-kind"}, config=trusted_config())
    assert invalid.status == RunnableStatus.FAILURE
    tool.client.post.assert_not_called()


@pytest.mark.parametrize("failure", ["missing_context", "redirect", "transport", "oversized"])
def test_remote_tool_fails_closed_without_exposing_credentials(failure):
    tool, response = make_tool()
    config = trusted_config()
    if failure == "missing_context":
        config = RunnableConfig()
    elif failure == "redirect":
        response.status_code = 302
    elif failure == "transport":
        tool.client.post.side_effect = RuntimeError("private-control-token endpoint-body")
    else:
        response.iter_content.return_value = [b"x" * (1024 * 1024 + 1)]
    result = tool.run({"kind": "alerts"}, config=config)
    assert result.status == RunnableStatus.FAILURE
    assert "private-control-token" not in str(result.output)
    assert "endpoint-body" not in str(result.output)


def test_remote_tool_yaml_round_trip_preserves_schema_and_fixed_connection(tmp_path):
    tool, _ = make_tool()
    path = tmp_path / "workflow.yaml"
    Workflow(flow=Flow(nodes=[tool])).to_yaml_file(path)
    restored = Workflow.from_yaml_file(path, init_components=True)
    node = restored.flow.nodes[0]
    assert isinstance(node, RemoteContextTool)
    assert node.connection.url == "https://control.example/query"
    assert node.input_schema.model_validate({"kind": "watchlists"}).kind == "watchlists"
    assert "private-control-token" not in path.read_text()
