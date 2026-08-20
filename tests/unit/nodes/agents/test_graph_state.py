import json

import pytest

from dynamiq.nodes.agents.orchestrators.graph_state import _parse_manager_agent_input


@pytest.mark.parametrize(
    ("manager_result", "expected_input"),
    [
        (json.dumps({"input": "Analyze this json document"}), "Analyze this json document"),
        (json.dumps({"input": "Explain ```json fenced output"}), "Explain ```json fenced output"),
        (f"```\n{json.dumps({'input': 'Analyze this json document'})}\n```", "Analyze this json document"),
        (f"```json\n{json.dumps({'input': 'Analyze this json document'})}\n```", "Analyze this json document"),
        (f"```JSON\n{json.dumps({'input': 'Analyze this json document'})}\n```", "Analyze this json document"),
    ],
)
def test_parse_manager_agent_input_preserves_json_substring(manager_result: str, expected_input: str):
    assert _parse_manager_agent_input(manager_result) == expected_input
