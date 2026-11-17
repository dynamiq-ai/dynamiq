import json

import pytest

from dynamiq.connections import OpenAI as OpenAIConnection
from dynamiq.nodes.agents.orchestrators.graph import START, GraphOrchestrator
from dynamiq.nodes.agents.orchestrators.graph_manager import GraphAgentManager
from dynamiq.nodes.agents.orchestrators.graph_state import GraphState
from dynamiq.nodes.llms import OpenAI
from dynamiq.runnables import RunnableConfig, RunnableResult, RunnableStatus


@pytest.mark.parametrize("state_id", ["parse_json", "parse_```json"])
@pytest.mark.parametrize("fence", [None, "", "json", "JSON"])
def test_manager_selects_state_without_changing_id(mocker, state_id, fence):
    target = GraphState(id=state_id)
    manager = GraphAgentManager(
        llm=OpenAI(
            model="gpt-4o-mini",
            connection=OpenAIConnection(api_key="test-key"),
            is_postponed_component_init=True,
        )
    )
    orchestrator = GraphOrchestrator(manager=manager, states=[target])
    response = json.dumps({"state": state_id})
    if fence is not None:
        response = f"```{fence}\n{response}\n```"
    mocker.patch.object(
        GraphAgentManager,
        "run",
        return_value=RunnableResult(status=RunnableStatus.SUCCESS, output={"content": {"result": response}}),
    )

    assert orchestrator.get_next_state_by_manager(orchestrator._state_by_id[START], RunnableConfig()) is target
