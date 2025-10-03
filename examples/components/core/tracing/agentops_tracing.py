from typing import Any

from agentops import Client, ErrorEvent, LLMEvent, ToolEvent
from agentops.helpers import get_ISO_time

from dynamiq import Workflow
from dynamiq.callbacks import BaseCallbackHandler
from dynamiq.callbacks.base import get_parent_run_id, get_run_id
from dynamiq.callbacks.tracing import RunType
from dynamiq.connections import Exa
from dynamiq.flows import Flow
from dynamiq.nodes import NodeGroup
from dynamiq.nodes.agents import Agent
from dynamiq.nodes.tools.exa_search import ExaTool
from dynamiq.nodes.types import InferenceMode
from dynamiq.runnables import RunnableConfig
from dynamiq.utils import format_value, generate_uuid
from dynamiq.utils.env import get_env_var
from examples.llm_setup import setup_llm


class AgentOpsCallbackHandler(BaseCallbackHandler):
    """AgentOps callback handler"""

    def __init__(
        self,
        api_key: str | None = None,
        endpoint: str | None = None,
        max_wait_time: int | None = None,
        max_queue_size: int | None = None,
        tags: list[str] | None = None,
        auto_start_session: bool | None = True,
        trace_id: str | None = None,
    ):
        client_params = {
            "api_key": api_key,
            "endpoint": endpoint,
            "max_wait_time": max_wait_time,
            "max_queue_size": max_queue_size,
            "default_tags": tags,
            "auto_start_session": auto_start_session,
            "instrument_llm_calls": False,
        }
        self.trace_id = trace_id or generate_uuid()
        self.ao_client = Client()
        self.ao_client.configure(
            **{k: v for k, v in client_params.items() if v is not None},
        )
        self.ao_client.initialize()
        self.events = {}

    def on_node_start(self, serialized: dict[str, Any], input_data: dict[str, Any], **kwargs: Any):
        run_id = get_run_id(kwargs)
        node_group = serialized.get("group")
        parent_run_id = get_parent_run_id(kwargs)
        params = dict(
            id=run_id,
            name=serialized["name"],
            type=RunType.NODE,
            trace_id=self.trace_id,
            parent_run_id=parent_run_id,
            input=format_value(input_data),
            metadata={"node": serialized, "run_depends": kwargs.get("run_depends", [])},
        )
        if node_group == NodeGroup.LLMS:
            prompt = serialized["prompt"] or kwargs.get("prompt")
            prompt_messages = [m.model_dump() for m in prompt.messages] if prompt else []
            self.events[str(run_id)] = LLMEvent(model=serialized["model"], prompt=prompt_messages, params=params)
        else:
            self.events[str(run_id)] = ToolEvent(name=serialized["name"], logs=params)

    def on_node_error(self, serialized: dict[str, Any], error: BaseException, **kwargs: Any):
        run_id = get_run_id(kwargs)
        event = self.events[str(run_id)]
        error_event = ErrorEvent(trigger_event=event, exception=error)
        self.ao_client.record(error_event)

    def on_node_end(self, serialized: dict[str, Any], output_data: dict[str, Any], **kwargs: Any):
        run_id = get_run_id(kwargs)
        node_group = serialized.get("group")
        event = self.events[str(run_id)]
        event.end_timestamp = get_ISO_time()
        event.returns = format_value(output_data)
        if node_group == NodeGroup.LLMS:
            event.completion = format_value(output_data)
        self.ao_client.record(event)

    def on_workflow_end(self, serialized: dict[str, Any], output_data: dict[str, Any], **kwargs: Any):
        self.ao_client.end_session("Success")

    def on_workflow_error(self, serialized: dict[str, Any], error: BaseException, **kwargs: Any):
        self.ao_client.end_session("Fail")

    @property
    def current_session_ids(self):
        return self.ao_client.current_session_ids


def get_react_agent():
    llm = setup_llm()
    connection_exa = Exa()
    tool_search = ExaTool(connection=connection_exa)
    agent = Agent(
        name="Agent",
        id="Agent",
        llm=llm,
        tools=[tool_search],
        inference_mode=InferenceMode.XML,
    )
    return agent


def get_simple_agent():
    llm = setup_llm()
    agent = Agent(
        name="Agent",
        llm=llm,
        role="Agent, goal to provide information based on the user input",
    )
    return agent


if __name__ == "__main__":
    agent = get_react_agent()
    workflow = Workflow(
        flow=Flow(nodes=[agent]),
    )
    agentops_tracing = AgentOpsCallbackHandler(api_key=get_env_var("AGENTOPS_API_KEY"))
    workflow.run(
        {"input": "who won euro 2024"},
        config=RunnableConfig(callbacks=[agentops_tracing]),
    )
