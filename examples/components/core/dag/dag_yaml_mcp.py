import json
import os

from dynamiq import Workflow, runnables
from dynamiq.callbacks import TracingCallbackHandler
from dynamiq.connections import MCPStdio
from dynamiq.connections import OpenAI as OpenAIConnection
from dynamiq.connections.managers import get_connection_manager
from dynamiq.flows import Flow
from dynamiq.nodes.agents.react import ReActAgent
from dynamiq.nodes.llms.openai import OpenAI
from dynamiq.nodes.tools.mcp_adapter import MCPServerAdapter
from dynamiq.serializers.loaders.yaml import WorkflowYAMLLoader
from dynamiq.utils import JsonWorkflowEncoder


def setup_agent_with_server(path_to_server: str):
    """
    Sets up a ReAct agent using an MCPServerAdapter.

    Args:
        path_to_server (str): Path to the MCP server.

    Returns:
        Workflow: A workflow object containing the configured agent.
    """
    llm = OpenAI(
        name="OpenAI LLM",
        id="openai-llm",
        connection=OpenAIConnection(id="openai-conn"),
        model="gpt-4o-mini",
        temperature=0.3,
        max_tokens=1000,
    )

    stdio_connection = MCPStdio(
        id="mcp-conn",
        command="python",
        args=[path_to_server],
    )

    mcp_tool_adapter = MCPServerAdapter(
        id="mcp-adapter",
        name="mcp-adapter",
        connection=stdio_connection,
    )

    agent = ReActAgent(
        name="react-agent",
        id="react-agent",
        llm=llm,
        tools=[mcp_tool_adapter],
        max_loops=5,
    )

    wf = Workflow(id="workflow-id", name="workflow", flow=Flow(id="flow-id", name="flow", nodes=[agent]))
    return wf


def setup_agent_with_tool(path_to_server: str):
    """
    Sets up a ReAct agent using an MCPTool.

    Args:
        path_to_server (str): Path to the MCP server.

    Returns:
        Workflow: A workflow object containing the configured agent.
    """
    llm = OpenAI(
        name="OpenAI LLM",
        id="openai-llm",
        connection=OpenAIConnection(id="openai-conn"),
        model="gpt-4o-mini",
        temperature=0.3,
        max_tokens=1000,
    )

    stdio_connection = MCPStdio(
        id="mcp-conn",
        command="python",
        args=[path_to_server],
    )

    mcp_tool_adapter = MCPServerAdapter(
        id="mcp-adapter",
        name="mcp-adapter",
        connection=stdio_connection,
    )

    tools = mcp_tool_adapter.get_mcp_tools()

    agent = ReActAgent(
        name="react-agent",
        id="react-agent",
        llm=llm,
        tools=tools,
        max_loops=5,
    )

    wf = Workflow(id="workflow-id", name="workflow", flow=Flow(id="flow-id", name="flow", nodes=[agent]))
    return wf


def run_wf(wf):
    """
    Runs a given workflow with a input query and prints the result.

    Args:
        wf (Workflow): The workflow to execute.
    """
    tracing_retrieval_wf = TracingCallbackHandler()
    result = wf.run(
        input_data={
            "input": "add 1 to 3 and mutiply by 2 using tools.",
        },
        config=runnables.RunnableConfig(callbacks=[tracing_retrieval_wf]),
    )
    dumped_traces_wf = json.dumps(
        {"runs": [run.to_dict() for run in tracing_retrieval_wf.runs.values()]},
        cls=JsonWorkflowEncoder,
    )
    return result, dumped_traces_wf


def load_wf_to_yaml(wf, yaml_file_path="dag_mcp_server.yaml"):
    """
    Serializes a workflow into a YAML file.

    Args:
        wf (Workflow): The workflow to save.
        yaml_file_path (str): Path to save the YAML representation of the workflow.
    """
    wf.to_yaml_file(yaml_file_path)


def load_wf_from_yaml(yaml_file_path="dag_mcp_server.yaml"):
    """
    Deserializes a workflow from a YAML file.

    Args:
        yaml_file_path (str): Path to the YAML representation of the workflow.
    """
    with get_connection_manager() as cm:
        wf_data = WorkflowYAMLLoader.load(
            file_path=yaml_file_path,
            connection_manager=cm,
            init_components=True,
        )
        wf = Workflow.from_yaml_file_data(file_data=wf_data)
        return wf


if __name__ == "__main__":
    path_to_server = os.path.join("..", "..", "tools", "mcp_server_as_tool", "mcp_servers", "math_server.py")

    wf = setup_agent_with_server(path_to_server)
    load_wf_to_yaml(wf, yaml_file_path="dag_mcp_server.yaml")
    wf = load_wf_from_yaml(yaml_file_path="dag_mcp_server.yaml")

    result, traces = run_wf(wf)
    print(result)
    print(traces)

    wf = setup_agent_with_tool(path_to_server)
    load_wf_to_yaml(wf, yaml_file_path="dag_mcp_tool.yaml")
    wf = load_wf_from_yaml(yaml_file_path="dag_mcp_tool.yaml")

    result, traces = run_wf(wf)
    print(result)
    print(traces)
