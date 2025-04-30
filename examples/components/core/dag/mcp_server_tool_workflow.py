import os

from dynamiq import Workflow
from dynamiq.connections.connections import MPC as MPCConnection
from dynamiq.connections.managers import get_connection_manager
from dynamiq.nodes.agents.react import ReActAgent
from dynamiq.nodes.tools.mcp_adapter import MCPServerAdapter
from dynamiq.serializers.loaders.yaml import WorkflowYAMLLoader
from examples.llm_setup import setup_llm


def setup_agent_with_server():
    """
    Sets up a ReAct agent using an MCPServerAdapter.

    Returns:
        Workflow: A workflow object containing the configured agent.
    """
    llm = setup_llm()

    stdio_connection = MPCConnection(
        command="python",
        args=[os.path.join("../../tools/mcp_server_as_tool", "mcp_servers", "math_servers.py")],
    )
    mcp_tool_adapter = MCPServerAdapter(connection=stdio_connection, selection_mode="SELECT", tool_filter_names=["add"])

    agent = ReActAgent(
        name="react-agent",
        id="react-agent",
        llm=llm,
        tools=[mcp_tool_adapter],
        max_loops=5,
    )

    wf = Workflow()
    wf.flow.add_nodes(agent)
    return wf


def setup_agent_with_tool():
    """
    Sets up a ReAct agent using an MCPTool.

    Returns:
        Workflow: A workflow object containing the configured agent.
    """
    llm = setup_llm()

    stdio_connection = MPCConnection(
        command="python",
        args=[os.path.join("../../tools/mcp_server_as_tool", "mcp_servers", "math_servers.py")],
    )
    mcp_tool_adapter = MCPServerAdapter(
        connection=stdio_connection,
    )

    agent = ReActAgent(
        name="react-agent",
        id="react-agent",
        llm=llm,
        tools=mcp_tool_adapter.get_mcp_tools(),
        max_loops=5,
    )

    wf = Workflow()
    wf.flow.add_nodes(agent)
    return wf


def run_wf(wf):
    """
    Runs a given workflow with a input query and prints the result.

    Args:
        wf (Workflow): The workflow to execute.
    """
    result = wf.run(
        input_data={
            "input": "add 12 by 3.",
        }
    )
    print(result)


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
    wf = setup_agent_with_server()
    load_wf_to_yaml(wf)
    wf = load_wf_from_yaml()

    wf = setup_agent_with_tool()
    load_wf_to_yaml(wf, yaml_file_path="dag_mcp_tool.yaml")
    wf = load_wf_from_yaml(yaml_file_path="dag_mcp_tool.yaml")

    run_wf(wf)
