import os

from dynamiq import Workflow
from dynamiq.connections.connections import MPC as MPCConnection
from dynamiq.connections.managers import get_connection_manager
from dynamiq.nodes.agents.react import ReActAgent
from dynamiq.nodes.tools.mcp_adapter import MCPServerAdapter
from dynamiq.serializers.loaders.yaml import WorkflowYAMLLoader
from examples.llm_setup import setup_llm


def setup_wf():
    llm = setup_llm()

    stdio_connection = MPCConnection(
        command="python",
        args=[os.path.join("../../tools/mcp_server_as_tool", "mcp_servers", "math_servers.py")],
    )
    mcp_tool_adapter = MCPServerAdapter(connection=stdio_connection)

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


def run_wf(wf):
    result = wf.run(
        input_data={
            "input": "multiply 12 by 3.",
        }
    )
    print(result)


def load_wf_to_yaml(wf, yaml_file_path="dag_mcp_server.yaml"):
    wf.to_yaml_file(yaml_file_path)


def load_wf_from_yaml(yaml_file_path="dag_mcp_server.yaml"):
    with get_connection_manager() as cm:
        print("YAML Loading started")
        wf_data = WorkflowYAMLLoader.load(
            file_path=yaml_file_path,
            connection_manager=cm,
            init_components=True,
        )
        wf = Workflow.from_yaml_file_data(file_data=wf_data)
        print("YAML Loading finished")
        return wf


if __name__ == "__main__":
    wf = setup_wf()
    # run_wf(wf)
    load_wf_to_yaml(wf)
