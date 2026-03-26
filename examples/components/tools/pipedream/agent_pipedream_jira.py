import os

from dynamiq import Workflow, runnables
from dynamiq.callbacks import TracingCallbackHandler
from dynamiq.connections.managers import get_connection_manager
from dynamiq.nodes.agents import Agent
from dynamiq.nodes.types import InferenceMode
from dynamiq.serializers.loaders.yaml import WorkflowYAMLLoader
from dynamiq.utils.logger import logger

from dynamiq.connections import PipedreamOAuth2 as PipedreamConnection
from dynamiq.nodes.tools import Pipedream
from examples.components.tools.extra_utils import setup_llm
from examples.components.tools.pipedream.configurable_props_templates import JIRA_CREATE_ISSUE_PROPS

AGENT_ROLE = """
Project Manager with experience in agile development teams.
Your main responsibility is to analyze high-level task descriptions (prompts)
and convert them into well-defined, structured, and actionable Jira issues for task creation.

You must:
- ALWAYS FORMAT YOUR RESPONSE IN MARKDOWN.
- Create a clear and concise summary.
- Break down the task into smaller subtasks if applicable.
- Add labels or components if relevant.
- Ensure the ticket is ready for technical team consumption.

Use professional, neutral language. Avoid vague descriptions.
"""

PROMPT = """
We want to improve the onboarding experience for new users.
The goal is to introduce an interactive walkthrough that highlights key features of the platform.
This should adapt based on the user’s role (e.g., admin vs. regular user) and be dismissible at any point.
"""
EXTERNAL_USER_ID = ""
DYNAMIC_PROPS_ID = ""
AUTH_PROVISION_ID = ""
CLOUD_ID = ""


def pipedream_configuration(json_input_schema: dict):
    pipedream = Pipedream(
        input_props=json_input_schema,
        external_user_id=EXTERNAL_USER_ID,
        action_id="jira-create-issue",
        dynamic_props_id=DYNAMIC_PROPS_ID,
        configurable_props={
            "app": {"authProvisionId": AUTH_PROVISION_ID},
            "summary": "test",
            "cloudId": CLOUD_ID,
            "projectId": "10000",
            "issueTypeId": "10003",
        },
        connection=PipedreamConnection(),
        is_optimized_for_agents=True,
    )
    return pipedream


def agent_jira_create_task():
    pipedream_tool = pipedream_configuration(json_input_schema=JIRA_CREATE_ISSUE_PROPS)
    llm = setup_llm(model_provider="gpt", model_name="o3-mini", max_tokens=100000)

    agent_software = Agent(
        name="Agent",
        llm=llm,
        tools=[pipedream_tool],
        role=AGENT_ROLE,
        max_loops=10,
        inference_mode=InferenceMode.XML,
    )

    try:
        result = agent_software.run(
            input_data={"input": PROMPT},
        )
        print(result.output.get("content"))
        return result.output.get("content"), result.output.get("intermediate_steps", {})
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        return "", {}


def dag_agent_jira_create_task():
    yaml_file_path = os.path.join(os.path.dirname(__file__), "agent_pipedream.yaml")
    tracing = TracingCallbackHandler()

    with get_connection_manager() as cm:
        # Load the workflow from the YAML file
        wf_data = WorkflowYAMLLoader.load(
            file_path=yaml_file_path,
            connection_manager=cm,
            init_components=True,
        )
        wf = Workflow.from_yaml_file_data(file_data=wf_data, wf_id="pipedream-agent-workflow")

        # Run the workflow with the first input
        result_1 = wf.run(
            input_data={"input": PROMPT},
            config=runnables.RunnableConfig(callbacks=[tracing]),
        )
        logger.info(f"Result: {result_1.output}")

    logger.info(f"Workflow {wf.id} finished. Results:")
    for node_id, result in wf.flow._results.items():
        logger.info(f"Node {node_id}-{wf.flow._node_by_id[node_id].name}: \n{result}")


if __name__ == "__main__":
    agent_jira_create_task()
    dag_agent_jira_create_task()
