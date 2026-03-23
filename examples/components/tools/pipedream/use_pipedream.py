import logging
import os

from dynamiq import Workflow, runnables
from dynamiq.callbacks import TracingCallbackHandler
from dynamiq.connections.managers import get_connection_manager
from dynamiq.serializers.loaders.yaml import WorkflowYAMLLoader

from dynamiq.connections import PipedreamOAuth2 as PipedreamConnection
from dynamiq.nodes.tools import Pipedream
from examples.pipedream.configurable_props_templates import GOOGLE_DRIVE_EXRACT_FILES_PROPS, JIRA_CREATE_ISSUE_PROPS

JIRA_AUTH_PROVISION_ID = ""
JIRA_DYNAMIC_PROPS_ID = ""
JIRA_EXTERNAL_USER_ID = ""
JIRA_CLOUD_ID = ""

DRIVE_AUTH_PROVISION_ID = ""
DRIVE_EXTERNAL_USER_ID = ""

logger = logging.getLogger(__name__)


def jira_create_task(input_props: dict):
    pipedream = Pipedream(
        input_props=input_props,
        external_user_id=JIRA_EXTERNAL_USER_ID,
        action_id="jira-create-issue",
        dynamic_props_id=JIRA_DYNAMIC_PROPS_ID,
        configurable_props={"app": {"authProvisionId": JIRA_AUTH_PROVISION_ID}, "summary": "test"},
        connection=PipedreamConnection(),
    )
    result = pipedream.run(
        input_data={
            "cloudId": JIRA_CLOUD_ID,
            "projectId": "10000",
            "issueTypeId": "10003",
        }
    )
    print(result)


def google_drive_list_files(input_props: dict):
    pipedream = Pipedream(
        input_props=input_props,
        external_user_id=DRIVE_EXTERNAL_USER_ID,
        action_id="google_drive-list-files",
        configurable_props={"googleDrive": {"authProvisionId": DRIVE_AUTH_PROVISION_ID}},
        connection=PipedreamConnection(),
    )
    result = pipedream.run(input_data={})
    print(result.output["content"])


def dag_jira_create_task():
    yaml_file_path = os.path.join(os.path.dirname(__file__), "pipedream.yaml")
    tracing = TracingCallbackHandler()

    with get_connection_manager() as cm:
        # Load the workflow from the YAML file
        wf_data = WorkflowYAMLLoader.load(
            file_path=yaml_file_path,
            connection_manager=cm,
            init_components=True,
        )
        wf = Workflow.from_yaml_file_data(file_data=wf_data, wf_id="pipedream-tool-workflow")

        # Run the workflow with the first input
        result_1 = wf.run(
            input_data={
                "cloudId": JIRA_CLOUD_ID,
                "projectId": "10000",
                "issueTypeId": "10003",
            },
            config=runnables.RunnableConfig(callbacks=[tracing]),
        )
        logger.info(f"Result 1: {result_1.output}")

        # Run the workflow with the second input
        result_2 = wf.run(
            input_data={
                "cloudId": "a43a5370-4920-4b16-9766-d0f8284905f6",
                "projectId": "10000",
                "issueTypeId": "10003",
                "summary": "Input data test",
            },
            config=runnables.RunnableConfig(callbacks=[tracing]),
        )
        logger.info(f"Result 2: {result_2.output}")

    logger.info(f"Workflow {wf.id} finished. Results:")
    for node_id, result in wf.flow._results.items():
        logger.info(f"Node {node_id}-{wf.flow._node_by_id[node_id].name}: \n{result}")


if __name__ == "__main__":
    jira_create_task(input_props=JIRA_CREATE_ISSUE_PROPS)
    dag_jira_create_task()
    google_drive_list_files(input_props=GOOGLE_DRIVE_EXRACT_FILES_PROPS)
