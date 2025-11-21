from pathlib import Path

from dynamiq import Workflow, runnables
from dynamiq.callbacks import DynamiqTracingCallbackHandler, TracingCallbackHandler
from dynamiq.connections.managers import get_connection_manager

DEFAULT_COMPANY = "Acme Robotics"


def run_workflow(company: str = DEFAULT_COMPANY, trace_ui: bool = True):
    yaml_path = Path(__file__).parent / "workflow_manager.yaml"
    tracing = TracingCallbackHandler()
    callbacks = [tracing]
    if trace_ui:
        callbacks.append(DynamiqTracingCallbackHandler(access_key="",
                                                       base_url=""))

    with get_connection_manager() as cm:
        wf = Workflow.from_yaml_file(file_path=str(yaml_path), connection_manager=cm, init_components=True)
        wf.run(
            input_data={"input": company},
            config=runnables.RunnableConfig(callbacks=callbacks),
        )

    return tracing.runs


if __name__ == "__main__":
    run_workflow()
