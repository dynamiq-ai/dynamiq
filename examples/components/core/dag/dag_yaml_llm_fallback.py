import json
import logging
import os

from dotenv import find_dotenv, load_dotenv

from dynamiq import Workflow, connections, runnables
from dynamiq.callbacks import TracingCallbackHandler
from dynamiq.connections.managers import get_connection_manager
from dynamiq.flows.flow import Flow
from dynamiq.nodes.llms import Anthropic, OpenAI
from dynamiq.nodes.llms.base import FallbackConfig, FallbackTrigger
from dynamiq.prompts import Message, Prompt
from dynamiq.serializers.loaders.yaml import WorkflowYAMLLoader
from dynamiq.utils import JsonWorkflowEncoder

logger = logging.getLogger(__name__)
load_dotenv(find_dotenv())


def run_llm_fallback_workflow(yaml_file_path: str):
    with get_connection_manager() as cm:
        wf_data = WorkflowYAMLLoader.load(
            file_path=yaml_file_path,
            connection_manager=cm,
            init_components=True,
        )

        tracing = TracingCallbackHandler()
        wf = Workflow.from_yaml_file_data(file_data=wf_data, wf_id="llm-fallback-workflow")

        result = wf.run(
            input_data={"query": "What is the capital of France? Answer in one word."},
            config=runnables.RunnableConfig(callbacks=[tracing]),
        )

        dumped_wf = json.dumps(
            {"runs": [run.to_dict() for run in tracing.runs.values()]},
            cls=JsonWorkflowEncoder,
        )
        return result, dumped_wf


def run_llm_fallback_programmatic():
    prompt = Prompt(messages=[Message(role="user", content="{{query}}")])
    fallback_llm = Anthropic(
        name="Anthropic-Fallback",
        model="claude-sonnet-4-20250514",
        connection=connections.Anthropic(),
        prompt=prompt,
    )
    primary_llm = OpenAI(
        name="OpenAI-Primary",
        model="gpt-4o",
        connection=connections.OpenAI(),
        prompt=prompt,
        fallback=FallbackConfig(
            llm=fallback_llm,
            enabled=True,
            trigger=FallbackTrigger.ANY,
        ),
    )

    wf = Workflow(flow=Flow(nodes=[primary_llm]))
    tracing = TracingCallbackHandler()

    result = wf.run(
        input_data={"query": "What is 2 + 2? Answer with just the number."},
        config=runnables.RunnableConfig(callbacks=[tracing]),
    )

    dumped_wf = json.dumps(
        {"runs": [run.to_dict() for run in tracing.runs.values()]},
        cls=JsonWorkflowEncoder,
    )
    return result, dumped_wf


if __name__ == "__main__":
    with get_connection_manager() as cm:
        dag_yaml_file_path = os.path.join(os.path.dirname(__file__), "dag_llm_fallback.yaml")
        result_yaml, dumped_wf_yaml = run_llm_fallback_workflow(dag_yaml_file_path)
        logger.info(f"YAML workflow result: {result_yaml}")

        result_programmatic, dumped_wf_programmatic = run_llm_fallback_programmatic()
        logger.info(f"Programmatic workflow result: {result_programmatic}")
