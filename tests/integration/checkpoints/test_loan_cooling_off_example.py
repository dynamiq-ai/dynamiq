"""The loan cooling-off example exists twice, as Python and as YAML. Both must pause and resume the same way."""

import importlib.util
from datetime import timedelta
from pathlib import Path

import pytest

from dynamiq import Workflow
from dynamiq.checkpoints.backends.in_memory import InMemory
from dynamiq.checkpoints.config import CheckpointBehavior, CheckpointConfig
from dynamiq.runnables import RunnableConfig, RunnableStatus
from dynamiq.serializers.loaders.yaml import WorkflowYAMLLoader

EXAMPLES = Path(__file__).resolve().parents[3] / "examples" / "components" / "core" / "checkpoints"


def load_python_example():
    spec = importlib.util.spec_from_file_location("loan_cooling_off", EXAMPLES / "loan_cooling_off.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


example = load_python_example()


def run_python_twin(application: dict) -> tuple[list[str], dict]:
    backend = InMemory()
    flow = example.build_flow(backend)
    flow.run_sync(input_data=application)
    checkpoint = backend.get_latest_by_flow(flow.id)
    with example.clock_at(checkpoint.resume_at + timedelta(minutes=1)):
        resumed = flow.run_sync(input_data=None, resume_from=checkpoint.id)
    assert resumed.status == RunnableStatus.SUCCESS
    return sorted(checkpoint.completed_node_ids), resumed.output


def run_yaml_twin(application: dict) -> tuple[list[str], dict]:
    workflow = Workflow.from_yaml_file_data(
        file_data=WorkflowYAMLLoader.load(file_path=str(EXAMPLES / "loan_cooling_off.yaml"), init_components=True)
    )
    backend = InMemory()

    def config(resume_from: str | None = None) -> RunnableConfig:
        return RunnableConfig(
            checkpoint=CheckpointConfig(
                enabled=True,
                backend=backend,
                behavior=CheckpointBehavior.REPLACE,
                checkpoint_on_start_enabled=False,
                checkpoint_after_node_enabled=False,
                resume_from=resume_from,
            )
        )

    workflow.run_sync(input_data=application, config=config())
    checkpoint = backend.get_latest_by_flow(workflow.flow.id)
    with example.clock_at(checkpoint.resume_at + timedelta(minutes=1)):
        resumed = workflow.run_sync(input_data=None, config=config(resume_from=checkpoint.id))
    assert resumed.status == RunnableStatus.SUCCESS
    return sorted(checkpoint.completed_node_ids), resumed.output


def outcome(output: dict) -> dict:
    """Each step's status and output, without when the cooling-off ended: that depends on when the run started."""
    steps = {node_id: (result["status"], result["output"]) for node_id, result in output.items()}
    status, waited = steps["cooling-off"]
    steps["cooling-off"] = (status, {key: value for key, value in waited.items() if key != "waited_until"})
    return steps


@pytest.mark.parametrize(
    "application, handled_by",
    [
        ({"loan_id": "L-1", "amount": 25_000, "monthly_income": 4_200}, "disburse"),
        ({"loan_id": "L-2", "amount": 250_000, "monthly_income": 21_000}, "notify-ops"),
    ],
    ids=["paid-out", "payout-fails-to-operations"],
)
def test_yaml_and_python_examples_pause_and_resume_alike(application, handled_by):
    python_finished, python_output = run_python_twin(application)
    yaml_finished, yaml_output = run_yaml_twin(application)

    assert python_finished == yaml_finished == ["application", "credit-check"]
    assert outcome(yaml_output) == outcome(python_output)
    assert yaml_output[handled_by]["status"] == RunnableStatus.SUCCESS.value
