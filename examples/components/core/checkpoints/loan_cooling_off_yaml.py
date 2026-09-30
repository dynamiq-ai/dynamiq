"""Runs loan_cooling_off.yaml the way the platform runs a deployed workflow: checkpointing comes with the run's
config, the first run pauses at the cooling-off Delay, and a later run resumes from its checkpoint.

Runs offline: every step is a Python node and the checkpoint backend is in memory. The Delay's clock is moved
forward instead of waiting fourteen days.
"""

import os
from datetime import datetime, timedelta
from unittest.mock import patch

from dynamiq import Workflow
from dynamiq.checkpoints.backends.in_memory import InMemory
from dynamiq.checkpoints.config import CheckpointBehavior, CheckpointConfig
from dynamiq.nodes.operators import delay as delay_module
from dynamiq.runnables import RunnableConfig
from dynamiq.serializers.loaders.yaml import WorkflowYAMLLoader

YAML_PATH = os.path.join(os.path.dirname(__file__), "loan_cooling_off.yaml")


def checkpointing(backend: InMemory, resume_from: str | None = None) -> RunnableConfig:
    """The checkpoint settings the platform runs with: one checkpoint per run, saved when the run pauses."""
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


def clock_at(moment: datetime):
    """Moves the Delay's clock, standing in for the days a real run spends paused."""

    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return moment

    return patch.object(delay_module, "datetime", Clock)


def process(application: dict) -> None:
    workflow = Workflow.from_yaml_file_data(
        file_data=WorkflowYAMLLoader.load(file_path=YAML_PATH, init_components=True)
    )
    backend = InMemory()
    print(f"\n=== Loan {application['loan_id']} for {application['amount']:,} ===")

    paused = workflow.run_sync(input_data=application, config=checkpointing(backend))
    checkpoint = backend.get_latest_by_flow(workflow.flow.id)
    print(f"run 1: {paused.error.message}")
    finished = sorted(checkpoint.completed_node_ids)
    print(f"       resumes at {checkpoint.resume_at:%Y-%m-%d %H:%M} UTC, finished: {finished}")

    with clock_at(checkpoint.resume_at + timedelta(minutes=1)):
        resumed = workflow.run_sync(input_data=None, config=checkpointing(backend, resume_from=checkpoint.id))
    outcome = {node_id: result["status"] for node_id, result in resumed.output.items()}
    print(f"run 2: {resumed.status.value}, steps {outcome}")


def main() -> None:
    process({"loan_id": "L-2001", "amount": 25_000, "monthly_income": 4_200})
    process({"loan_id": "L-2002", "amount": 250_000, "monthly_income": 21_000})


if __name__ == "__main__":
    main()
