"""A loan paid out after a statutory cooling-off period, with operations told when the payout fails.

    application -> credit-check -> cooling-off (Delay, 14 days) -> disburse
                                                                      |
                                                            (on failure) notify-ops

Fourteen days is far longer than a worker should hold a run, so the Delay pauses it: the checkpoint records the
finished credit check and the time to resume, and the run returns. The platform stores that time and resumes the
run when it comes; this example moves the clock forward instead. The resumed run does not repeat the credit check,
and when the payout fails for good, the error edge hands it to operations instead of failing the loan run.

Runs offline: every step is a Python node and the checkpoint backend is in memory.
"""

from datetime import datetime, timedelta, timezone
from unittest.mock import patch

from dynamiq import flows
from dynamiq.checkpoints.backends.in_memory import InMemory
from dynamiq.checkpoints.config import CheckpointConfig
from dynamiq.nodes import DependencyTrigger, ErrorHandling
from dynamiq.nodes.operators import Delay
from dynamiq.nodes.operators import delay as delay_module
from dynamiq.nodes.tools import Python
from dynamiq.nodes.utils import Input

COOLING_OFF = timedelta(days=14)
PAYOUT_LIMIT = 100_000

CREDIT_CHECK = """
def run(input_data):
    application = input_data["application"]["output"]
    affordable = application["monthly_income"] * 12 >= application["amount"] / 2
    return {"approved": affordable, "score": 712 if affordable else 540}
"""

DISBURSE = f"""
def run(input_data):
    amount = input_data["application"]["output"]["amount"]
    if amount > {PAYOUT_LIMIT}:
        raise RuntimeError(f"payout rail rejected {{amount}}: above the {PAYOUT_LIMIT} limit")
    return {{"disbursed": amount}}
"""

NOTIFY_OPS = """
def run(input_data):
    return {"ticket": "OPS-1", "reason": input_data["disburse"]["error"]["message"]}
"""


def build_flow(backend: InMemory) -> flows.Flow:
    application = Input(id="application")
    credit_check = Python(id="credit-check", name="credit check", code=CREDIT_CHECK)
    credit_check.depends_on(application)
    cooling_off = Delay(id="cooling-off", name="cooling-off period", duration_seconds=COOLING_OFF.total_seconds())
    cooling_off.depends_on(credit_check)
    disburse = Python(
        id="disburse",
        name="disburse",
        code=DISBURSE,
        error_handling=ErrorHandling(max_retries=1, retry_interval_seconds=1),
    )
    disburse.depends_on([application, cooling_off])
    notify_ops = Python(id="notify-ops", name="notify operations", code=NOTIFY_OPS)
    notify_ops.depends_on(disburse, trigger=DependencyTrigger.FAILURE)
    return flows.Flow(
        id="loan-disbursement",
        nodes=[application, credit_check, cooling_off, disburse, notify_ops],
        checkpoint=CheckpointConfig(enabled=True, backend=backend, checkpoint_after_node_enabled=False),
    )


def clock_at(moment: datetime):
    """Moves the Delay's clock, standing in for the days a real run spends paused."""

    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return moment

    return patch.object(delay_module, "datetime", Clock)


def process(application: dict) -> None:
    backend = InMemory()
    flow = build_flow(backend)
    print(f"\n=== Loan {application['loan_id']} for {application['amount']:,} ===")

    paused = flow.run_sync(input_data=application)
    checkpoint = backend.get_latest_by_flow(flow.id)
    print(f"run 1: {paused.error.message}")
    print(f"       checkpoint {checkpoint.status.value}, finished: {sorted(checkpoint.completed_node_ids)}")

    with clock_at(checkpoint.resume_at - timedelta(days=1)):
        early = flow.run_sync(input_data=None, resume_from=checkpoint.id)
    print(f"a day early: {early.error.message}")

    with clock_at(checkpoint.resume_at + timedelta(minutes=1)):
        resumed = flow.run_sync(input_data=None, resume_from=checkpoint.id)
    outcome = {node_id: result["status"] for node_id, result in resumed.output.items()}
    print(f"on time: run {resumed.status.value}, steps {outcome}")
    if outcome["notify-ops"] == "success":
        print(f"         operations ticket: {resumed.output['notify-ops']['output']['content']}")
    else:
        print(f"         paid out: {resumed.output['disburse']['output']['content']}")


def main() -> None:
    started = datetime.now(timezone.utc)
    process({"loan_id": "L-1001", "amount": 25_000, "monthly_income": 4_200})
    process({"loan_id": "L-1002", "amount": 250_000, "monthly_income": 21_000})
    print(f"\nBoth loans waited {COOLING_OFF.days} days in {datetime.now(timezone.utc) - started} of real time.")


if __name__ == "__main__":
    main()
