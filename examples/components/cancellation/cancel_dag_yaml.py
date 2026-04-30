"""Example: Cancel a YAML-defined DAG workflow.

Demonstrates:
- Loading a workflow from cancel_dag.yaml
- Running it with CancellationConfig
- Cancelling from a background thread after a delay
- Inspecting the result (CANCELED status, partial output)

The YAML defines:
  Summarizer Agent -> Deep Researcher Agent (with slow tool)

The cancel signal arrives while the Deep Researcher is running,
so the Summarizer completes but Deep Researcher is canceled.
"""

import os
import threading
import time

from dynamiq import Workflow
from dynamiq.callbacks import TracingCallbackHandler
from dynamiq.connections.managers import get_connection_manager
from dynamiq.runnables import RunnableConfig, RunnableStatus

CANCEL_AFTER_SECONDS = 15.0


def main():
    print("=== Cancel YAML DAG Workflow ===\n")

    yaml_path = os.path.join(os.path.dirname(__file__), "cancel_dag.yaml")

    with get_connection_manager() as cm:
        wf = Workflow.from_yaml_file(
            file_path=yaml_path,
            connection_manager=cm,
            init_components=True,
        )

    tracing = TracingCallbackHandler()
    config = RunnableConfig(callbacks=[tracing])
    token = config.cancellation.token

    result_holder = {}

    def run_wf():
        result_holder["result"] = wf.run_sync(
            input_data={"input": "The future of space exploration and colonization"},
            config=config,
        )

    thread = threading.Thread(target=run_wf)
    thread.start()

    print(f"Workflow running from YAML... will cancel in {CANCEL_AFTER_SECONDS}s")
    time.sleep(CANCEL_AFTER_SECONDS)
    print("Sending cancel signal!")
    token.cancel()

    thread.join(timeout=30.0)

    result = result_holder.get("result")
    if not result:
        print("No result returned")
        return

    print(f"\nWorkflow status: {result.status}")

    if result.output:
        print("\nPer-node results:")
        for node_id, node_out in result.output.items():
            node_obj = wf.flow._node_by_id.get(node_id)
            name = node_obj.name if node_obj else node_id[:12]
            status = node_out.get("status", "?")
            print(f"  {name:20s}: {status}")

    print("\nTracing runs:")
    for run in tracing.runs.values():
        status_str = run.status.value if run.status else "unknown"
        cancel_reason = run.metadata.get("cancellation_reason", "")
        suffix = f" (reason: {cancel_reason})" if cancel_reason else ""
        print(f"  [{run.type.value:8s}] {run.name or '?':20s} -> {status_str}{suffix}")

    if result.status == RunnableStatus.CANCELED:
        print("\nYAML workflow was successfully canceled!")


if __name__ == "__main__":
    main()
