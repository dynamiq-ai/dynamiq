import json
import threading
import time

from dynamiq import Workflow
from dynamiq.callbacks import TracingCallbackHandler
from dynamiq.callbacks.tracing import RunStatus
from dynamiq.flows import Flow
from dynamiq.nodes.agents import Agent
from dynamiq.nodes.node import InputTransformer, NodeDependency
from dynamiq.nodes.tools.python import Python
from dynamiq.nodes.utils.utils import Input, Output
from dynamiq.runnables import RunnableConfig
from dynamiq.utils import JsonWorkflowEncoder
from examples.llm_setup import setup_llm

CANCEL_AFTER_SECONDS = 15.0

INPUT_ID = "entry"
SUMMARIZER_ID = "summarizer"
RESEARCHER_ID = "researcher"
OUTPUT_ID = "exit"


def build_workflow():
    llm = setup_llm()

    input_node = Input(id=INPUT_ID)

    summarizer = Agent(
        id=SUMMARIZER_ID,
        name="Quick Summarizer",
        llm=llm,
        role="Summarize the input in one sentence. Respond immediately without tools.",
        max_loops=2,
        depends=[NodeDependency(node=input_node)],
        input_transformer=InputTransformer(
            selector={"input": f"$.{INPUT_ID}.output.input"},
        ),
    )

    slow_tool = Python(
        name="deep-analysis",
        description="Performs deep analysis on a topic. Input: {'topic': '<topic>'}.",
        code="""
import time
def run(params: dict):
    topic = params.get("topic", "unknown")
    time.sleep(3)
    return {"content": f"Deep analysis of: {topic}"}
""",
    )

    researcher = Agent(
        id=RESEARCHER_ID,
        name="Deep Researcher",
        llm=llm,
        role=(
            "You are a thorough researcher. You receive a summary as input. "
            "Use the deep-analysis tool for each aspect mentioned in the summary. "
            "Analyze at least 3 different aspects before answering."
        ),
        tools=[slow_tool],
        max_loops=10,
        depends=[NodeDependency(node=summarizer)],
        input_transformer=InputTransformer(
            selector={"input": f"$.{SUMMARIZER_ID}.output.content"},
        ),
    )

    output_node = Output(
        id=OUTPUT_ID,
        depends=[NodeDependency(node=researcher)],
    )

    flow = Flow(nodes=[input_node, summarizer, researcher, output_node])
    return Workflow(flow=flow)


def print_trace_summary(tracing: TracingCallbackHandler):
    """Print a readable summary of all trace runs."""
    print("\n" + "=" * 60)
    print("TRACE SUMMARY")
    print("=" * 60)

    for run_id, run in tracing.runs.items():
        status = run.status.value if run.status else "unknown"
        run_type = run.type.value if run.type else "?"
        print(f"  [{run_type:8s}] {run.name or '?':20s} -> {status}")

    status_counts: dict[str, int] = {}
    for run in tracing.runs.values():
        s = run.status.value if run.status else "unknown"
        status_counts[s] = status_counts.get(s, 0) + 1
    print(f"\nStatus counts: {status_counts}")

    try:
        json.dumps(
            {"runs": [run.to_dict() for run in tracing.runs.values()]},
            cls=JsonWorkflowEncoder,
        )
        print("Trace is JSON-serializable: YES")
    except Exception as e:
        print(f"Trace JSON serialization error: {e}")


def main():
    print("=== Cancellation with Tracing Example ===\n")
    print("Pipeline: Input -> Summarizer Agent -> Researcher Agent -> Output\n")

    wf = build_workflow()

    tracing = TracingCallbackHandler()
    config = RunnableConfig(callbacks=[tracing])
    token = config.cancellation.token

    result_holder = {}

    def run_wf():
        result_holder["result"] = wf.run_sync(
            input_data={"input": "Explain the impact of quantum computing on cryptography"},
            config=config,
        )

    thread = threading.Thread(target=run_wf)
    thread.start()

    print(f"Workflow running... will cancel in {CANCEL_AFTER_SECONDS}s")
    time.sleep(CANCEL_AFTER_SECONDS)
    print("Sending cancel signal!")
    token.cancel()

    thread.join(timeout=30.0)

    result = result_holder.get("result")
    if result:
        print(f"\nWorkflow status: {result.status}")
        print_trace_summary(tracing)

        canceled_runs = [r for r in tracing.runs.values() if r.status == RunStatus.CANCELED]
        succeeded_runs = [r for r in tracing.runs.values() if r.status == RunStatus.SUCCEEDED]
        print(f"\nCanceled runs: {len(canceled_runs)}")
        print(f"Succeeded runs: {len(succeeded_runs)}")


if __name__ == "__main__":
    main()
