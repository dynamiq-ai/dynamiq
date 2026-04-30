import threading
import time

from dynamiq.callbacks import TracingCallbackHandler
from dynamiq.flows import Flow
from dynamiq.nodes.agents import Agent
from dynamiq.nodes.tools.python import Python
from dynamiq.runnables import RunnableConfig
from dynamiq.workflow import Workflow
from examples.llm_setup import setup_llm

AGENT_ROLE = (
    "You are a research assistant. Use the provided tools to gather information. "
    "Always use the slow-research tool for each sub-question before answering."
)

CANCEL_AFTER_SECONDS = 15.0


def build_workflow():
    llm = setup_llm()

    slow_tool = Python(
        name="slow-research",
        description="Performs deep research on a topic. Input: {'topic': '<topic>'}. Returns research results.",
        code="""
import time
def run(params: dict):
    topic = params.get("topic", "unknown")
    time.sleep(2)
    return {"content": f"Research results for: {topic}"}
""",
    )

    agent = Agent(
        name="Research Agent",
        llm=llm,
        role=AGENT_ROLE,
        tools=[slow_tool],
        max_loops=10,
    )

    return Workflow(flow=Flow(nodes=[agent])), agent


def main():
    print("=== Cancel Agent Mid-Loop Example ===\n")

    wf, agent = build_workflow()

    tracing = TracingCallbackHandler()
    config = RunnableConfig(callbacks=[tracing])
    token = config.cancellation.token

    result_holder = {}

    def run_workflow():
        result_holder["result"] = wf.run_sync(
            input_data={
                "input": "Research these topics one by one: quantum computing, "
                "neural networks, blockchain, and genetic algorithms."
            },
            config=config,
        )

    thread = threading.Thread(target=run_workflow)
    thread.start()

    print(f"Workflow running... will cancel in {CANCEL_AFTER_SECONDS}s")
    time.sleep(CANCEL_AFTER_SECONDS)
    print("Sending cancel signal!")
    token.cancel()

    thread.join(timeout=30.0)

    result = result_holder.get("result")
    if result:
        print(f"\nResult status: {result.status}")

        # Inspect tracing
        print("\nTracing runs:")
        for run_id, run in tracing.runs.items():
            status_str = run.status.value if run.status else "unknown"
            cancel_reason = run.metadata.get("cancellation_reason", "")
            suffix = f" (reason: {cancel_reason})" if cancel_reason else ""
            print(f"  {run.type.value}: {run.name} -> {status_str}{suffix}")

        print("\nAgent was successfully canceled mid-loop!")


if __name__ == "__main__":
    main()
