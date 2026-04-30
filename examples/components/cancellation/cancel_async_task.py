"""Example: Cancel a workflow using asyncio task.cancel() and token.cancel().

Tests both async cancellation methods side-by-side:

  Method 1: task.cancel()   — native asyncio CancelledError, caught by framework
  Method 2: token.cancel()  — cooperative CancellationToken signal

Both produce RunnableResult(status=CANCELED) with proper tracing.
"""

import asyncio

from dynamiq.callbacks import TracingCallbackHandler
from dynamiq.flows import Flow
from dynamiq.nodes.agents import Agent
from dynamiq.nodes.node import NodeDependency
from dynamiq.nodes.tools.python import Python
from dynamiq.runnables import RunnableConfig
from dynamiq.workflow import Workflow
from examples.llm_setup import setup_llm

SLOW_10S = """
import time
def run(input_data):
    topic = input_data.get("topic", input_data.get("input", "?"))
    elapsed = 0.0
    while elapsed < 10.0:
        time.sleep(0.05)
        elapsed += 0.05
    return {"content": f"Done: {topic}"}
"""


def _llm(prefix):
    llm = setup_llm()
    llm.name = f"{prefix}-llm"
    return llm


def build_workflow(prefix):
    agent1 = Agent(
        id=f"{prefix}-summarizer",
        name=f"{prefix}-summarizer-agent",
        llm=_llm(f"{prefix}-summarizer"),
        role="Summarize the input in one sentence. Do not use any tools.",
        max_loops=2,
    )
    agent2 = Agent(
        id=f"{prefix}-researcher",
        name=f"{prefix}-researcher-agent",
        llm=_llm(f"{prefix}-researcher"),
        role=f"Use {prefix}-slow-research for every sub-topic. Research 3+ aspects.",
        tools=[
            Python(
                id=f"{prefix}-tool",
                name=f"{prefix}-slow-research",
                description='Research tool. Input: {{"topic": "<t>"}}.',
                code=SLOW_10S,
            )
        ],
        max_loops=10,
        depends=[NodeDependency(node=agent1)],
    )
    return Workflow(
        name=f"{prefix}-workflow",
        flow=Flow(nodes=[agent1, agent2]),
    )


async def test_task_cancel():
    """Method 1: asyncio task.cancel() — framework catches CancelledError."""
    print("\n" + "=" * 60)
    print("  METHOD 1: task.cancel()")
    print("=" * 60)

    wf = build_workflow("async-task-cancel")
    tracing = TracingCallbackHandler()
    config = RunnableConfig(callbacks=[tracing])

    task = asyncio.create_task(
        wf.run_async(
            input_data={"input": "Explain quantum computing applications"},
            config=config,
        )
    )

    await asyncio.sleep(5)
    print("  Calling task.cancel()...")
    task.cancel()

    try:
        result = await task
    except asyncio.CancelledError:
        print("  WARNING: CancelledError leaked (should be caught by framework)")
        result = None

    status = result.status.value if result else "no-result"
    error_msg = result.error.message if result and result.error else "-"
    print(f"  Status: {status}")
    print(f"  Error:  {error_msg}")
    print("  Tracing:")
    for run in tracing.runs.values():
        s = run.status.value if run.status else "?"
        err = ""
        if run.error and isinstance(run.error, dict):
            err = f"  error: {run.error.get('message', '')}"
        print(f"    [{run.type.value:8s}] {run.name:40s} {s}{err}")
    return result


async def test_token_cancel():
    """Method 2: token.cancel() — cooperative CancellationToken."""
    print("\n" + "=" * 60)
    print("  METHOD 2: token.cancel()")
    print("=" * 60)

    wf = build_workflow("async-token-cancel")
    tracing = TracingCallbackHandler()
    config = RunnableConfig(callbacks=[tracing])
    token = config.cancellation.token

    task = asyncio.create_task(
        wf.run_async(
            input_data={"input": "Explain quantum computing applications"},
            config=config,
        )
    )

    await asyncio.sleep(5)
    print("  Calling token.cancel()...")
    token.cancel()

    result = await task

    status = result.status.value if result else "no-result"
    error_msg = result.error.message if result and result.error else "-"
    print(f"  Status: {status}")
    print(f"  Error:  {error_msg}")
    print("  Tracing:")
    for run in tracing.runs.values():
        s = run.status.value if run.status else "?"
        err = ""
        if run.error and isinstance(run.error, dict):
            err = f"  error: {run.error.get('message', '')}"
        print(f"    [{run.type.value:8s}] {run.name:40s} {s}{err}")
    return result


async def main():
    r1 = await test_task_cancel()
    r2 = await test_token_cancel()

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    s1 = r1.status.value if r1 else "no-result"
    s2 = r2.status.value if r2 else "no-result"
    print(f"  task.cancel()  -> {s1}")
    print(f"  token.cancel() -> {s2}")

    ok = s1 == "canceled" and s2 == "canceled"
    print(f"\n  {'Both methods work!' if ok else 'Some tests failed.'}")


if __name__ == "__main__":
    asyncio.run(main())
