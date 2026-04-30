"""Example: Cancel all HITL (Human-in-the-Loop) methods.

Tests cancellation for every HITL path:

1. ApprovalConfig (STREAM) — node with approval via streaming queue
2. ApprovalConfig (CONSOLE) — node with approval via console input()
3. HumanFeedbackTool (STREAM) — agent tool that asks user via streaming
4. HumanFeedbackTool (CONSOLE) — agent tool that asks user via console

"""

import threading
import time
from queue import Queue
from threading import Event

from dynamiq.callbacks import TracingCallbackHandler
from dynamiq.flows import Flow
from dynamiq.nodes.agents import Agent
from dynamiq.nodes.tools.human_feedback import HumanFeedbackTool
from dynamiq.nodes.tools.python import Python
from dynamiq.runnables import RunnableConfig
from dynamiq.types.feedback import ApprovalConfig, FeedbackMethod
from dynamiq.types.streaming import StreamingConfig
from dynamiq.workflow import Workflow
from examples.llm_setup import setup_llm

CANCEL_AFTER = 5.0


def _run_and_cancel(wf, input_data, label, config, cancel_after=CANCEL_AFTER):
    token = config.cancellation.token
    out = {}

    def go():
        out["r"] = wf.run_sync(input_data=input_data, config=config)

    t = threading.Thread(target=go)
    t.start()
    print(f"\n{'='*60}\n  [{label}] running... cancel in {cancel_after}s")
    time.sleep(cancel_after)
    token.cancel()
    print(f"  [{label}] canceled!")
    t.join(timeout=15)
    r = out.get("r")
    status = r.status.value if r else "NO RESULT"
    error_msg = r.error.message if r and r.error else "-"
    print(f"  [{label}] status = {status}")
    print(f"  [{label}] error  = {error_msg}")
    return r


# ── 1. HumanFeedbackTool with STREAMING ─────────────────────────────────
def test_hitl_streaming():
    """Agent uses HumanFeedbackTool(STREAM). Cancel while tool waits for user input."""
    print("\n=== 1. HumanFeedbackTool (STREAM) — Cancel while waiting for user ===")

    llm = setup_llm()
    input_queue = Queue()
    done_event = Event()

    hf_tool = HumanFeedbackTool(
        id="t-hitl-stream-tool",
        name="t-hitl-stream-ask-user",
        input_method=FeedbackMethod.STREAM,
        output_method=FeedbackMethod.STREAM,
        streaming=StreamingConfig(
            enabled=True,
            input_queue=input_queue,
            input_queue_done_event=done_event,
            timeout=60.0,
            input_queue_poll_interval=5.0,
        ),
    )

    agent = Agent(
        id="t-hitl-stream-agent",
        name="t-hitl-stream-agent",
        llm=llm,
        role=("You must ask the user for clarification using t-hitl-stream-ask-user " "before answering any question."),
        tools=[hf_tool],
        max_loops=5,
        streaming=StreamingConfig(enabled=True),
    )

    wf = Workflow(name="t-hitl-stream-wf", flow=Flow(nodes=[agent]))
    tracing = TracingCallbackHandler()
    config = RunnableConfig(callbacks=[tracing])

    r = _run_and_cancel(wf, {"input": "Help me write a blog post"}, "hitl-stream", config)
    done_event.set()
    return r


# ── 2. HumanFeedbackTool with CONSOLE ────────────────────────────────────
def test_hitl_console():
    """Agent uses HumanFeedbackTool(CONSOLE). Cancel while input() blocks in daemon thread."""
    print("\n=== 2. HumanFeedbackTool (CONSOLE) — Cancel while input() blocks ===")

    llm = setup_llm()

    hf_tool = HumanFeedbackTool(
        id="t-hitl-console-tool",
        name="t-hitl-console-ask-user",
        input_method=FeedbackMethod.CONSOLE,
        output_method=FeedbackMethod.CONSOLE,
    )

    agent = Agent(
        id="t-hitl-console-agent",
        name="t-hitl-console-agent",
        llm=llm,
        role=(
            "You must ask the user for clarification using t-hitl-console-ask-user " "before answering any question."
        ),
        tools=[hf_tool],
        max_loops=5,
    )

    wf = Workflow(name="t-hitl-console-wf", flow=Flow(nodes=[agent]))
    tracing = TracingCallbackHandler()
    config = RunnableConfig(callbacks=[tracing])

    r = _run_and_cancel(wf, {"input": "Help me write a blog post"}, "hitl-console", config)
    return r


# ── 3. Tool with ApprovalConfig (STREAM) inside Agent ────────────────────
def test_tool_approval_streaming():
    """Agent has a tool with per-call approval (STREAM). Cancel during approval wait."""
    print("\n=== 3. Tool ApprovalConfig (STREAM) — Cancel during per-tool approval ===")

    llm = setup_llm()
    input_queue = Queue()
    done_event = Event()

    email_tool = Python(
        id="t-tool-approval-email",
        name="t-tool-approval-send-email",
        description="Sends an email. Input: {'email': '<content>'}.",
        code='def run(input_data): return {"content": "Email sent."}',
        approval=ApprovalConfig(
            enabled=True,
            feedback_method=FeedbackMethod.STREAM,
            msg_template="Email: {{input_data.email}}. Approve?",
        ),
        streaming=StreamingConfig(
            enabled=True,
            input_queue=input_queue,
            input_queue_done_event=done_event,
            timeout=60.0,
        ),
    )

    agent = Agent(
        id="t-tool-approval-agent",
        name="t-tool-approval-agent",
        llm=llm,
        role="Write and send an email using t-tool-approval-send-email.",
        tools=[email_tool],
        max_loops=5,
        streaming=StreamingConfig(enabled=True),
    )

    wf = Workflow(name="t-tool-approval-wf", flow=Flow(nodes=[agent]))
    tracing = TracingCallbackHandler()
    config = RunnableConfig(callbacks=[tracing])

    r = _run_and_cancel(wf, {"input": "Send an email about the meeting"}, "tool-approval", config)
    done_event.set()
    return r


# ── 4. Tool with ApprovalConfig (CONSOLE) inside Agent ───────────────────
def test_tool_approval_console():
    """Agent has a tool with per-call approval (CONSOLE). Cancel during input() wait."""
    print("\n=== 4. Tool ApprovalConfig (CONSOLE) — Cancel during per-tool approval ===")

    llm = setup_llm()

    email_tool = Python(
        id="t-tool-approval-console-email",
        name="t-tool-approval-console-send-email",
        description="Sends an email. Input: {'email': '<content>'}.",
        code='def run(input_data): return {"content": "Email sent."}',
        approval=ApprovalConfig(
            enabled=True,
            feedback_method=FeedbackMethod.CONSOLE,
            msg_template="Email: {{input_data.email}}. Approve?",
        ),
    )

    agent = Agent(
        id="t-tool-approval-console-agent",
        name="t-tool-approval-console-agent",
        llm=llm,
        role="Write and send an email using t-tool-approval-console-send-email.",
        tools=[email_tool],
        max_loops=5,
    )

    wf = Workflow(name="t-tool-approval-console-wf", flow=Flow(nodes=[agent]))
    tracing = TracingCallbackHandler()
    config = RunnableConfig(callbacks=[tracing])

    r = _run_and_cancel(wf, {"input": "Send an email about the meeting"}, "tool-approval-console", config)
    return r


ALL_TESTS = [
    ("1", "HumanFeedbackTool (STREAM)", test_hitl_streaming),
    ("2", "HumanFeedbackTool (CONSOLE)", test_hitl_console),
    ("3", "Tool ApprovalConfig (STREAM)", test_tool_approval_streaming),
    ("4", "Tool ApprovalConfig (CONSOLE)", test_tool_approval_console),
]


def main():
    print("=" * 60)
    print("HITL CANCELLATION TEST — ALL METHODS")
    print("=" * 60)

    results = {}
    for num, desc, fn in ALL_TESTS:
        try:
            r = fn()
            status = r.status.value if r else "no-result"
            msg = r.error.message if r and r.error else ""
        except Exception as e:
            status = f"ERROR: {e}"
            msg = ""
        results[num] = (desc, status, msg)

    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    for num, (desc, status, msg) in results.items():
        icon = "✓" if status == "canceled" else "✗"
        suffix = f"  [{msg}]" if msg else ""
        print(f"  [{num}] {icon} {desc:40s} -> {status}{suffix}")

    canceled = sum(1 for _, s, _ in results.values() if s == "canceled")
    print(f"\n{canceled}/{len(results)} tests returned CANCELED")


if __name__ == "__main__":
    main()
