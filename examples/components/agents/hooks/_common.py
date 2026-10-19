import argparse
import json
import os
from contextlib import contextmanager
from typing import Any
from unittest.mock import patch

from dotenv import find_dotenv, load_dotenv

from dynamiq import Workflow
from dynamiq.callbacks import DynamiqTracingCallbackHandler, TracingCallbackHandler
from dynamiq.connections.managers import get_connection_manager
from dynamiq.nodes.agents import Agent
from dynamiq.nodes import Node
from dynamiq.nodes.llms import OpenAI
from dynamiq.runnables import RunnableConfig, RunnableResult, RunnableStatus
from dynamiq.types.feedback import ApprovalInputData

load_dotenv(find_dotenv(), override=True)


def parse_args(description: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--offline", action="store_true", help="scripted LLM, no network (default without a key)")
    parser.add_argument("--online", action="store_true", help="force the real LLM even without OPENAI_API_KEY")
    parser.add_argument("--ui", action="store_true", help="send traces to the Dynamiq UI even when offline")
    parser.add_argument("--question", help="run this single question instead of the built-in scenarios")
    return parser.parse_args()


def is_offline(args: argparse.Namespace) -> bool:
    return args.offline or (not args.online and not os.getenv("OPENAI_API_KEY"))


def send_to_ui(args: argparse.Namespace) -> bool:
    """An offline (scripted) run stays local unless ``--ui`` is given: it should not put fake traces in your project."""
    return args.ui or not is_offline(args)


def make_tracing(ui: bool) -> tuple[TracingCallbackHandler, bool]:
    """UI tracing when ``ui`` and DYNAMIQ_TRACE_ACCESS_KEY is set, in-memory tracing otherwise."""
    access_key = os.getenv("DYNAMIQ_TRACE_ACCESS_KEY")
    if access_key and ui:
        handler = DynamiqTracingCallbackHandler(
            access_key=access_key, base_url=os.getenv("DYNAMIQ_TRACE_BASE_URL", "https://collector.getdynamiq.ai")
        )
        return handler, True
    return TracingCallbackHandler(), False


def load_agent(yaml_path: str) -> tuple[Workflow, Agent]:
    with get_connection_manager() as cm:
        workflow = Workflow.from_yaml_file(file_path=yaml_path, connection_manager=cm, init_components=True)
    agent = next(node for node in workflow.flow.nodes if isinstance(node, Agent))
    return workflow, agent


def step_tool(action: str, **action_input) -> dict[str, Any]:
    return {"action": action, "input": action_input}


def step_final(text: str) -> dict[str, Any]:
    return {"final": text}


@contextmanager
def offline_llm(scripts: dict[str, list[dict]]):
    """Replace every OpenAI LLM call by a script (STRUCTURED_OUTPUT format).

    ``scripts`` maps a marker (a phrase of an agent's role) to its steps; ``"*"`` is the default. The next step is
    chosen by how many tool results the agent's own conversation holds, so sub-agents and parallel runs are fine.
    Yields the list of prompts the LLM was sent.
    """

    seen: list[str] = []  # everything the (scripted) LLM was sent: lets an example prove what never reached it

    def run(self, **kwargs):
        messages = list(kwargs["prompt"].messages)
        seen.append("\n".join(str(m.content) for m in messages))
        system = " ".join(str(m.content) for m in messages if getattr(m.role, "value", m.role) == "system")
        steps = next((steps for marker, steps in scripts.items() if marker != "*" and marker in system), scripts["*"])
        observed = sum(1 for m in messages if str(m.content).startswith("Observation"))
        step = steps[min(observed, len(steps) - 1)]
        if "final" in step:
            reply = {"thought": "(scripted)", "action": "finish", "action_input": step["final"]}
        else:
            reply = {"thought": "(scripted)", "action": step["action"], "action_input": step["input"]}
        return RunnableResult(status=RunnableStatus.SUCCESS, input={}, output={"content": json.dumps(reply)})

    os.environ.setdefault("OPENAI_API_KEY", "offline-placeholder")
    with patch.object(OpenAI, "run", run):
        yield seen


@contextmanager
def scripted_human(answers: list[str]):
    """Answer console approval questions from ``answers`` ("" approves, anything else declines with that feedback)."""
    pending = list(answers)

    def ask(self, template, config=None):
        answer = pending.pop(0) if pending else ""
        print(f"[approval asked] {template.strip()}\n[scripted human answers] {answer!r}")
        return ApprovalInputData(feedback=answer)

    with patch.object(Node, "send_console_approval_message", ask):
        yield


@contextmanager
def llm_session(offline: bool, scripts: dict[str, list[dict]], approvals: list[str] | None = None):
    """Offline: the scripted LLM (and a scripted human for approvals), yielding the prompts the LLM was sent.
    Online: the real LLM and a real console prompt, yielding ``[]``."""
    if not offline:
        yield []
        return
    with offline_llm(scripts) as seen, scripted_human(approvals or []):
        yield seen


def run_scenario(workflow: Workflow, agent: Agent, title: str, input_data: dict, ui: bool = False) -> RunnableResult:
    """Run one question (one trace per scenario), then print the outcome and the hook events from the trace."""
    tracing, sent_to_ui = make_tracing(ui)
    print(f"\n=== {title}")
    print(f"input: {input_data.get('input')!r}  (metadata: {input_data.get('metadata', {})})")
    trusted = {"user_id": input_data.get("user_id"), "metadata": input_data.get("metadata", {})}
    result = workflow.run_sync(
        input_data=input_data, config=RunnableConfig(callbacks=[tracing], trusted_context=trusted)
    )
    output = result.output.get(agent.id, {}).get("output") if result.output else None
    print(f"status: {result.status.value}")
    if output:
        print(f"answer: {output['content']}")
        print(f"blocked: {output.get('blocked')}  blocked_by: {output.get('blocked_by')}")
    else:
        node_result = result.output.get(agent.id, {}) if result.output else {}
        print(f"error: {node_result.get('error') or 'run failed'}")

    events = [e for run in tracing.runs.values() for e in run.metadata.get("hooks", [])]
    print(f"hook events in the trace ({len(events)}):")
    for event in events:
        extras = {k: v for k, v in event.items() if k not in ("hook", "type", "point", "decision", "tool")}
        tool = f" tool={event['tool']}" if "tool" in event else ""
        print(f"  [{event['hook']}] {event['point']}{tool} -> {event['decision']} {extras or ''}")
    print(f"trace_id={tracing.trace_id} sent_to_ui={sent_to_ui}")
    return result


def report_leaks(seen: list[str], secrets: list[str]) -> None:
    """Print, for each secret, whether it ever appeared in a prompt the LLM was sent (offline runs only)."""
    text = "\n".join(seen)
    for secret in secrets:
        print(f"LLM was sent {secret!r}: {secret in text}")
