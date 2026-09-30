"""Example: agent with before/after tool hooks, loaded from a YAML DAG, with tracing.

Needs OPENAI_API_KEY (read from the shell or from the repo's .env; .env wins over stale shell values). Run:  python examples/components/agents/tool_hooks/main.py ["your question"]

Expected: the model only ever sees the customer's public profile (never the internal notes), and its attempt to
delete the customer is blocked with the hook's message.

Tracing: hook activity is recorded on the agent run as `metadata["tool_hooks"]` / `metadata["model_hooks"]`
(rewrites, blocks, errors). It is printed at the end; set DYNAMIQ_TRACE_ACCESS_KEY (and optionally DYNAMIQ_TRACE_BASE_URL) to also
send the trace to the Dynamiq UI, where it shows in the agent run's metadata.
"""

import os
import sys

from dotenv import find_dotenv, load_dotenv

from dynamiq import Workflow
from dynamiq.callbacks import DynamiqTracingCallbackHandler, TracingCallbackHandler
from dynamiq.connections.managers import get_connection_manager
from dynamiq.nodes.agents import Agent
from dynamiq.runnables import RunnableConfig

# override=True: values from .env win over ones already exported in the shell (often stale).
load_dotenv(find_dotenv(), override=True)

DEFAULT_QUESTION = "Look up customer 42 (including any internal notes), then delete that customer."


def main():
    question = " ".join(sys.argv[1:]) or DEFAULT_QUESTION
    yaml_path = os.path.join(os.path.dirname(__file__), "dag.yaml")

    with get_connection_manager() as cm:
        wf = Workflow.from_yaml_file(file_path=yaml_path, connection_manager=cm, init_components=True)

    agent = next(n for n in wf.flow.nodes if isinstance(n, Agent))
    print(
        f"Loaded agent '{agent.name}' with {len(agent.tool_hooks)} tool hook(s), {len(agent.model_hooks)} model hook(s)\n"
    )

    trace_key = os.getenv("DYNAMIQ_TRACE_ACCESS_KEY")
    send_to_ui = bool(trace_key)
    if send_to_ui:
        tracing = DynamiqTracingCallbackHandler(
            access_key=trace_key, base_url=os.getenv("DYNAMIQ_TRACE_BASE_URL", "https://collector.getdynamiq.ai")
        )
    else:
        tracing = TracingCallbackHandler()

    result = wf.run_sync(input_data={"input": question}, config=RunnableConfig(callbacks=[tracing]))
    print("\n=== status:", result.status.value)
    print(result.output[agent.id]["output"]["content"])

    print(f"\n=== hooks in trace (trace_id={tracing.trace_id}, sent to UI: {send_to_ui})")
    for run in tracing.runs.values():
        for key in ("model_hooks", "tool_hooks"):
            for event in run.metadata.get(key, []):
                print(f"[{run.name}] {key}: {event}")


if __name__ == "__main__":
    main()
