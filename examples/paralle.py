"""
Native parallel tool calling example.

Demonstrates an agent using FUNCTION_CALLING inference mode with
parallel_tool_calls_enabled=True.  In this mode the LLM provider
returns multiple tool_calls in a single response (no run_parallel
meta-tool needed).  The agent detects the multiple calls, routes
them through the existing _execute_tools batch path, and produces
the same streaming events as the run_parallel approach.

Required env vars: OPENAI_API_KEY, E2B_API_KEY, DYNAMIQ_ACCESS_KEY (or DYNAMIQ_SERVICE_TOKEN)
"""

import json

from dynamiq import Workflow
from dynamiq.callbacks.tracing import DynamiqTracingCallbackHandler
from dynamiq.connections import E2B as E2BConnection
from dynamiq.flows import Flow
from dynamiq.nodes.agents import Agent
from dynamiq.nodes.llms import OpenAI
from dynamiq.nodes.types import InferenceMode
from dynamiq.runnables import RunnableConfig
from dynamiq.sandboxes.base import SandboxConfig
from dynamiq.sandboxes.e2b import E2BSandbox
from dynamiq.types.streaming import StreamingConfig, StreamingMode
from dynamiq.utils import JsonWorkflowEncoder

llm = OpenAI(model="gpt-5.4", temperature=0.1)

e2b_sandbox = E2BSandbox(connection=E2BConnection())

agent = Agent(
    name="File Writer Assistant",
    llm=llm,
    role=(
        "You are an assistant with access to sandbox tools including "
        "FileWriteTool and SandboxShellTool. Use them to create and "
        "manage files inside the sandbox."
    ),
    inference_mode=InferenceMode.FUNCTION_CALLING,
    parallel_tool_calls_enabled=True,
    streaming=StreamingConfig(enabled=True, mode=StreamingMode.ALL),
    sandbox=SandboxConfig(enabled=True, backend=e2b_sandbox),
    max_loops=5,
)

tracing = DynamiqTracingCallbackHandler(base_url="https://collector.sandbox.getdynamiq.ai")
wf = Workflow(flow=Flow(nodes=[agent]))

result = wf.run(
    input_data={
        "input": (
            "Create these two files simultaneously:\n"
            "1) 'hello.txt' with content 'Hello from the Dynamiq sandbox!'\n"
            "2) 'notes/summary.md' with content '# Summary\n\nThis file was created by an agent.'\n"
            "Use FileWriteTool write action for both files at the same time."
        )
    },
    config=RunnableConfig(callbacks=[tracing]),
)

print("\n=== TRACING TREE ===\n")

runs_by_parent: dict[str, list] = {}
for run in tracing.runs.values():
    parent = str(run.parent_run_id) if run.parent_run_id else "root"
    runs_by_parent.setdefault(parent, []).append(run)


def print_tree(parent_key: str = "root", indent: int = 0):
    children = runs_by_parent.get(parent_key, [])
    children.sort(key=lambda r: r.start_time)
    for run in children:
        status = run.status.value if run.status else "?"
        node_name = run.metadata.get("node", {}).get("name", run.name) if run.metadata else run.name
        depends = run.metadata.get("run_depends", []) if run.metadata else []
        depend_names = [d.get("node", {}).get("name", "?") for d in depends] if depends else []
        depends_str = f" depends={depend_names}" if depend_names else ""
        print(
            f"{'  ' * indent}├── {node_name} "
            f"[{run.type.value}] "
            f"status={status} "
            f"run_id={run.id} "
            f"parent={run.parent_run_id}"
            f"{depends_str}"
        )
        print_tree(str(run.id), indent + 1)


print_tree()

print("\n=== FULL TRACING JSON ===\n")
traces_json = json.dumps(
    {"runs": [run.to_dict() for run in tracing.runs.values()]},
    cls=JsonWorkflowEncoder,
    indent=2,
)
print(traces_json[:5000])

with open("parallel_tools_trace.json", "w") as f:
    f.write(traces_json)
print(f"\nFull trace saved to parallel_tools_trace.json ({len(tracing.runs)} runs)")
