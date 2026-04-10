"""Example: Explicit SubAgentTool with checkpointing and resume (multi-node flow).

Demonstrates:
- Multi-node flow: Input → Agent (with SubAgentTool) → Python post-processor → Output
- pending_node_ids populated and shrinking as nodes complete
- node_states filled per node with input_data, output_data, internal_state
- SubAgentTool._call_count persisted in tool_states
- Checkpoint chain in APPEND mode
- Simulated crash + resume from mid-run checkpoint
"""

from dotenv import load_dotenv

from dynamiq import Workflow
from dynamiq.callbacks import TracingCallbackHandler
from dynamiq.checkpoints.backends.in_memory import InMemory
from dynamiq.checkpoints.checkpoint import CheckpointStatus
from dynamiq.checkpoints.config import CheckpointBehavior, CheckpointConfig
from dynamiq.connections import Exa
from dynamiq.flows import Flow
from dynamiq.nodes import Behavior
from dynamiq.nodes.agents import Agent
from dynamiq.nodes.node import ErrorHandling, NodeDependency
from dynamiq.nodes.tools import ExaTool
from dynamiq.nodes.tools.agent_tool import SubAgentTool
from dynamiq.nodes.tools.python import Python
from dynamiq.nodes.types import InferenceMode
from dynamiq.nodes.utils.utils import Input, Output
from dynamiq.runnables import RunnableConfig
from dynamiq.utils.logger import logger
from examples.llm_setup import setup_llm

INPUT_ID = "input-node"
AGENT_ID = "manager-agent"
CHILD_ID = "researcher-agent"
SUB_TOOL_ID = "researcher-tool"
PROCESSOR_ID = "post-processor"
OUTPUT_ID = "output-node"
FLOW_ID = "subagent-checkpoint-flow"


def make_researcher_agent(llm):
    """Child agent that has a lookup tool and a summarize tool."""
    lookup_tool = Python(
        name="Lookup",
        description='Looks up information on a topic. Expects {"topic": "<topic>"}.',
        code="""
def run(params: dict):
    topic = params.get("topic", "unknown")
    data = {
        "AI": "Artificial Intelligence is a broad field of computer science focused on creating smart machines.",
        "Python": "Python is a high-level programming language known for readability and versatility.",
        "Checkpoints": "Checkpoints save intermediate state so long-running processes can resume after failure.",
    }
    info = data.get(topic, f"No detailed info found for '{topic}', but it is a valid research topic.")
    return {"content": f"Lookup result for '{topic}': {info}"}
""",
        is_parallel_execution_allowed=True,
    )

    summarize_tool = Python(
        name="Summarize",
        description='Summarizes text. Expects {"text": "<text to summarize>"}.',
        code="""
def run(params: dict):
    text = params.get("text", "")
    words = text.split()
    summary = " ".join(words[:20]) + ("..." if len(words) > 20 else "")
    return {"content": f"Summary: {summary}"}
""",
        is_parallel_execution_allowed=True,
    )

    return Agent(
        id=CHILD_ID,
        name="Researcher",
        role=(
            "You are a research assistant. Use the Lookup tool to find information "
            "and the Summarize tool to create concise summaries."
        ),
        llm=llm,
        tools=[lookup_tool, summarize_tool],
        max_loops=4,
        parallel_tool_calls_enabled=True,
        is_parallel_execution_allowed=True,
    )


def build_flow(parent_llm, child_llm, backend):
    """Build a multi-node flow: Input → Agent (with SubAgentTool) → PostProcessor → Output."""
    researcher = make_researcher_agent(child_llm)

    researcher_tool = SubAgentTool(
        id=SUB_TOOL_ID,
        name="Researcher",
        description=(
            "Delegates research tasks to a specialist agent. "
            'Call with {"input": "<research question>"}. '
            "The researcher can look up topics and summarize findings."
        ),
        agent=researcher,
        max_calls=5,
        is_parallel_execution_allowed=True,
    )

    input_node = Input(id=INPUT_ID)
    web_search_tool = ExaTool(
        name="web-search",
        connection=Exa(),
        limit=5,
        is_optimized_for_agents=True,
        is_postponed_component_init=True,
        error_handling=ErrorHandling(
            retry_interval_seconds=1.0,
            max_retries=2,
            backoff_rate=2.0,
            behavior=Behavior.RAISE,
        ),
        is_parallel_execution_allowed=True,
    )
    agent_node = Agent(
        id=AGENT_ID,
        name="Project Manager",
        role=(
            "You are a project manager. When the user asks for research, "
            "delegate to the Researcher tool. You can call it multiple times "
            "for different topics. Use ParallelToolCallsTool to call multiple subagents."
            " Combine results into a final report.Call it couple in paralel for different research blocks"
        ),
        llm=parent_llm,
        inference_mode=InferenceMode.XML,
        tools=[researcher_tool, web_search_tool],
        max_loops=8,
        parallel_tool_calls_enabled=True,
        depends=[NodeDependency(node=input_node)],
    )

    post_processor = Python(
        id=PROCESSOR_ID,
        name="Post Processor",
        description="Formats the agent output into a clean report.",
        code="""
def run(params: dict):
    agent_output = params.get("input", "")
    report = f"=== RESEARCH REPORT ===\\n{agent_output}\\n=== END REPORT ==="
    return {"content": report}
""",
        depends=[NodeDependency(node=agent_node)],
    )

    output_node = Output(
        id=OUTPUT_ID,
        depends=[NodeDependency(node=post_processor)],
    )

    return Flow(
        id=FLOW_ID,
        nodes=[input_node, agent_node, post_processor, output_node],
        checkpoint=CheckpointConfig(
            enabled=True,
            backend=backend,
            checkpoint_after_node_enabled=True,
            checkpoint_mid_agent_loop_enabled=True,
            behavior=CheckpointBehavior.APPEND,
        ),
    )


def run_subagent_with_checkpoints():
    """Run the multi-node flow with checkpointing."""
    parent_llm = setup_llm()
    child_llm = setup_llm()
    backend = InMemory()
    flow = build_flow(parent_llm, child_llm, backend)
    load_dotenv()
    tracing = TracingCallbackHandler()
    logger.info("=== Running multi-node SubAgentTool example with checkpoints ===")
    wf = Workflow(flow=flow)

    result = wf.run(
        input_data={
            "input": (
                "Research three topics for me: 'AI' and 'Checkpoints' and 'LLM'.Use tools call in parallel "
                "to work on exa search and subagent check at the same time. "
                "Use ParallelToolCallsTool to call multiple exa tools in the same call, same for subagents. "
                "Use the researcher for each topic separately, then give me a combined report."
            ),
        },
        config=RunnableConfig(callbacks=[tracing]),
    )

    logger.info(f"Status: {result.status}")
    for node_id in [AGENT_ID, PROCESSOR_ID, OUTPUT_ID]:
        node_out = result.output.get(node_id, {}).get("output", {})
        content = node_out.get("content", str(node_out))[:120]
        logger.info(f"  [{node_id}] {content}...")

    inspect_all_checkpoints(backend)

    return backend, flow


def inspect_all_checkpoints(backend):
    """Walk the full checkpoint chain and display pending_node_ids + node_states per snapshot."""
    cp = backend.get_latest_by_flow(FLOW_ID)
    if not cp:
        logger.info("No checkpoint found.")
        return

    chain = backend.get_chain(cp.id)
    logger.info(f"\n{'='*60}")
    logger.info(f"Checkpoint chain: {len(chain)} snapshots (newest first)")
    logger.info(f"{'='*60}")

    for i, snap in enumerate(chain):
        logger.info(f"\n--- Snapshot [{i}] id={snap.id[:12]}... ---")
        logger.info(f"  status:             {snap.status}")
        logger.info(f"  completed_node_ids: {snap.completed_node_ids}")
        logger.info(f"  pending_node_ids:   {snap.pending_node_ids}")

        if snap.node_states:
            for node_id, ns in snap.node_states.items():
                logger.info(f"  node_states['{node_id}']:")
                logger.info(f"    status:     {ns.status}")
                logger.info(f"    has_input:  {ns.input_data is not None}")
                logger.info(f"    has_output: {ns.output_data is not None}")

                if ns.internal_state:
                    keys = list(ns.internal_state.keys())
                    logger.info(f"    internal_state keys: {keys}")

                    tool_states = ns.internal_state.get("tool_states", {})
                    if tool_states:
                        for tid, ts in tool_states.items():
                            logger.info(f"    tool_states['{tid}']: {ts}")

                    iteration = ns.internal_state.get("iteration")
                    if iteration:
                        logger.info(f"    iteration: completed={iteration.get('completed_iterations', 0)}")
        else:
            logger.info("  node_states: (empty)")


def simulate_crash_and_resume(backend, flow):
    """Simulate a crash mid-agent and resume, showing call_count + pending_node_ids preserved."""
    cp = backend.get_latest_by_flow(flow.id)
    if not cp:
        logger.info("No checkpoint to resume from.")
        return

    logger.info(f"\n{'='*60}")
    logger.info("=== Simulating crash (revert agent to ACTIVE) ===")
    logger.info(f"{'='*60}")
    cp.node_states[AGENT_ID].status = CheckpointStatus.ACTIVE.value
    cp.node_states[AGENT_ID].output_data = None
    cp.completed_node_ids = [nid for nid in cp.completed_node_ids if nid != AGENT_ID]
    if AGENT_ID not in cp.pending_node_ids:
        cp.pending_node_ids.append(AGENT_ID)
    for nid in [PROCESSOR_ID, OUTPUT_ID]:
        cp.node_states.pop(nid, None)
        if nid in cp.completed_node_ids:
            cp.completed_node_ids.remove(nid)
        if nid not in cp.pending_node_ids:
            cp.pending_node_ids.append(nid)
    cp.status = CheckpointStatus.ACTIVE
    backend.save(cp)

    logger.info("  Reverted checkpoint:")
    logger.info(f"    completed: {cp.completed_node_ids}")
    logger.info(f"    pending:   {cp.pending_node_ids}")

    parent_llm = setup_llm()
    child_llm = setup_llm()
    flow2 = build_flow(parent_llm, child_llm, backend)

    logger.info("\n=== Resuming from checkpoint ===")
    result2 = flow2.run_sync(input_data=None, resume_from=cp.id)

    logger.info(f"Resume status: {result2.status}")
    for node_id in [AGENT_ID, PROCESSOR_ID, OUTPUT_ID]:
        node_out = result2.output.get(node_id, {}).get("output", {})
        content = node_out.get("content", str(node_out))[:120]
        logger.info(f"  [{node_id}] {content}...")

    inspect_all_checkpoints(backend)

    return result2


if __name__ == "__main__":
    backend, flow = run_subagent_with_checkpoints()
    simulate_crash_and_resume(backend, flow)
