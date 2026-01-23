import json
import logging
import os

from dynamiq import Workflow, runnables
from dynamiq.callbacks import TracingCallbackHandler
from dynamiq.connections.managers import get_connection_manager
from dynamiq.utils import JsonWorkflowEncoder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

INPUT_DATA_SIMPLE = "What are the customs rules in Dubai?"
INPUT_DATA_COMPLEX = (
    "I'm traveling to Dubai with my family. What should I know about customs, attractions, and transportation?"
)
INPUT_DATA_SPECIFIC = "Tell me about Dubai Metro and how to use it"


def run_agent_rag_with_reranking(
    input_query: str,
    yaml_filename: str = "agent_rag_with_reranker.yaml",
    save_traces: bool = True,
):
    """
    Run agent RAG workflow with reranking and tracing.

    Args:
        input_query (str): User query/input
        yaml_filename (str): YAML configuration file name
        save_traces (bool): Whether to save trace data to file

    Returns:
        tuple: (workflow_result, trace_data, agent_output)
    """
    yaml_file_path = os.path.join(os.path.dirname(__file__), yaml_filename)
    logger.info(f"Loading workflow from: {yaml_file_path}")

    tracing = TracingCallbackHandler()

    with get_connection_manager() as cm:
        wf = Workflow.from_yaml_file(
            file_path=yaml_file_path,
            connection_manager=cm,
            init_components=True,
        )

        logger.info(f"Workflow ID: {wf.id}")
        logger.info(f"Flow nodes: {[node.name for node in wf.flow.nodes]}")

        agent_node = wf.flow.nodes[0]
        logger.info(f"Agent: {agent_node.name}")
        logger.info(f"Agent tools: {[tool.name for tool in agent_node.tools]}")

        for tool in agent_node.tools:
            if hasattr(tool, "ranker") and tool.ranker:
                logger.info(f"Tool '{tool.name}' has ranker: {tool.ranker.name}")

        logger.info("\n" + "=" * 80)
        logger.info("RUNNING WORKFLOW")
        logger.info("=" * 80)
        logger.info(f"Input: {input_query}\n")

        result = wf.run(
            input_data={"input": input_query},
            config=runnables.RunnableConfig(callbacks=[tracing]),
        )

        logger.info("Workflow execution completed")

        trace_data = {
            "runs": [run.to_dict() for run in tracing.runs.values()],
            "workflow_id": wf.id,
            "input": input_query,
        }

        trace_json = json.dumps(trace_data, cls=JsonWorkflowEncoder, indent=2)

        logger.info("\n" + "=" * 80)
        logger.info("WORKFLOW RESULTS")
        logger.info("=" * 80)

        agent_output = None
        for node_id, node_result in wf.flow._results.items():
            node = wf.flow._node_by_id[node_id]
            logger.info(f"\nNode: {node.name} (ID: {node_id})")
            logger.info(f"Status: {node_result.status}")

            if hasattr(node, "tools"):
                agent_output = node_result.output.get("content", "")
                logger.info(f"\nAgent Response:\n{agent_output}")

                logger.info("\n--- Agent Tool Usage ---")
                for run in tracing.runs.values():
                    if hasattr(run, "name") and "Tool" in run.name:
                        logger.info(f"\nTool: {run.name}")
                        logger.info(f"  Input: {run.input}")
                        if "documents" in run.output:
                            docs = run.output["documents"]
                            logger.info(f"  Retrieved documents: {len(docs)}")
                            for idx, doc in enumerate(docs, 1):
                                score = getattr(doc, "score", None)
                                content_preview = doc.content[:100] if hasattr(doc, "content") else ""
                                logger.info(f"    Doc {idx}: Score={score:.4f}, Preview={content_preview}...")

        logger.info("\n" + "=" * 80)

        if save_traces:
            trace_filename = f"traces_agent_rag_reranker_{hash(input_query) % 10000}.json"
            trace_path = os.path.join(os.path.dirname(__file__), trace_filename)
            with open(trace_path, "w") as f:
                f.write(trace_json)
            logger.info(f"Traces saved to: {trace_path}")

        return result, trace_data, agent_output


def compare_with_and_without_reranking():
    """
    Compare agent behavior with and without reranking.

    Note: This requires having both YAML files:
    - agent_rag_with_reranker.yaml (with ranker)
    - agent_rag.yaml (without ranker)
    """
    query = INPUT_DATA_COMPLEX

    logger.info("\n" + "=" * 80)
    logger.info("COMPARISON: Agent RAG with vs without Reranking")
    logger.info("=" * 80)

    logger.info("\n--- Running WITH Reranker ---")
    result_with, trace_with, output_with = run_agent_rag_with_reranking(
        input_query=query,
        yaml_filename="agent_rag_with_reranker.yaml",
    )

    try:
        logger.info("\n--- Running WITHOUT Reranker ---")
        result_without, trace_without, output_without = run_agent_rag_with_reranking(
            input_query=query,
            yaml_filename="agent_rag.yaml",
        )

        # Compare outputs
        logger.info("\n" + "=" * 80)
        logger.info("COMPARISON RESULTS")
        logger.info("=" * 80)
        logger.info(f"\nWith Reranker:\n{output_with}")
        logger.info(f"\nWithout Reranker:\n{output_without}")

    except FileNotFoundError:
        logger.warning("agent_rag.yaml not found for comparison")


def main():
    """Main execution function."""
    try:
        logger.info("\n" + "=" * 80)
        logger.info("Example 1: Simple Query")
        logger.info("=" * 80)
        run_agent_rag_with_reranking(INPUT_DATA_SIMPLE)

        logger.info("\n" + "=" * 80)
        logger.info("Example 2: Complex Query")
        logger.info("=" * 80)
        run_agent_rag_with_reranking(INPUT_DATA_COMPLEX)

        logger.info("\n" + "=" * 80)
        logger.info("Example 3: Specific Query")
        logger.info("=" * 80)
        run_agent_rag_with_reranking(INPUT_DATA_SPECIFIC)

        logger.info("\n" + "=" * 80)
        logger.info("All examples completed successfully!")
        logger.info("Check trace JSON files for detailed execution visualization")
        logger.info("=" * 80)

    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
