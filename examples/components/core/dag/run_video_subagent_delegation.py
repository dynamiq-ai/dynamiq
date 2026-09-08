import json
import logging
import os
import sys

from dotenv import find_dotenv, load_dotenv

from dynamiq import Workflow, runnables
from dynamiq.callbacks import DynamiqTracingCallbackHandler, TracingCallbackHandler
from dynamiq.connections.managers import get_connection_manager
from dynamiq.utils import JsonWorkflowEncoder

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

load_dotenv(find_dotenv(), override=True)

DEFAULT_QUESTION = (
    "Explain what's happening in the attached video: describe the scene, call out any "
    "problem or issue shown, and summarize what's going on overall."
)


def make_tracer() -> TracingCallbackHandler:
    access_key = os.environ.get("DYNAMIQ_TRACE_ACCESS_KEY")
    if not access_key:
        return TracingCallbackHandler()
    base_url = os.environ.get("DYNAMIQ_TRACE_BASE_URL", "https://collector.sandbox.getdynamiq.ai")
    logger.info("DYNAMIQ_TRACE_ACCESS_KEY set -- streaming trace to %s", base_url)
    return DynamiqTracingCallbackHandler(base_url=base_url, access_key=access_key)


def main():
    if len(sys.argv) < 2:
        raise SystemExit(f"Usage: python {sys.argv[0]} <video_path> [question] [--trace-out path.json]")
    video_path = sys.argv[1]
    question = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] != "--trace-out" else DEFAULT_QUESTION
    trace_out = os.path.join(os.path.dirname(__file__), "video_subagent_trace.json")
    if "--trace-out" in sys.argv:
        trace_out = sys.argv[sys.argv.index("--trace-out") + 1]

    yaml_path = os.path.join(os.path.dirname(__file__), "video_subagent_delegation.yaml")
    with get_connection_manager() as cm:
        wf = Workflow.from_yaml_file(file_path=yaml_path, connection_manager=cm, init_components=True)

        with open(video_path, "rb") as f:
            video_bytes = f.read()

        tracer = make_tracer()
        result = wf.run(
            input_data={"input": question, "videos": [video_bytes]},
            config=runnables.RunnableConfig(callbacks=[tracer]),
        )

        parent_agent = wf.flow.nodes[0]
        node_result = result.output.get(parent_agent.id, {})
        content = node_result.get("output", {}).get("content")
        logger.info("status: %s", node_result.get("status"))
        logger.info("answer: %s", json.dumps(content, indent=2) if isinstance(content, dict) else content)

        with open(trace_out, "w") as f:
            json.dump({"runs": [run.to_dict() for run in tracer.runs.values()]}, f, cls=JsonWorkflowEncoder, indent=2)
        logger.info("trace written to %s (%d runs)", trace_out, len(tracer.runs))

        try:
            from examples.components.core.tracing.draw import draw_graph_in_png, get_graph_by_traces

            png_out = os.path.splitext(trace_out)[0] + ".png"
            graph = get_graph_by_traces(list(tracer.runs.values()))
            draw_graph_in_png(graph, png_out)
            logger.info("execution graph written to %s", png_out)
        except ImportError:
            logger.info("pygraphviz not installed -- skipping PNG graph (JSON trace above still has everything)")


if __name__ == "__main__":
    main()
