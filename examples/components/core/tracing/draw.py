import logging
import os.path
from collections import defaultdict
from enum import Enum
from uuid import UUID

from pydantic import BaseModel

from dynamiq.callbacks.tracing import Run, RunStatus, RunType

logger = logging.getLogger(__name__)


class GraphEdgeType(str, Enum):
    DEPENDS = "depends"
    PARENT = "parent"


class GraphEdge(BaseModel):
    source: str
    target: str
    type: GraphEdgeType


class GraphNode(BaseModel):
    id: str | UUID
    name: str
    status: RunStatus


class Graph(BaseModel):
    nodes: list[GraphNode]
    edges: list[GraphEdge]


def get_graph_by_traces(traces: list[Run]) -> Graph:
    nodes, edges = [], []
    node_by_trace_id = {}
    # Single Node ID can have multiple trace IDs (multiple runs of the same node)
    traces_ids_by_node_id = defaultdict(list)
    for trace in traces:
        if trace.type != RunType.NODE:
            continue
        node_data = trace.metadata.get("node", {})
        trace_id = str(trace.id)
        nodes.append(
            GraphNode(
                id=trace_id,
                name=node_data.get("name", "Unknown"),
                status=trace.status,
            )
        )
        for node_dep in node_data.get("depends", []):
            edges.append(
                GraphEdge(
                    source=traces_ids_by_node_id[node_dep["node"]["id"]][-1],
                    target=trace_id,
                    type=GraphEdgeType.DEPENDS,
                )
            )

        if run_node_depends := trace.metadata.get("run_depends", []):
            for run_node_dep in run_node_depends:
                edges.append(
                    GraphEdge(
                        source=traces_ids_by_node_id[run_node_dep["node"]["id"]][-1],
                        target=trace_id,
                        type=GraphEdgeType.DEPENDS,
                    )
                )
        elif parent_node := node_by_trace_id.get(str(trace.parent_run_id)):
            edges.append(
                GraphEdge(
                    source=traces_ids_by_node_id[parent_node["id"]][-1],
                    target=trace_id,
                    type=GraphEdgeType.PARENT,
                )
            )

        node_by_trace_id[trace_id] = node_data
        traces_ids_by_node_id[node_data["id"]].append(trace_id)

    return Graph(nodes=nodes, edges=edges)


def draw_graph_in_png(graph: Graph, output_path: str) -> None:
    import pygraphviz as pgv

    viz = pgv.AGraph(strict=False, directed=True, nodesep=0.9, ranksep=1.0)

    # Add nodes and edges to the graph
    for node in graph.nodes:
        # Pastel green and pastel red
        fillcolor = "#77DD77" if node.status == RunStatus.SUCCEEDED else "#FF6961"

        viz.add_node(
            str(node.id),
            label=f"<<B>{node.name}:{node.id[-4:]}</B>>",
            style="filled,rounded",
            fillcolor=fillcolor,
            fontsize=15,
            fontname="arial",
            shape="ellipse" if node.status == RunStatus.SUCCEEDED else "box",
            width=1.5,
            height=0.5,
            penwidth=2,
            fontcolor="#333333",
        )

    for seq_id, edge in enumerate(graph.edges):
        viz.add_edge(
            str(edge.source),
            str(edge.target),
            key=str(seq_id),
            fontsize=12,
            fontname="arial",
            # Pastel blue and pastel orange
            color="#779ECB" if edge.type == GraphEdgeType.DEPENDS else "#FFB347",
            style="solid" if edge.type == GraphEdgeType.DEPENDS else "dashed",
            arrowhead="normal" if edge.type == GraphEdgeType.DEPENDS else "vee",
            penwidth=2,
        )

    # Save the graph as PNG with higher resolution
    try:
        return viz.draw(output_path, format="png", prog="dot", args="-Gdpi=300")
    except Exception as e:
        logger.error(f"Error drawing graph: {e}")
    finally:
        viz.close()


def draw_simple_agent_graph_in_png(
    output_path: str = os.path.join(os.path.dirname(__file__), "simple_agent_graph.png")
) -> None:
    from examples.components.agents.agents.simple_agent_wf import run_simple_custom_workflow

    _, traces = run_simple_custom_workflow()
    graph = get_graph_by_traces([run for _, run in traces.items()])
    draw_graph_in_png(graph, output_path)


def draw_simple_graph_orchestrator_graph_in_png(
    output_path: str = os.path.join(os.path.dirname(__file__), "simple_graph_orchestrator.png")
) -> None:
    from examples.components.agents.orchestrators.graph_orchestrator.graph_orchestrator_yaml import run_workflow

    traces = run_workflow()

    graph = get_graph_by_traces([run for _, run in traces.items()])
    draw_graph_in_png(graph, output_path)


def draw_graph_orchestrator_graph_in_png(
    output_path: str = os.path.join(os.path.dirname(__file__), "graph_orchestrator.png")
) -> None:
    from examples.components.agents.orchestrators.graph_orchestrator.code_assistant import run_orchestrator

    _, traces = run_orchestrator(request="Write 100 lines of code.")

    graph = get_graph_by_traces([run for _, run in traces.items()])
    draw_graph_in_png(graph, output_path)


def draw_reflection_agent_graph_in_png(
    output_path: str = os.path.join(os.path.dirname(__file__), "reflection_agent_graph.png")
) -> None:
    from examples.components.agents.agents.reflection_agent_wf import run_workflow

    _, traces = run_workflow()
    graph = get_graph_by_traces([run for _, run in traces.items()])
    draw_graph_in_png(graph, output_path)


def draw_react_agent_graph_in_png(
    output_path: str = os.path.join(os.path.dirname(__file__), "react_agent_graph.png")
) -> None:
    from examples.components.agents.agents.react_agent_wf import run_workflow

    _, traces = run_workflow()

    graph = get_graph_by_traces([run for _, run in traces.items()])
    draw_graph_in_png(graph, output_path)


def draw_job_posting_linear_agent_graph_in_png(
    output_path: str = os.path.join(os.path.dirname(__file__), "job_posting_linear_agent_graph.png")
) -> None:
    from examples.use_cases.job_posting.main import run_planner

    _, traces = run_planner()
    graph = get_graph_by_traces([run for _, run in traces.items()])
    draw_graph_in_png(graph, output_path)


def draw_literature_overview_adaptive_agent_graph_in_png(
    output_path: str = os.path.join(os.path.dirname(__file__), "literature_overview_adaptive_agent.png")
) -> None:
    from examples.use_cases.literature_overview.main_orchestrator import run_workflow

    _, traces = run_workflow()
    graph = get_graph_by_traces([run for _, run in traces.items()])
    draw_graph_in_png(graph, output_path)


def draw_adaptive_coding_react_agent_graph_in_png(
    output_path: str = os.path.join(os.path.dirname(__file__), "adaptive_coding_react_agent.png")
) -> None:
    from examples.components.agents.orchestrators.adaptive_orchestrator.adaptive_coding_workflow import run_coding_task

    _, traces = run_coding_task()
    graph = get_graph_by_traces([run for _, run in traces.items()])
    draw_graph_in_png(graph, output_path)


def draw_multi_tool_workflow_graph_in_png(
    output_path: str = os.path.join(os.path.dirname(__file__), "multi_tool_workflow_graph.png")
) -> None:
    from examples.components.agents.agents.agent_multi_tool_workflow import run_workflow

    """Draw the execution graph of the multi-tool workflow."""
    output, traces = run_workflow()

    graph = get_graph_by_traces([run for _, run in traces.items()])
    draw_graph_in_png(graph, output_path)

    logger.info(f"Graph saved to {output_path}")
    return output


if __name__ == "__main__":
    draw_simple_agent_graph_in_png()
    draw_reflection_agent_graph_in_png()
    draw_react_agent_graph_in_png()
    draw_job_posting_linear_agent_graph_in_png()
    draw_literature_overview_adaptive_agent_graph_in_png()
    draw_adaptive_coding_react_agent_graph_in_png()
    draw_simple_graph_orchestrator_graph_in_png()
    draw_graph_orchestrator_graph_in_png()
    draw_multi_tool_workflow_graph_in_png()
