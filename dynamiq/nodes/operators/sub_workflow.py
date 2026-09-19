from typing import Any, ClassVar, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict

from dynamiq.flows import Flow
from dynamiq.nodes import Node, NodeGroup
from dynamiq.nodes.node import ensure_config
from dynamiq.nodes.types import SubWorkflowField
from dynamiq.runnables import RunnableConfig, RunnableResult, RunnableStatus
from dynamiq.types.cancellation import CanceledException
from dynamiq.types.dry_run import DryRunConfig

# The ids of the flows running above this node, carried through the run kwargs so a flow that calls
# itself, directly or through other flows, is refused before it runs on its own state.
ACTIVE_FLOWS_KWARG = "sub_workflow_flows"

# Kwargs that describe this node's own run: the flow mints its own ids, and the executor sets the rest per node.
_NODE_RUN_KWARGS = frozenset({"run_id", "parent_run_id", "execution_run_id", "run_depends"})


def _failure_reason(result: RunnableResult) -> str:
    # The flow reports which nodes failed; their own messages are what explain the failure.
    if result.error and result.error.failed_nodes:
        return "; ".join(f"{node.name or node.id}: {node.error_message}" for node in result.error.failed_nodes)
    return result.error.message if result.error else "unknown error"


class SubWorkflowInputSchema(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")


class SubWorkflow(Node):
    """Runs another flow as one step of this one.

    The mapped inputs are the inner flow's input, and the result of its Output node is this node's
    output; a flow without exactly one Output node returns every node's output keyed by node id. The
    inner run is traced under this node's run and executes a copy of the flow, so several nodes may
    hold the same flow and run at once. `workflow_id` and `workflow_version_id` name the
    workflow the flow was resolved from and are kept for the platform; the flow itself is what runs.
    A flow that is already running above this node, which is what a workflow calling itself looks
    like at run time, fails the run instead of recursing.
    """

    name: str | None = "sub_workflow"
    group: Literal[NodeGroup.OPERATORS] = NodeGroup.OPERATORS
    flow: Flow | None = None
    workflow_id: str | None = None
    workflow_version_id: str | None = None
    input_fields: list[SubWorkflowField] = []
    output_fields: list[SubWorkflowField] = []
    input_schema: ClassVar[type[SubWorkflowInputSchema]] = SubWorkflowInputSchema

    @property
    def to_dict_exclude_params(self):
        return super().to_dict_exclude_params | {"flow": True}

    def to_dict(self, **kwargs) -> dict:
        """The flow is referenced by id; the workflow dump emits it in its own section."""
        data = super().to_dict(**kwargs)
        data["flow"] = self.flow.id if self.flow else None
        return data

    def dry_run_cleanup(self, dry_run_config: DryRunConfig | None = None) -> None:
        for node in self.flow.nodes if self.flow else []:
            node.dry_run_cleanup(dry_run_config)

    def execute(self, input_data: SubWorkflowInputSchema, config: RunnableConfig = None, **kwargs) -> Any:
        """Runs the flow with the inputs and returns its Output node's result."""
        if self.flow is None:
            raise ValueError(f"Sub-workflow '{self.name}' has no flow to run")
        active_flows: tuple[str, ...] = kwargs.get(ACTIVE_FLOWS_KWARG, ())
        if self.flow.id in active_flows:
            raise ValueError(f"Sub-workflow '{self.name}': flow '{self.flow.id}' is already running above this node")

        values = input_data.model_dump()
        missing = [field.name for field in self.input_fields if field.required and values.get(field.name) is None]
        if missing:
            raise ValueError(f"Sub-workflow '{self.name}': required inputs missing: {', '.join(missing)}")

        run_id = kwargs.get("run_id", uuid4())
        config = ensure_config(config)
        self.run_on_node_execute_run(config.callbacks, **{**kwargs, "parent_run_id": run_id})

        flow_kwargs = {key: value for key, value in kwargs.items() if key not in _NODE_RUN_KWARGS} | {
            "parent_run_id": run_id,
            ACTIVE_FLOWS_KWARG: active_flows + (self.flow.id,),
        }
        # A flow keeps its run state on itself, so each run executes its own copy: two nodes that hold
        # the same flow, or one node run twice at once, would otherwise overwrite each other's results.
        flow = self.flow.clone()
        # The parent's checkpoint settings, a resume id above all, describe the parent's run; the copied
        # flow runs whole on its own checkpoint config, and the parent records this node once it completes.
        result = flow.run_sync(values, config.model_copy(update={"checkpoint": None}), **flow_kwargs)
        if result.status == RunnableStatus.CANCELED:
            raise CanceledException()
        if result.status != RunnableStatus.SUCCESS:
            raise ValueError(f"Sub-workflow '{self.name}' failed: {_failure_reason(result)}")
        return self._output_of(flow, result.output)

    def _output_of(self, flow: Flow, results: dict[str, dict[str, Any]]) -> Any:
        # Local import to avoid a circular import: nodes.utils builds Input and Output on the Pass operator.
        from dynamiq.nodes.utils import Output

        output_nodes = [node for node in flow.nodes if isinstance(node, Output)]
        if len(output_nodes) == 1:
            return results[output_nodes[0].id]["output"]
        return {node_id: result["output"] for node_id, result in results.items()}
