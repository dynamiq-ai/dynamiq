from typing import Any, Callable, ClassVar, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, PrivateAttr

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
    like at run time, fails the run instead of recursing. Under a dry run, what the copy's writers
    ingested is cleaned up when the flow holding this node ends, not when the copy returns, so a node
    after this one still reads it.
    """

    name: str | None = "sub_workflow"
    group: Literal[NodeGroup.OPERATORS] = NodeGroup.OPERATORS
    flow: Flow | None = None
    workflow_id: str | None = None
    workflow_version_id: str | None = None
    input_fields: list[SubWorkflowField] = []
    output_fields: list[SubWorkflowField] = []
    input_schema: ClassVar[type[SubWorkflowInputSchema]] = SubWorkflowInputSchema
    _dry_run_flows: list[Flow] = PrivateAttr(default_factory=list)

    @property
    def to_dict_exclude_params(self):
        return super().to_dict_exclude_params | {"flow": True}

    def to_dict(self, **kwargs) -> dict:
        """The flow is referenced by id; the workflow dump emits it in its own section."""
        data = super().to_dict(**kwargs)
        data["flow"] = self.flow.id if self.flow else None
        return data

    def get_clone_attr_initializers(self) -> dict[str, Callable[[Node], Any]]:
        # A shallow copy would share the list: the copies that ran on this node are its own to clean up.
        return super().get_clone_attr_initializers() | {"_dry_run_flows": lambda _: []}

    def dry_run_cleanup(self, dry_run_config: DryRunConfig | None = None) -> None:
        """Cleans up what the copies that ran under a dry run wrote.

        The flow held here never runs itself, so its nodes hold nothing to clean. Each run executes
        a copy that is told not to clean up at its own end: the flow holding this node reaches this
        hook once its whole run ends, so a node after this one still reads what the sub-workflow
        ingested, the way it would with the same nodes inlined.
        """
        flows, self._dry_run_flows = self._dry_run_flows, []
        for flow in flows:
            flow.dry_run_cleanup(dry_run_config)

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
        dry_run = config.dry_run if config.dry_run and config.dry_run.enabled else None
        if dry_run:
            # Kept before the run, so what a copy wrote before failing or timing out is cleaned up too.
            self._dry_run_flows.append(flow)
        # The parent's checkpoint settings, a resume id above all, describe the parent's run; the copied
        # flow runs whole on its own checkpoint config, and the parent records this node once it completes.
        # Nor does the copy clean up its dry-run writes at its end: the flow holding this node does,
        # through dry_run_cleanup, once the nodes after this one have read them.
        result = flow.run_sync(
            values, config.model_copy(update={"checkpoint": None}), cleanup_dry_run=False, **flow_kwargs
        )
        if dry_run and all(kept is not flow for kept in self._dry_run_flows):
            # A timed-out run keeps going in its thread after this node has failed; a copy that ends after
            # the flow holding this node cleaned up was released without its writes, so it cleans up now.
            flow.dry_run_cleanup(dry_run)
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
