import re
from typing import Any, Callable, ClassVar, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

import dynamiq.utils.jsonpath as jsonpath
from dynamiq.executors.context import ContextAwareThreadPoolExecutor
from dynamiq.nodes import Behavior, Node, NodeGroup
from dynamiq.nodes.cloning import carry_mock_exclusions, regenerate_node_ids
from dynamiq.nodes.node import Transformer, ensure_config
from dynamiq.nodes.tools.mcp import resolve_mcp_node
from dynamiq.nodes.types import ChoiceCondition, ChoiceHitPolicy, ConditionOperator
from dynamiq.runnables import RunnableConfig, RunnableResult, RunnableStatus
from dynamiq.types.cancellation import CanceledException, check_cancellation
from dynamiq.types.dry_run import DryRunConfig
from dynamiq.utils import generate_uuid
from dynamiq.utils.logger import logger


class ChoiceOption(BaseModel):
    """Represents an option for a choice node."""

    id: str = Field(default_factory=generate_uuid)
    name: str | None = None
    condition: ChoiceCondition | None = None


class ChoiceInputSchema(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")


class Choice(Node):
    """Represents a choice node in a flow."""

    name: str | None = "choice"
    group: Literal[NodeGroup.OPERATORS] = NodeGroup.OPERATORS
    options: list[ChoiceOption] = []
    hit_policy: ChoiceHitPolicy = ChoiceHitPolicy.FIRST
    input_schema: ClassVar[type[ChoiceInputSchema]] = ChoiceInputSchema

    @property
    def to_dict_exclude_params(self):
        return super().to_dict_exclude_params | {"options": True}

    def to_dict(self, include_secure_params: bool = True, for_tracing=False, **kwargs) -> dict:
        """Converts the instance to a dictionary.

        Returns:
            dict: A dictionary representation of the instance.
        """
        data = super().to_dict(include_secure_params=include_secure_params, for_tracing=for_tracing, **kwargs)
        data["options"] = [option.model_dump(**kwargs) for option in self.options]
        return data

    def execute(
        self, input_data: ChoiceInputSchema, config: RunnableConfig = None, **kwargs
    ) -> dict[str, RunnableResult]:
        """
        Executes the choice node.

        Args:
            input_data: The input data for the node.
            config: The runnable configuration.
            **kwargs: Additional keyword arguments.

        Returns:
            A dictionary of RunnableResults for each option.
        """
        results = {}
        if self.options:
            run_id = kwargs.get("run_id", uuid4())
            config = ensure_config(config)
            merged_kwargs = {**kwargs, "parent_run_id": run_id}

            self.run_on_node_execute_run(config.callbacks, **merged_kwargs)

            if self.hit_policy == ChoiceHitPolicy.ALL:
                return self._evaluate_all(input_data)

            is_success_evaluation = False
            for option in self.options:
                # The first match ends the walk: every option after it is skipped, an option without a
                # condition matching on its own.
                if is_success_evaluation:
                    results[option.id] = RunnableResult(
                        status=RunnableStatus.SKIP, input=input_data.model_dump(), output=None
                    )
                elif option.condition and self.evaluate(option.condition, input_data.model_dump()):
                    results[option.id] = RunnableResult(
                        status=RunnableStatus.SUCCESS, input=input_data.model_dump(), output=True
                    )
                    is_success_evaluation = True
                elif not option.condition:
                    results[option.id] = RunnableResult(
                        status=RunnableStatus.SUCCESS, input=input_data.model_dump(), output=True
                    )
                    is_success_evaluation = True
                else:
                    results[option.id] = RunnableResult(
                        status=RunnableStatus.FAILURE, input=input_data.model_dump(), output=False
                    )

        return results

    def _evaluate_all(self, input_data: ChoiceInputSchema) -> dict[str, RunnableResult]:
        """Every conditioned option judged on its own; a fallback runs only when none of them held.

        A fallback, an option without a condition, is decided over the whole list rather than over the
        options before it, so it stays the branch for a record nothing routed wherever it sits.
        """
        values = input_data.model_dump()
        held = [
            (option, self.evaluate(option.condition, values) if option.condition else None) for option in self.options
        ]
        any_held = any(matched for _, matched in held)
        results = {}
        for option, matched in held:
            if option.condition:
                status = RunnableStatus.SUCCESS if matched else RunnableStatus.FAILURE
                results[option.id] = RunnableResult(status=status, input=input_data.model_dump(), output=matched)
            elif any_held:
                results[option.id] = RunnableResult(
                    status=RunnableStatus.SKIP, input=input_data.model_dump(), output=None
                )
            else:
                results[option.id] = RunnableResult(
                    status=RunnableStatus.SUCCESS, input=input_data.model_dump(), output=True
                )
        return results

    @staticmethod
    def evaluate(cond: ChoiceCondition, input_data: Any) -> bool:
        """
        Evaluates a choice condition.

        Args:
            cond: The condition to evaluate.
            input_data: The input data to evaluate against.

        Returns:
            A boolean indicating whether the condition is met.

        Raises:
            ValueError: If the operator is not supported.
        """
        value = jsonpath.filter(input_data, cond.variable)

        if cond.operator == ConditionOperator.OR:
            return (
                any(Choice.evaluate(cond, value) for cond in cond.operands)
                and not cond.is_not
            )
        elif cond.operator == ConditionOperator.AND:
            return (
                all(Choice.evaluate(cond, value) for cond in cond.operands)
                and not cond.is_not
            )
        # boolean
        elif cond.operator == ConditionOperator.BOOLEAN_EQUALS:
            return (value == cond.value) == (not cond.is_not)
        # numeric
        if cond.operator == ConditionOperator.NUMERIC_EQUALS:
            return (value == cond.value) == (not cond.is_not)
        elif cond.operator == ConditionOperator.NUMERIC_GREATER_THAN:
            return (value > cond.value) == (not cond.is_not)
        elif cond.operator == ConditionOperator.NUMERIC_GREATER_THAN_OR_EQUALS:
            return (value >= cond.value) == (not cond.is_not)
        elif cond.operator == ConditionOperator.NUMERIC_LESS_THAN:
            return (value < cond.value) == (not cond.is_not)
        elif cond.operator == ConditionOperator.NUMERIC_LESS_THAN_OR_EQUALS:
            return (value <= cond.value) == (not cond.is_not)
        # string
        elif cond.operator == ConditionOperator.STRING_EQUALS:
            return (value == cond.value) == (not cond.is_not)
        elif cond.operator == ConditionOperator.STRING_GREATER_THAN:
            return (value > cond.value) == (not cond.is_not)
        elif cond.operator == ConditionOperator.STRING_GREATER_THAN_OR_EQUALS:
            return (value >= cond.value) == (not cond.is_not)
        elif cond.operator == ConditionOperator.STRING_LESS_THAN:
            return (value < cond.value) == (not cond.is_not)
        elif cond.operator == ConditionOperator.STRING_LESS_THAN_OR_EQUALS:
            return (value <= cond.value) == (not cond.is_not)
        elif cond.operator == ConditionOperator.STRING_STARTS_WITH:
            return (str(value).startswith(str(cond.value))) == (not cond.is_not)
        elif cond.operator == ConditionOperator.STRING_CONTAINS:
            return (str(cond.value) in str(value)) == (not cond.is_not)
        elif cond.operator == ConditionOperator.STRING_REGEXP:
            try:
                return bool(re.search(str(cond.value), str(value))) == (not cond.is_not)
            except re.error as e:
                raise ValueError(f"Invalid regular expression '{cond.value}': {e}")
        elif cond.operator == ConditionOperator.STRING_ENDS_WITH:
            return (str(value).endswith(str(cond.value))) == (not cond.is_not)
        else:
            raise ValueError(f"Operator {cond.operator} not supported.")


class MapInputSchema(BaseModel):
    input: list = Field(..., description="Parameter to provide list of inputs.")


class Map(Node):
    """Represents a map node in a flow."""

    name: str | None = "map"
    group: Literal[NodeGroup.OPERATORS] = NodeGroup.OPERATORS
    node: Node
    behavior: Behavior | None = Behavior.RETURN
    input_schema: ClassVar[type[MapInputSchema]] = MapInputSchema
    max_workers: int = 1
    _dry_run_nodes: list[Node] = PrivateAttr(default_factory=list)

    @property
    def to_dict_exclude_params(self):
        """
        Property to define which parameters should be excluded when converting the class instance to a dictionary.

        Returns:
            dict: A dictionary defining the parameters to exclude.
        """
        return super().to_dict_exclude_params | {"node": True}

    def to_dict(self, **kwargs) -> dict:
        """Converts the instance to a dictionary.

        Returns:
            dict: A dictionary representation of the instance.
        """
        data = super().to_dict(**kwargs)
        data["node"] = self.node.to_dict(**kwargs)
        return data

    def get_clone_attr_initializers(self) -> dict[str, Callable[[Node], Any]]:
        # A shallow copy would share the list: the clones that ran on this node are its own to clean up.
        return super().get_clone_attr_initializers() | {"_dry_run_nodes": lambda _: []}

    def dry_run_cleanup(self, dry_run_config: DryRunConfig | None = None) -> None:
        """Cleans up what the node and the clones that ran per item under a dry run wrote.

        Each item runs on a clone, which is what holds the writes and, for a sub-workflow, the copies
        of its flow waiting for this cleanup; one clone's failure stops no other.
        """
        nodes, self._dry_run_nodes = self._dry_run_nodes, []
        for node in [self.node, *nodes]:
            try:
                node.dry_run_cleanup(dry_run_config)
            except Exception as e:
                logger.error(f"Map: failed to clean up dry run resources for node {node.id}: {e}")

    def execute_workflow(self, index, data, config, merged_kwargs, node):
        """Execute a single workflow and handle errors."""
        id_map: dict[str, set[str]] = {}
        node_copy = regenerate_node_ids(node.clone(), id_map)
        # Only a clone that overrides the base hook holds anything to clean; keeping the rest would
        # retain one node per item for a whole run, and a dry run is the default.
        if (
            config is not None
            and config.dry_run
            and config.dry_run.enabled
            and node_copy.dry_run_cleanup.__qualname__ != "Node.dry_run_cleanup"
        ):
            self._dry_run_nodes.append(node_copy)

        # Create an isolated config per iteration with unique streaming override for the cloned node
        local_config = config
        try:
            local_config = config.model_copy(deep=False) if config is not None else RunnableConfig()
            if node_config := local_config.nodes_override.get(self.node.id):
                local_config.nodes_override[node_copy.id] = node_config
            local_config = carry_mock_exclusions(local_config, id_map)
        except Exception as e:
            logger.warning(f"Map: failed to prepare isolated streaming config for iteration {index}: {e}")

        result = node_copy.run(data, local_config, **merged_kwargs)
        if result.status == RunnableStatus.CANCELED:
            raise CanceledException()
        if result.status != RunnableStatus.SUCCESS:
            if self.behavior == Behavior.RAISE:
                raise ValueError(f"Node under iteration index {index + 1} has failed.")
        return result.output

    def execute(self, input_data: MapInputSchema, config: RunnableConfig = None, **kwargs):
        """
        Executes the map node.

        Args:
            input_data: The input data for the node.
            config: The runnable configuration.
            **kwargs: Additional keyword arguments.

        Returns:
            A list of outputs from executing the flow on each input item.

        Raises:
            Exception: If the input is not a list or if any flow execution fails.
        """
        input_data = input_data.input

        run_id = kwargs.get("run_id", uuid4())
        config = ensure_config(config)
        merged_kwargs = {**kwargs, "parent_run_id": run_id}
        self.run_on_node_execute_run(config.callbacks, **kwargs)

        # Resolve an MCPServer to its single MCPTool.
        run_node = resolve_mcp_node(self.node)

        try:
            check_cancellation(config)
            with ContextAwareThreadPoolExecutor(max_workers=self.max_workers) as executor:
                results = list(
                    executor.map(
                        lambda args: self.execute_workflow(args[0], args[1], config, merged_kwargs, run_node),
                        enumerate(input_data),
                    )
                )
        except CanceledException:
            if config and getattr(config, "cancellation", None) and config.cancellation.token:
                config.cancellation.token.cancel()
            raise
        except Exception as e:
            logger.error(str(e))
            raise ValueError(f"Map node failed to execute:{str(e)}")

        return {"output": results}


class Pass(Node):
    """Represents a pass node in a flow."""

    group: Literal[NodeGroup.OPERATORS] = NodeGroup.OPERATORS
    transformers: list[Transformer] = []

    def execute(self, input_data: dict[str, Any], config: RunnableConfig = None, **kwargs):
        """
        Executes the pass node.

        Args:
            input_data: The input data for the node.
            config: The runnable configuration.
            **kwargs: Additional keyword arguments.

        Returns:
            The input data if no transformers are present, otherwise the transformed data.
        """
        config = ensure_config(config)
        merged_kwargs = {**kwargs, "parent_run_id": kwargs.get("run_id", uuid4())}
        self.run_on_node_execute_run(config.callbacks, **merged_kwargs)

        output = input_data
        for transformer in self.transformers:
            output = self.transform(output, transformer)

        return output
