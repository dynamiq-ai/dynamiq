from concurrent.futures import ThreadPoolExecutor
from typing import Any, ClassVar, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

import dynamiq.utils.jsonpath as jsonpath
from dynamiq.nodes import Behavior, Node, NodeGroup
from dynamiq.nodes.node import Transformer, ensure_config
from dynamiq.nodes.types import ChoiceCondition, ConditionOperator
from dynamiq.runnables import RunnableConfig, RunnableResult, RunnableStatus
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

    name: str | None = "Choice"
    group: Literal[NodeGroup.OPERATORS] = NodeGroup.OPERATORS
    options: list[ChoiceOption] = []
    input_schema: ClassVar[type[ChoiceInputSchema]] = ChoiceInputSchema

    @property
    def to_dict_exclude_params(self):
        return super().to_dict_exclude_params | {"options": True}

    def to_dict(self, include_secure_params: bool = False, **kwargs) -> dict:
        """Converts the instance to a dictionary.

        Returns:
            dict: A dictionary representation of the instance.
        """
        # Separately dump ChoiceOption list as it has nested ChoiceCondition model
        # Bug: https://github.com/pydantic/pydantic/issues/9670
        data = self.model_dump(
            exclude=kwargs.pop("exclude", self.to_dict_exclude_params),
            serialize_as_any=True,
            **kwargs,
        )
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

            is_success_evaluation = False
            for option in self.options:
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
        else:
            raise ValueError(f"Operator {cond.operator} not supported.")


class MapInputSchema(BaseModel):
    input: list = Field(..., description="Parameter to provide list of inputs.")


class Map(Node):
    """Represents a map node in a flow."""

    name: str | None = "Map"
    group: Literal[NodeGroup.OPERATORS] = NodeGroup.OPERATORS
    node: Node
    behavior: Behavior | None = Behavior.RETURN
    input_schema: ClassVar[type[MapInputSchema]] = MapInputSchema
    max_workers: int = 1

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

    def execute_workflow(self, index, data, config, merged_kwargs):
        """Execute a single workflow and handle errors."""
        result = self.node.run(data, config, **merged_kwargs)
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

        try:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                results = executor.map(
                    lambda args: self.execute_workflow(args[0], args[1], config, merged_kwargs), enumerate(input_data)
                )
        except Exception as e:
            logger.error(str(e))
            raise ValueError(f"Map node failed to execute:{str(e)}")

        return {"output": list(results)}


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
