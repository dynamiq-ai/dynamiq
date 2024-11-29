from enum import Enum
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field

import dynamiq.utils.jsonpath as jsonpath
from dynamiq.nodes import Behavior, Node, NodeGroup
from dynamiq.nodes.node import Transformer, ensure_config
from dynamiq.runnables import RunnableConfig, RunnableResult, RunnableStatus
from dynamiq.utils import generate_uuid
from dynamiq.utils.logger import logger


class ConditionOperator(Enum):
    """Enum representing various condition operators for choice nodes."""

    OR = "or"
    AND = "and"
    BOOLEAN_EQUALS = "boolean-equals"
    BOOLEAN_EQUALS_PATH = "boolean-equals-path"
    NUMERIC_EQUALS = "numeric-equals"
    NUMERIC_EQUALS_PATH = "numeric-equals-path"
    NUMERIC_GREATER_THAN = "numeric-greater-than"
    NUMERIC_GREATER_THAN_PATH = "numeric-greater-than-path"
    NUMERIC_GREATER_THAN_OR_EQUALS = "numeric-greater-than-or-equals"
    NUMERIC_GREATER_THAN_OR_EQUALS_PATH = "numeric-greater-than-or-equals-path"
    NUMERIC_LESS_THAN = "numeric-less-than"
    NUMERIC_LESS_THAN_PATH = "numeric-less-than-path"
    NUMERIC_LESS_THAN_OR_EQUALS = "numeric-less-than-or-equals"
    NUMERIC_LESS_THAN_OR_EQUALS_PATH = "numeric-less-than-or-equals-path"
    STRING_EQUALS = "string-equals"
    STRING_EQUALS_PATH = "string-equals-path"
    STRING_GREATER_THAN = "string-greater-than"
    STRING_GREATER_THAN_PATH = "string-greater-than-path"
    STRING_GREATER_THAN_OR_EQUALS = "string-greater-than-or-equals"
    STRING_GREATER_THAN_OR_EQUALS_PATH = "string-greater-than-or-equals-path"
    STRING_LESS_THAN = "string-less-than"
    STRING_LESS_THAN_PATH = "string-less-than-path"
    STRING_LESS_THAN_OR_EQUALS = "string-less-than-or-equals"
    STRING_LESS_THAN_OR_EQUALS_PATH = "string-less-than-or-equals-path"


class ChoiceCondition(BaseModel):
    """Represents a condition for a choice node."""

    variable: str = None
    operator: ConditionOperator = None
    value: Any = None
    is_not: bool = False
    operands: list["ChoiceCondition"] = None


class ChoiceOption(BaseModel):
    """Represents an option for a choice node."""

    id: str = Field(default_factory=generate_uuid)
    name: str | None = None
    condition: ChoiceCondition | None = None


class ChoiceExecute(BaseModel):
    """Represents the execution of a choice."""

    condition: ChoiceCondition | None = None


class Choice(Node):
    """Represents a choice node in a flow."""

    name: str | None = "Choice"
    group: Literal[NodeGroup.OPERATORS] = NodeGroup.OPERATORS
    options: list[ChoiceOption] = []

    @property
    def to_dict_exclude_params(self):
        return super().to_dict_exclude_params | {"options": True}

    def to_dict(self, **kwargs) -> dict:
        """Converts the instance to a dictionary.

        Returns:
            dict: A dictionary representation of the instance.
        """
        # Separately dump ChoiceOption list as it has nested ChoiceCondition model
        # Bug: https://github.com/pydantic/pydantic/issues/9670
        data = self.model_dump(
            exclude=self.to_dict_exclude_params,
            serialize_as_any=True,
            **kwargs,
        )
        data["options"] = [option.model_dump(**kwargs) for option in self.options]
        return data

    def execute(
        self, input_data: dict[str, Any], config: RunnableConfig = None, **kwargs
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
                        status=RunnableStatus.SKIP, input=input_data, output=None
                    )
                elif option.condition and self.evaluate(option.condition, input_data):
                    results[option.id] = RunnableResult(
                        status=RunnableStatus.SUCCESS, input=input_data, output=True
                    )
                    is_success_evaluation = True
                elif not option.condition:
                    results[option.id] = RunnableResult(
                        status=RunnableStatus.SUCCESS, input=input_data, output=True
                    )
                    is_success_evaluation = True
                else:
                    results[option.id] = RunnableResult(
                        status=RunnableStatus.FAILURE, input=input_data, output=False
                    )

        return results

    @staticmethod
    def evaluate(cond: ChoiceCondition, input: Any) -> bool:
        """
        Evaluates a choice condition.

        Args:
            cond: The condition to evaluate.
            input: The input data to evaluate against.

        Returns:
            A boolean indicating whether the condition is met.

        Raises:
            ValueError: If the operator is not supported.
        """
        value = jsonpath.filter(input, cond.variable, "blah")

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
            return value == cond.value and not cond.is_not
        # numeric
        if cond.operator == ConditionOperator.NUMERIC_EQUALS:
            return value == cond.value and not cond.is_not
        elif cond.operator == ConditionOperator.NUMERIC_GREATER_THAN:
            return value > cond.value and not cond.is_not
        elif cond.operator == ConditionOperator.NUMERIC_GREATER_THAN_OR_EQUALS:
            return value >= cond.value and not cond.is_not
        elif cond.operator == ConditionOperator.NUMERIC_LESS_THAN:
            return value < cond.value and not cond.is_not
        elif cond.operator == ConditionOperator.NUMERIC_LESS_THAN_OR_EQUALS:
            return value <= cond.value and not cond.is_not
        # string
        elif cond.operator == ConditionOperator.STRING_EQUALS:
            return value == cond.value and not cond.is_not
        elif cond.operator == ConditionOperator.STRING_GREATER_THAN:
            return value > cond.value and not cond.is_not
        elif cond.operator == ConditionOperator.STRING_GREATER_THAN_OR_EQUALS:
            return value >= cond.value and not cond.is_not
        elif cond.operator == ConditionOperator.STRING_LESS_THAN:
            return value < cond.value and not cond.is_not
        elif cond.operator == ConditionOperator.STRING_LESS_THAN_OR_EQUALS:
            return value <= cond.value and not cond.is_not
        else:
            raise ValueError(f"Operator {cond.operator} not supported.")


class Map(Node):
    """Represents a map node in a flow."""

    name: str | None = "Map"
    group: Literal[NodeGroup.OPERATORS] = NodeGroup.OPERATORS
    node: Node
    behavior: Behavior | None = Behavior.RETURN

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

    def execute(
        self, input_data: dict[str, Any], config: RunnableConfig = None, **kwargs
    ):
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
        if isinstance(input_data, dict):
            input_data = input_data["input"]

        if not isinstance(input_data, list):
            logger.error(f"Map operator {self.id} input is not a list.")
            raise ValueError(f"Map operator {self.id} input is not a list.")

        output = []
        run_id = kwargs.get("run_id", uuid4())
        config = ensure_config(config)
        merged_kwargs = {**kwargs, "parent_run_id": run_id}

        self.run_on_node_execute_run(config.callbacks, **kwargs)

        for index, data in enumerate(input_data, start=1):
            result = self.node.run(data, config, **merged_kwargs)
            if result.status != RunnableStatus.SUCCESS:
                if self.behavior == Behavior.RAISE:
                    raise ValueError(f"Map node failed to execute: node under iteration index {index} has failed.")
                logger.error(f"Node under iteration index {index} has failed.")
            output.append(result.output)

        return {"output": output}


class Pass(Node):
    """Represents a pass node in a flow."""

    group: Literal[NodeGroup.OPERATORS] = NodeGroup.OPERATORS
    transformers: list[Transformer] = []

    def execute_with_retry(
        self, input_data: dict[str, Any], config: RunnableConfig = None, **kwargs
    ):
        """
        Executes the pass node with retry logic.

        Args:
            input_data: The input data for the node.
            config: The runnable configuration.
            **kwargs: Additional keyword arguments.

        Returns:
            The input data if no transformers are present, otherwise the transformed data.
        """
        if self.transformers:
            return super().execute_with_retry(input_data, config, **kwargs)

        return input_data

    def execute(
        self, input_data: dict[str, Any], config: RunnableConfig = None, **kwargs
    ):
        """
        Executes the pass node.

        Args:
            input_data: The input data for the node.
            config: The runnable configuration.
            **kwargs: Additional keyword arguments.

        Returns:
            The input data if no transformers are present, otherwise the transformed data.
        """
        output = input_data
        if self.transformers:
            run_id = kwargs.get("run_id", uuid4())
            config = ensure_config(config)
            merged_kwargs = {**kwargs, "parent_run_id": run_id}

            self.run_on_node_execute_run(config.callbacks, **merged_kwargs)

            for transformer in self.transformers:
                output = self.transform(output, transformer, self.id)

        return output
