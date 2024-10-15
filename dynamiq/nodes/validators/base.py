from abc import abstractmethod
from typing import Any, Literal

from dynamiq.nodes import Behavior, Node, NodeGroup
from dynamiq.nodes.node import ensure_config
from dynamiq.runnables import RunnableConfig


class BaseValidator(Node):
    group: Literal[NodeGroup.VALIDATORS] = NodeGroup.VALIDATORS
    name: str | None = "Validator"
    behavior: Behavior | None = Behavior.RETURN

    def execute(
        self, input_data: dict[str, Any], config: RunnableConfig = None, **kwargs
    ):
        """Executes the validation process for a given value.

        Args:
            input_data (dict[str, Any]): The input data containing the value to check under "content" key.
            config (RunnableConfig, optional): Configuration for the execution. Defaults to None.
            **kwargs: Additional keyword arguments.

        Returns:
            input_data: A dictionary with the following key if behavior is return:
                - "valid" (bool): boolean indicating if the value is valid.
                - "content" (Any): passed value if everything is correct.
            bool

        Raises:
            ValueError: If the value is not valid and behavior equal raise type.
        """
        config = ensure_config(config)
        self.run_on_node_execute_run(config.callbacks, **kwargs)
        try:
            self.validate(input_data["content"])
        except Exception as error:
            if self.behavior == Behavior.RETURN:
                return {"valid": False, "content": input_data["content"]}
            raise ValueError(str(error))
        return {"valid": True, "content": input_data["content"]}

    @abstractmethod
    def validate(self, content):
        pass
