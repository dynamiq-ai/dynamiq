from typing import Any, Literal

import sympy as sp
from pydantic import ConfigDict

from dynamiq.nodes import NodeGroup
from dynamiq.nodes.node import Node, ensure_config
from dynamiq.runnables import RunnableConfig
from dynamiq.utils.logger import logger


class CalculatorTool(Node):
    """
    A tool to evaluate mathematical expressions.

    Attributes:
        group (Literal[NodeGroup.TOOLS]): The group of the node.
        name (str): The name of the tool.
        description (str): The description of the tool.
    """

    group: Literal[NodeGroup.TOOLS] = NodeGroup.TOOLS
    name: str = "Calculator"
    description: str = (
        "Tool to evaluate mathematical expressions. Provide a raw string of the math operation to parse and evaluate."
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def _evaluate_expression(self, expression: str) -> Any:
        """
        Parse and evaluate the given mathematical expression.

        Args:
            expression (str): The mathematical expression to evaluate.

        Returns:
            Any: The result of the evaluated expression or an error message if evaluation fails.
        """
        try:
            result = sp.sympify(expression)
            return result
        except Exception as e:
            return str(e)

    def execute(self, input_data: dict[str, Any], config: RunnableConfig = None, **kwargs) -> dict[str, Any]:
        """
        Execute the tool with the provided input data and configuration.

        Args:
            input_data (dict[str, Any]): The input data containing the expression to evaluate.
            config (RunnableConfig, optional): The configuration for the runnable. Defaults to None.

        Returns:
            dict[str, Any]: The result of the evaluation.
        """
        logger.debug(f"Tool {self.name} - {self.id}: started with input data {input_data}")

        config = ensure_config(config)
        self.run_on_node_execute_run(config.callbacks, **kwargs)

        expression = input_data.get("input", "")
        result = self._evaluate_expression(expression)

        logger.debug(f"Tool {self.name} - {self.id}: finished with result {result}")
        return {"content": result}


if __name__ == "__main__":
    calculator_tool = CalculatorTool()
    result = calculator_tool.execute(
        {
            "input": "1 + 2",
        }
    )
    print(result)
