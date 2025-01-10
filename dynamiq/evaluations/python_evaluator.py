import logging
from functools import cached_property
from typing import Any

from pydantic import BaseModel, Field, PrivateAttr, model_validator

from dynamiq.nodes.tools.python import Python as PythonNode

logger = logging.getLogger(__name__)


class PythonMetricSingleInput(BaseModel):
    """
    Input model for a single data point to the custom Python metric.

    This allows the user-defined Python code to access input_data arbitrarily.
    """

    data: dict[str, Any] = Field(default_factory=dict)


class PythonMetricMultipleInput(BaseModel):
    """
    Input model for multiple data points to the custom Python metric.
    """

    data_list: list[dict[str, Any]]

    @model_validator(mode="after")
    def validate_non_empty(self):
        if len(self.data_list) == 0:
            raise ValueError("data_list cannot be empty.")
        return self


class PythonMetricSingleOutput(BaseModel):
    """
    Output model for a single data point metric run.

    Attributes:
        score (float): A numeric metric score for the single input.
    """

    score: float


class PythonMetricMultipleOutput(BaseModel):
    """
    Output model for multiple data points metric run.

    Attributes:
        scores (List[float]): A list of metric scores for each input in data_list.
    """

    scores: list[float]


class PythonMetric(BaseModel):
    """
    A custom Python metric that executes user-defined Python code to compute a score.

    The Python code must define a 'run(input_data: dict)' function that returns
    an integer, float, or something convertible to float.

    Example usage of user-defined code in code_str:
        def run(input_data):
            # Retrieve arbitrary values from input_data
            answer = input_data.get("answer")
            expected = input_data.get("expected")
            return 1.0 if answer == expected else 0.0
    """

    name: str = "python_metric"
    code_str: str

    # Private attribute to hold the PythonNode
    _python_node: PythonNode = PrivateAttr()

    class Config:
        arbitrary_types_allowed = True

    @cached_property
    def type(self) -> str:
        """
        Returns a string indicating the fully qualified name of this class.
        """
        return f"{self.__module__}.{self.__class__.__name__}"

    def __init__(self, **data):
        """
        Custom constructor to create the PythonNode from the provided Python code.
        """
        super().__init__(**data)
        self._python_node = PythonNode(code=self.code_str)

    def run_single(self, data: dict[str, Any]) -> float:
        """
        Run the metric on a single input_data dictionary.

        Args:
            data (Dict[str, Any]): Arbitrary dictionary passed to the user code.

        Returns:
            float: Metric score for this single data point.
        """
        input_data = PythonMetricSingleInput(data=data)
        # Pass the dictionary to the PythonNode
        output = self._python_node.run(input_data=input_data.data)
        # The PythonNode places the return value in output.output['content']
        metric_value = output.output["content"]
        return float(metric_value)

    def run_many(self, data_list: list[dict[str, Any]]) -> list[float]:
        """
        Run the metric on multiple input data dictionaries in sequence.

        Args:
            data_list (List[Dict[str, Any]]): A list of dictionaries.

        Returns:
            List[float]: A list of metric scores, one per dictionary.
        """
        multiple_input = PythonMetricMultipleInput(data_list=data_list)
        scores = []
        for single_dict in multiple_input.data_list:
            score = self.run_single(single_dict)
            scores.append(score)
        return scores
