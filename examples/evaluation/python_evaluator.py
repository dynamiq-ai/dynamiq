import logging
from functools import cached_property
from typing import Any

from pydantic import BaseModel, PrivateAttr, model_validator

from dynamiq.nodes.tools.python import Python as PythonNode

logger = logging.getLogger(__name__)


class PythonMetricInput(BaseModel):
    """
    Input model for the custom Python metric.

    Attributes:
        answers (List[Any]): List of answer values (e.g., predictions).
        expected_answers (List[Any]): List of expected (ground-truth) values.
    """

    answers: list[Any]
    expected_answers: list[Any]

    @model_validator(mode="after")
    def check_equal_length(self):
        if len(self.answers) != len(self.expected_answers):
            raise ValueError("answers and expected_answers must have the same length.")
        return self


class PythonMetricOutput(BaseModel):
    """
    Output model for the custom Python metric.

    Attributes:
        scores (List[float]): A numeric score for each answer/expected_answer pair.
    """

    scores: list[float]


class PythonMetric(BaseModel):
    """
    A custom Python metric that executes user-defined Python code to compute a score.

    Attributes:
        name (str): The metric name.
        code_str (str): The Python code to execute. Must define a `run(input_data)` function
                        that returns an integer or float score.
    """

    name: str = "python_metric"
    code_str: str

    # A private attribute to hold the PythonNode object
    _python_node: PythonNode = PrivateAttr()

    class Config:
        arbitrary_types_allowed = True  # So we can store PythonNode

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

    def run(self, answers: list[Any], expected_answers: list[Any]) -> list[float]:
        """
        Executes the user-defined Python code for each pair of answer/expected_answer.

        Args:
            answers (List[Any]): List of actual answers (e.g., model predictions).
            expected_answers (List[Any]): List of ground-truth or target values.

        Returns:
            List[float]: List of metric scores.
        """
        # Validate and structure the inputs
        input_data = PythonMetricInput(answers=answers, expected_answers=expected_answers)

        scores = []
        # For each pair, call the user-defined Python code
        for ans, exp_ans in zip(input_data.answers, input_data.expected_answers):
            output = self._python_node.run(
                input_data={
                    "answer": ans,
                    "expected_answer": exp_ans,
                }
            )
            # According to your code snippet, the metric score is in output.output['content']
            metric_value = output.output["content"]
            scores.append(float(metric_value))

        output_data = PythonMetricOutput(scores=scores)
        return output_data.scores
