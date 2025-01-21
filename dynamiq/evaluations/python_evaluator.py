from typing import Any

from pydantic import BaseModel, Field, PrivateAttr, model_validator

from dynamiq.evaluations import BaseEvaluator
from dynamiq.nodes.tools.python import Python as PythonNode
from dynamiq.utils.logger import logger


class PythonEvaluatorSingleInput(BaseModel):
    """
    Input model for a single data point to the custom Python evaluator.

    Allows the user-defined Python code to access input data arbitrarily.

    Attributes:
        data (dict[str, Any]): Arbitrary dictionary containing input data for evaluation.
    """

    data: dict[str, Any] = Field(default_factory=dict)


class PythonEvaluatorInput(BaseModel):
    """
    Input model for multiple data points to the custom Python evaluator.

    Attributes:
        data_list (list[dict[str, Any]]): A list of dictionaries containing input data points.
    """

    data_list: list[dict[str, Any]]

    @model_validator(mode="after")
    def validate_non_empty(self) -> "PythonEvaluatorInput":
        """
        Validate that the data_list is not empty.

        Raises:
            ValueError: If data_list is empty.

        Returns:
            PythonEvaluatorInput: The validated instance.
        """
        if len(self.data_list) == 0:
            raise ValueError("data_list cannot be empty.")
        return self


class PythonEvaluatorSingleOutput(BaseModel):
    """
    Output model for a single data point metric run.

    Attributes:
        score (float): A numeric metric score for the single input.
    """

    score: float


class PythonEvaluatorOutput(BaseModel):
    """
    Output model for multiple data points metric run.

    Attributes:
        scores (list[float]): A list of metric scores for each input in data_list.
    """

    scores: list[float]


class PythonEvaluator(BaseEvaluator):
    """
    A custom Python evaluator that executes user-defined Python code to compute scores.

    The Python code must define a 'run(input_data: dict)' function that returns
    an integer, float, or a value convertible to float.

    Example usage of user-defined code in code_str:
        def run(input_data):
            # Retrieve arbitrary values from input_data
            answer = input_data.get("answer")
            expected = input_data.get("expected")
            return 1.0 if answer == expected else 0.0

    Attributes:
        name (str): Name of the evaluator. Defaults to "python_metric".
        code (str): User-defined Python code as a string.
    """

    name: str = "python_metric"
    code: str = Field(..., description="User-defined Python code as a string.")

    # Private attribute to hold the PythonNode instance
    _python_node: PythonNode = PrivateAttr()

    def __init__(self, **data):
        """
        Initialize the PythonEvaluator instance and set up the PythonNode with user-defined code.

        Args:
            **data: Arbitrary keyword arguments for the BaseModel.
        """
        super().__init__(**data)
        self._initialize_python_node()

    def _initialize_python_node(self) -> None:
        """
        Initialize the PythonNode with the provided user-defined Python code.

        Raises:
            ImportError: If the PythonNode fails to initialize due to invalid code.
        """
        try:
            self._python_node = PythonNode(code=self.code)
            logger.debug("PythonNode initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize PythonNode: {e}")
            raise ValueError(f"Failed to initialize PythonNode: {e}") from e

    def run_single(self, input_data: dict[str, Any]) -> float:
        """
        Run the evaluator on a single input_data dictionary.

        Args:
            input_data (dict[str, Any]): Arbitrary dictionary passed to the user code.

        Returns:
            float: Metric score for this single data point.

        Raises:
            ValueError: If the user-defined code does not return a numeric value.
        """
        single_input = PythonEvaluatorSingleInput(data=input_data)

        # Execute the user-defined Python code
        output = self._python_node.run(input_data=single_input.data)

        # The PythonNode places the return value in output.output['content']
        metric_value = output.output.get("content")
        if metric_value is None:
            raise ValueError("User-defined code did not return a value in 'content'.")

        try:
            metric_score = round(float(metric_value), 2)
        except (TypeError, ValueError):
            raise ValueError("User-defined code returned a non-numeric value.")

        logger.debug(f"Computed metric score: {metric_score} for input_data: {input_data}")
        return metric_score

    def run(self, input_data_list: list[dict[str, Any]]) -> list[float]:
        """
        Run the evaluator on multiple input data dictionaries sequentially.

        Args:
            input_data_list (list[dict[str, Any]]): A list of dictionaries containing input data points.

        Returns:
            list[float]: A list of metric scores, one per dictionary.

        Raises:
            ValueError: If input_data_list is empty or if any evaluation fails.
        """
        multiple_input = PythonEvaluatorInput(data_list=input_data_list)
        scores: list[float] = []

        for idx, single_dict in enumerate(multiple_input.data_list, start=1):
            try:
                score = self.run_single(single_dict)
                scores.append(score)
                logger.info(f"Processed pair {idx}: Score = {score}")
            except Exception as e:
                logger.error(f"Failed to process pair {idx}: {e}")
                raise ValueError(f"Failed to process pair {idx}: {e}") from e

        output_data = PythonEvaluatorOutput(scores=scores)
        return output_data.scores
