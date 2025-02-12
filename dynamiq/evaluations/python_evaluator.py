import re
from typing import Any

from pydantic import BaseModel, Field, model_validator

from dynamiq.evaluations import BaseEvaluator
from dynamiq.nodes.tools.python import compile_and_execute, get_restricted_globals
from dynamiq.utils.logger import logger


class PythonEvaluatorSingleInput(BaseModel):
    """
    Model for a single evaluation input.

    Attributes:
        data (dict[str, Any]): A dictionary with input parameters.
    """
    data: dict[str, Any] = Field(default_factory=dict)


class PythonEvaluatorInput(BaseModel):
    """
    Model for batch evaluation input.

    Attributes:
        data_list (list[dict[str, Any]]): A list of input dictionaries.
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
        if not self.data_list:
            raise ValueError("The data_list must not be empty.")
        return self


class PythonEvaluatorSingleOutput(BaseModel):
    """
    Model for a single evaluation output.

    Attributes:
        score (float): Numeric metric value for the input.
    """
    score: float


class PythonEvaluatorOutput(BaseModel):
    """
    Model for batch evaluation output.

    Attributes:
        scores (list[float]): A list of computed metric scores.
    """
    scores: list[float]


class PythonEvaluator(BaseEvaluator):
    """
    Evaluator that executes custom user-defined code inside a safe sandbox.

    The user code must define a function named "evaluate" with any parameters.
    This evaluator uses regex to extract the function parameters and then calls it
    with input values passed as keyword arguments. It validates that the provided
    input contains all required keys, and does not include unexpected keys.

    Example user code:
        def evaluate(answer: int, expected: int = 42) -> float:
            return 1.0 if answer == expected else 0.0

    Attributes:
        name (str): Evaluator name; defaults to "python_metric".
        code (str): The user-defined Python code as a string.
    """
    name: str = "python_metric"
    code: str = Field(..., description="User-defined Python code as a string.")

    def __init__(self, **data: Any):
        super().__init__(**data)
        self._compile_user_code()

    def _compile_user_code(self) -> None:
        """
        Compile the user code using restricted execution. Save the compiled
        globals and parse the "evaluate" function's parameter names.
        """
        self._compiled_globals: dict = get_restricted_globals()
        try:
            compile_and_execute(self.code, self._compiled_globals)
        except Exception as compile_error:
            logger.error(f"User code compilation error: {compile_error}")
            raise ValueError(f"User code compilation error: {compile_error}") from compile_error

        if "evaluate" not in self._compiled_globals or not callable(self._compiled_globals["evaluate"]):
            raise ValueError("User code must define a callable function 'evaluate'.")

        (required_params, all_params) = self._extract_function_parameters(self.code)
        self._required_params: set[str] = set(required_params)
        self._all_params: set[str] = set(all_params)
        logger.debug(f"Extracted required parameters: {self._required_params}")
        logger.debug(f"Extracted all parameters: {self._all_params}")

    @staticmethod
    def _extract_function_parameters(code_str: str) -> tuple[list[str], list[str]]:
        """
        Extract parameter names from a function named "evaluate" in the given code string.

        Returns a tuple:
            (required_params, all_params)
        where 'required_params' is a list of parameter names without default values,
        and 'all_params' is a list of all parameter names declared in the function.

        Uses a regex pattern to locate the function signature.
        """
        function_pattern = re.compile(
            r"^\s*def\s+evaluate\s*\((.*?)\)\s*(?:->\s*.*?)?\s*:",
            re.MULTILINE | re.DOTALL,
        )
        matches = function_pattern.findall(code_str)
        if not matches:
            raise ValueError("No function named 'evaluate' found in the code.")
        if len(matches) > 1:
            raise ValueError("Multiple definitions of function 'evaluate' were found.")

        params_section = matches[0].strip()
        if not params_section:
            return ([], [])

        raw_params = [param.strip() for param in params_section.split(",") if param.strip()]
        required_params: list[str] = []
        all_params: list[str] = []
        for param in raw_params:
            has_default = "=" in param
            param_name = re.split(r"[:=]", param, maxsplit=1)[0].strip()
            if param_name:
                all_params.append(param_name)
                if not has_default:
                    required_params.append(param_name)
        return (required_params, all_params)

    def _validate_input_keys(self, provided_keys: set[str]) -> None:
        """
        Validate that provided_keys contains all required keys and no unexpected keys.

        Args:
            provided_keys (Set[str]): Keys present in the user input.

        Raises:
            ValueError: If any required keys are missing or if unexpected keys are found.
        """
        missing_keys = self._required_params - provided_keys
        if missing_keys:
            raise ValueError(f"Missing required keys: {sorted(missing_keys)}")
        extra_keys = provided_keys - self._all_params
        if extra_keys:
            raise ValueError(f"Unexpected keys provided: {sorted(extra_keys)}")

    def run_single(self, input_data: dict[str, Any]) -> float:
        """
        Evaluate the metric for a single input dictionary.

        The provided input_data must contain all required keys; optional keys may be omitted.

        Args:
            input_data (dict[str, Any]): Input parameters as a dictionary.

        Returns:
            float: The computed metric score.

        Raises:
            ValueError: If key validation fails, or the function returns no value
                        or a non-numeric value.
        """
        input_model = PythonEvaluatorSingleInput(data=input_data)
        provided_keys = set(input_model.data.keys())
        self._validate_input_keys(provided_keys)

        try:
            result = self._compiled_globals["evaluate"](**input_model.data)
        except Exception as func_error:
            logger.error(f"Error during evaluation: {func_error}")
            raise ValueError(f"Error during evaluation: {func_error}") from func_error

        if result is None:
            raise ValueError("User-defined 'evaluate' function returned no value.")
        try:
            score = round(float(result), 2)
        except (TypeError, ValueError) as conv_err:
            raise ValueError("User-defined function returned a non-numeric value.") from conv_err

        logger.debug(f"Computed score: {score} for input: {input_data}")
        return score

    def run(self, input_data_list: list[dict[str, Any]]) -> list[float]:
        """
        Evaluate the metric for a list of input dictionaries sequentially.

        Args:
            input_data_list (list[dict[str, Any]]): A list of input dictionaries.

        Returns:
            list[float]: A list of computed metric scores.

        Raises:
            ValueError: If input_data_list is empty or any evaluation fails.
        """
        batch_input = PythonEvaluatorInput(data_list=input_data_list)
        scores: list[float] = []
        for idx, data_dict in enumerate(batch_input.data_list, start=1):
            try:
                score = self.run_single(data_dict)
                scores.append(score)
                logger.info(f"Processed input {idx} with score = {score}")
            except Exception as error:
                logger.error(f"Failed processing input {idx}: {error}")
                raise ValueError(f"Failed processing input {idx}: {error}") from error

        output_model = PythonEvaluatorOutput(scores=scores)
        return output_model.scores
