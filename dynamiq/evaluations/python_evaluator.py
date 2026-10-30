import re
from typing import Any

import pydantic_monty
from pydantic import BaseModel, Field, model_validator

from dynamiq.evaluations import BaseEvaluator
from dynamiq.utils.logger import logger
from dynamiq.utils.monty import get_monty_pool

DEFAULT_TIME_LIMIT_SECONDS = 5.0
DEFAULT_MAX_MEMORY_BYTES = 64 * 1024 * 1024

# Monty returns the value of a snippet's trailing expression, so evaluate() is called from a trailer that
# unpacks this global.
_INPUTS_VAR = "dynamiq_evaluator_inputs_"
_SCRIPT_NAME = "python_metric.py"
# Reports whether the code defines an evaluate() function, without a NameError when it does not. Monty has no
# callable() builtin, so the check compares types with a function of its own.
_DEFINES_EVALUATE_CHECK = """
def dynamiq_function_probe_():
    pass

dynamiq_evaluate_defined_ = True
try:
    evaluate
except NameError:
    dynamiq_evaluate_defined_ = False
dynamiq_evaluate_defined_ and type(evaluate) == type(dynamiq_function_probe_)
"""


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
    Evaluator that runs custom user-defined code in the Monty interpreter.

    The user code must define a function named "evaluate" with any parameters.
    This evaluator uses regex to extract the function parameters and then calls it
    with input values passed as keyword arguments. It validates that the provided
    input contains all required keys, and does not include unexpected keys.

    Each call runs in a fresh Monty worker subprocess, which has no access to the host's environment
    variables, files or network, and is stopped once it exceeds timeout_seconds or max_memory_bytes.
    Monty supports a subset of Python and part of the standard library (json, re, math, datetime,
    collections, itertools, functools and others), but no third-party packages.

    Example user code:
        def evaluate(answer: int, expected: int = 42) -> float:
            return 1.0 if answer == expected else 0.0

    Attributes:
        name (str): Evaluator name; defaults to "python_metric".
        code (str): The user-defined Python code as a string.
        timeout_seconds (float): Longest a single run of the code may take.
        max_memory_bytes (int): Most heap memory a single run of the code may use.
    """
    name: str = "python_metric"
    code: str = Field(..., description="User-defined Python code as a string.")
    timeout_seconds: float = Field(default=DEFAULT_TIME_LIMIT_SECONDS, gt=0)
    max_memory_bytes: int = Field(default=DEFAULT_MAX_MEMORY_BYTES, gt=0)

    def __init__(self, **data: Any):
        super().__init__(**data)
        self._compile_user_code()

    def _compile_user_code(self) -> None:
        """
        Run the user code once in Monty to check that it compiles and defines a
        callable "evaluate", then parse the function's parameter names.
        """
        try:
            defines_evaluate = self._run_in_monty(f"{self.code}\n{_DEFINES_EVALUATE_CHECK}")
        except Exception as compile_error:
            logger.error(f"User code compilation error: {compile_error}")
            raise ValueError(f"User code compilation error: {compile_error}") from compile_error

        if defines_evaluate is not True:
            raise ValueError("User code must define a callable function 'evaluate'.")

        (required_params, all_params) = self._extract_function_parameters(self.code)
        self._required_params: set[str] = set(required_params)
        self._all_params: set[str] = set(all_params)
        logger.debug(f"Extracted required parameters: {self._required_params}")
        logger.debug(f"Extracted all parameters: {self._all_params}")

    def _run_in_monty(self, code: str, inputs: dict[str, Any] | None = None) -> Any:
        """Run code in a fresh Monty worker under this evaluator's limits and return its trailing expression."""
        limits = pydantic_monty.ResourceLimits(
            max_feed_duration_secs=self.timeout_seconds, max_memory=self.max_memory_bytes
        )
        with get_monty_pool().checkout(script_name=_SCRIPT_NAME, limits=limits) as session:
            # print() output is dropped rather than written to the host's stdout.
            return session.feed_run(code, inputs=inputs, print_callback=pydantic_monty.CollectString())

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
            result = self._run_in_monty(
                f"{self.code}\n\nevaluate(**{_INPUTS_VAR})\n", inputs={_INPUTS_VAR: input_model.data}
            )
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
