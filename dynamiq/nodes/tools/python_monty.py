import ast
import asyncio
from typing import Any, ClassVar, Literal

from dynamiq.nodes import Node, NodeGroup
from dynamiq.nodes.agents.exceptions import ToolExecutionException
from dynamiq.nodes.node import ensure_config
from dynamiq.nodes.tools.python import PythonInputSchema
from dynamiq.nodes.types import ActionType
from dynamiq.runnables import RunnableConfig
from dynamiq.utils.logger import logger

_INPUTS_VAR = "dynamiq_monty_inputs_"
_TYPE_CHECK_STUBS = f"{_INPUTS_VAR}: dict = {{}}\n"


def _run_contains_async_def(code: str) -> bool:
    """Return True if user code defines `async def run(...)` at module scope."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False
    for node in tree.body:
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "run":
            return True
    return False


def _wrap_user_code(code: str, use_multiple_params: bool) -> str:
    """Append a trailing call to `run(...)` so Monty returns its value."""
    is_async = _run_contains_async_def(code)
    call_prefix = "await " if is_async else ""
    if use_multiple_params:
        trailer = f"{call_prefix}run(**{_INPUTS_VAR})"
    else:
        trailer = f"{call_prefix}run({_INPUTS_VAR})"
    return f"{code}\n\n{trailer}\n"


class PythonMonty(Node):
    """
    Node for executing Python code using the Monty interpreter (a Rust-based minimal Python interpreter).

    Attributes:
        code (str): The Python code to execute. Must define a `run` function as the entry point.
        use_multiple_params (bool): If True, the input dict will be unpacked into multiple parameters
            when calling `run`. Otherwise, the entire input dict will be passed as a single argument.
        type_check (bool): If True, Monty will perform type checking based on type annotations in the user code.
            This can help catch errors but may increase execution time.
        input_schema (type): A Pydantic model class that defines the expected input schema for the node.
        is_files_allowed (bool): Whether to allow file inputs. Monty does not support file I/O,
            so this should generally be False.
    """

    group: Literal[NodeGroup.TOOLS] = NodeGroup.TOOLS
    action_type: ActionType = ActionType.CODE_EXECUTION
    name: str = "python-tool-monty"
    description: str = (
        "Executes Python code inside a Rust-based minimal Python interpreter.\n"
        "Does not support third-party libraries (pandas, numpy, requests, pydantic, "
        "matplotlib, etc.)"
    )
    code: str
    use_multiple_params: bool = False
    type_check: bool = False
    input_schema: ClassVar[type[PythonInputSchema]] = PythonInputSchema
    is_files_allowed: bool = False

    def execute(self, input_data: PythonInputSchema, config: RunnableConfig = None, **kwargs) -> Any:
        """
        Execute the configured Python code synchronously via the Monty interpreter.

        Args:
            input_data (PythonInputSchema): Inputs forwarded to the user-defined ``run`` function.
            config (RunnableConfig, optional): Execution configuration, including callbacks.
            **kwargs: Additional arguments passed to the execution context.

        Returns:
            Any: A dictionary with a ``content`` key holding the ``run`` return value.

        Raises:
            ToolExecutionException: If ``pydantic_monty`` is unavailable or the user code fails.
        """
        return asyncio.run(self.execute_async(input_data, config, **kwargs))

    async def execute_async(self, input_data: PythonInputSchema, config: RunnableConfig = None, **kwargs) -> Any:
        """
        Execute the configured Python code asynchronously via the Monty interpreter.

        Args:
            input_data (PythonInputSchema): Inputs forwarded to the user-defined ``run`` function.
            config (RunnableConfig, optional): Execution configuration, including callbacks.
            **kwargs: Additional arguments passed to the execution context.

        Returns:
            Any: A dictionary with a ``content`` key holding the ``run`` return value.

        Raises:
            ToolExecutionException: If ``pydantic_monty`` is unavailable or the user code fails.
        """
        logger.info(
            f"Tool {self.name} - {self.id}: started with INPUT DATA:\n"
            f"{input_data.model_dump() if hasattr(input_data, 'model_dump') else input_data}"
        )
        config = ensure_config(config)
        self.run_on_node_execute_run(config.callbacks, **kwargs)

        try:
            import pydantic_monty
        except ImportError as e:
            raise ToolExecutionException(
                "PythonMonty requires the 'pydantic-monty' package. " "Install with: poetry install --extras monty",
                recoverable=False,
            ) from e

        inputs_dict = dict(input_data)
        wrapped_code = _wrap_user_code(self.code, self.use_multiple_params)

        try:
            monty = pydantic_monty.Monty(
                wrapped_code,
                inputs=[_INPUTS_VAR],
                script_name="python_monty.py",
                type_check=self.type_check,
                type_check_stubs=_TYPE_CHECK_STUBS if self.type_check else None,
            )
            result = await monty.run_async(inputs={_INPUTS_VAR: inputs_dict})
        except Exception as e:
            error_msg = f"Code execution error: {str(e)}"
            logger.error(error_msg)
            raise ToolExecutionException(error_msg, recoverable=True) from e

        logger.info(f"Tool {self.name} - {self.id}: finished with RESULT:\n{str(result)[:200]}...")
        return self._format_result(result)

    def _format_result(self, result: Any) -> Any:
        if isinstance(result, dict) and "content" in result:
            if self.is_optimized_for_agents:
                return {**result, "content": str(result["content"])}
            return result
        if self.is_optimized_for_agents:
            return {"content": str(result)}
        return {"content": result}
