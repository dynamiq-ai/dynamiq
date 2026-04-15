import ast
import asyncio
from typing import Any, ClassVar, Literal

from pydantic import ConfigDict

from dynamiq.nodes import Node, NodeGroup
from dynamiq.nodes.agents.exceptions import ToolExecutionException
from dynamiq.nodes.node import ensure_config
from dynamiq.nodes.tools.python import PythonInputSchema
from dynamiq.nodes.types import ActionType
from dynamiq.runnables import RunnableConfig
from dynamiq.utils.logger import logger

_INPUTS_VAR = "__dynamiq_inputs__"


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
    """Node executing developer-authored Python code in a Monty sandbox."""

    group: Literal[NodeGroup.TOOLS] = NodeGroup.TOOLS
    action_type: ActionType = ActionType.CODE_EXECUTION
    name: str = "python-tool-monty"
    description: str = (
        "Executes developer-authored Python code in a Monty sandbox (Rust-based "
        "minimal Python interpreter). More isolated than the RestrictedPython "
        "backend: no filesystem, no network, no env access. Supported stdlib: "
        "sys, os, typing, asyncio, re, datetime, json. No third-party libraries. "
        "Class definitions and match statements are not yet supported by Monty."
    )
    code: str
    use_multiple_params: bool = False
    type_check: bool = False
    input_schema: ClassVar[type[PythonInputSchema]] = PythonInputSchema
    is_files_allowed: bool = False

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def execute(self, input_data: PythonInputSchema, config: RunnableConfig = None, **kwargs) -> Any:
        return asyncio.run(self.execute_async(input_data, config, **kwargs))

    async def execute_async(
        self, input_data: PythonInputSchema, config: RunnableConfig = None, **kwargs
    ) -> Any:
        logger.info(f"Tool {self.name} - {self.id}: started")
        config = ensure_config(config)
        self.run_on_node_execute_run(config.callbacks, **kwargs)

        try:
            import pydantic_monty
        except ImportError as e:
            raise ToolExecutionException(
                "PythonMonty requires the 'pydantic-monty' package. "
                "Install with: poetry install --extras monty",
                recoverable=False,
            ) from e

        inputs_dict = dict(input_data) if not isinstance(input_data, dict) else dict(input_data)
        wrapped_code = _wrap_user_code(self.code, self.use_multiple_params)

        try:
            m = pydantic_monty.Monty(
                wrapped_code,
                inputs=[_INPUTS_VAR],
                script_name="python_monty.py",
                type_check=self.type_check,
            )
            result = await m.run_async(inputs={_INPUTS_VAR: inputs_dict})
        except ToolExecutionException:
            raise
        except Exception as e:
            error_msg = f"Code execution error: {str(e)}"
            logger.error(error_msg)
            raise ToolExecutionException(error_msg, recoverable=True) from e

        return self._format_result(result)

    def _format_result(self, result: Any) -> Any:
        if isinstance(result, dict) and "content" in result:
            if self.is_optimized_for_agents:
                return {**result, "content": str(result["content"])}
            return result
        if self.is_optimized_for_agents:
            return {"content": str(result)}
        return {"content": result}
