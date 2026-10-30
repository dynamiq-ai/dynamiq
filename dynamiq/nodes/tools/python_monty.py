import ast
import asyncio
import datetime
import pathlib
from typing import Any, ClassVar, Literal

import pydantic_monty

from dynamiq.nodes import Node, NodeGroup
from dynamiq.nodes.agents.exceptions import ToolExecutionException
from dynamiq.nodes.node import ensure_config
from dynamiq.nodes.tools.python import PythonInputSchema
from dynamiq.nodes.types import ActionType
from dynamiq.runnables import RunnableConfig
from dynamiq.types.cancellation import check_cancellation
from dynamiq.utils.logger import logger
from dynamiq.utils.monty import MONTY_OS_POLICY, get_monty_pool

_INPUTS_VAR = "dynamiq_monty_inputs_"
_TYPE_CHECK_STUBS = f"{_INPUTS_VAR}: dict = {{}}\n"
_SCRIPT_NAME = "python_monty.py"
# Applies when error_handling.timeout_seconds is unset, so code that never returns cannot hold a worker forever.
DEFAULT_TIME_LIMIT_SECONDS = 60.0
DEFAULT_MAX_MEMORY_BYTES = 256 * 1024 * 1024
# The most characters a run() result may take when formatted or serialized. A result can repeat references to the
# same object, so a few dozen objects can stand for gigabytes of text.
DEFAULT_MAX_OUTPUT_CHARS = 10 * 1024 * 1024

# Upper bounds of the repr() of fixed-size values.
_FIXED_SIZES: dict[type, int] = {
    type(None): 4,
    bool: 5,
    float: 24,
    complex: 52,
    type(Ellipsis): 8,
    datetime.date: 40,
    datetime.datetime: 96,
    datetime.time: 48,
    datetime.timedelta: 64,
    datetime.timezone: 96,
}
_SEQUENCE_TYPES = (list, tuple, set, frozenset)
# A reference back to an enclosing container prints as "[...]".
_CYCLE_SIZE = 5


def _output_size(value: Any, limit: int) -> int:
    """Return an upper bound of the characters repr() or JSON would produce for value.

    Every repeated reference counts in full, as formatting expands it, but each distinct object is visited once,
    so the walk stays as small as the object graph Monty returned. The walk is the host's own code, so neither
    names nor __repr__ methods the snippet defines change the result.

    Raises:
        ValueError: If the bound exceeds limit, or value holds a type a Monty result does not convert to.
    """
    sizes: dict[int, int] = {}
    # Containers whose children are being sized, with those children. Sized objects stay referenced in visited,
    # so no id() is reused by another object during the walk.
    pending: dict[int, list[Any]] = {}
    visited: list[Any] = []
    stack: list[tuple[Any, bool]] = [(value, False)]
    while stack:
        item, children_sized = stack.pop()
        key = id(item)
        if key in sizes:
            continue
        if children_sized:
            children = pending.pop(key)
            size = 2 + sum(sizes.get(id(child), _CYCLE_SIZE) + 2 for child in children)
            if type(item) is pydantic_monty.MontyClassProxy:
                size += 32
        elif (size := _leaf_size(item)) is None:
            if key in pending:  # a reference back to an enclosing container
                continue
            children = _children(item)
            pending[key] = children
            stack.append((item, True))
            stack.extend((child, False) for child in children if id(child) not in sizes)
            # Each pending reference prints at least one character, which bounds the walk by the limit.
            if len(sizes) + len(stack) > limit:
                raise ValueError(f"run() returned more than {limit} characters of output")
            continue
        if size > limit:
            raise ValueError(f"run() returned more than {limit} characters of output")
        sizes[key] = size
        visited.append(item)
    return sizes[id(value)]


def _leaf_size(value: Any) -> int | None:
    """Return the repr() size bound of a value that holds no other values, or None for a container."""
    kind = type(value)
    if kind is str:
        if value.isprintable():
            return len(value) + value.count("\\") + value.count("'") + value.count('"') + 2
        return 10 * len(value) + 2
    if kind is bytes or kind is bytearray:
        return 4 * len(value) + 14
    if kind is int:
        return value.bit_length() * 30103 // 100000 + 3
    if kind in _FIXED_SIZES:
        return _FIXED_SIZES[kind]
    if isinstance(value, pathlib.PurePath):
        return 2 * len(value.as_posix()) + 32
    if kind in _SEQUENCE_TYPES or kind is dict or kind is pydantic_monty.MontyClassProxy:
        return None
    raise ValueError(f"run() returned a {kind.__name__}, which PythonMonty cannot pass on")


def _children(value: Any) -> list[Any]:
    """Return the values a container holds, keys included, and a class instance's name and attributes."""
    if type(value) is dict:
        return [part for pair in value.items() for part in pair]
    if type(value) is pydantic_monty.MontyClassProxy:
        return [value.name, value.attributes]
    return list(value)


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

    Each run gets a fresh Monty worker subprocess, which has no access to the host's environment variables,
    files or network. The run is stopped after error_handling.timeout_seconds, or DEFAULT_TIME_LIMIT_SECONDS
    when that is unset, and once it uses more than DEFAULT_MAX_MEMORY_BYTES of heap. It fails when run() returns
    a value that would format to more than DEFAULT_MAX_OUTPUT_CHARS characters.

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
            ToolExecutionException: If the user code fails or exceeds its limits.
        """
        config = ensure_config(config)
        check_cancellation(config)
        self.run_on_node_execute_run(config.callbacks, **kwargs)
        return self._format_result(self._run_in_monty(dict(input_data)))

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
            ToolExecutionException: If the user code fails or exceeds its limits.
        """
        config = ensure_config(config)
        check_cancellation(config)
        self.run_on_node_execute_run(config.callbacks, **kwargs)
        # The pool blocks until the worker answers, so the call runs off the event loop.
        result = await asyncio.to_thread(self._run_in_monty, dict(input_data))
        return self._format_result(result)

    def _run_in_monty(self, inputs: dict[str, Any]) -> Any:
        """Run the user code with inputs in a fresh Monty worker and return the value of its run()."""
        limits = pydantic_monty.ResourceLimits(
            max_feed_duration_secs=self.error_handling.timeout_seconds or DEFAULT_TIME_LIMIT_SECONDS,
            max_memory=DEFAULT_MAX_MEMORY_BYTES,
        )
        try:
            with get_monty_pool().checkout(
                script_name=_SCRIPT_NAME,
                limits=limits,
                os_policy=MONTY_OS_POLICY,
                type_check=self.type_check,
                type_check_stubs=_TYPE_CHECK_STUBS if self.type_check else None,
            ) as session:
                result = session.feed_run(
                    _wrap_user_code(self.code, self.use_multiple_params), inputs={_INPUTS_VAR: inputs}
                )
            # Checked before anything formats or serializes the result, which would expand repeated references.
            _output_size(result, DEFAULT_MAX_OUTPUT_CHARS)
            return result
        except Exception as e:
            error_msg = f"Code execution error: {str(e)}"
            logger.error(error_msg)
            raise ToolExecutionException(error_msg, recoverable=True) from e

    def _format_result(self, result: Any) -> Any:
        if isinstance(result, dict) and "content" in result:
            if self.is_optimized_for_agents:
                return {**result, "content": str(result["content"])}
            return result
        if self.is_optimized_for_agents:
            return {"content": str(result)}
        return {"content": result}
