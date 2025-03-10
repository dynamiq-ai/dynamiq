import importlib
import io
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict
from RestrictedPython import compile_restricted, safe_builtins, utility_builtins
from RestrictedPython.Eval import default_guarded_getattr, default_guarded_getitem, default_guarded_getiter
from RestrictedPython.Guards import guarded_unpack_sequence

from dynamiq.nodes import Node, NodeGroup
from dynamiq.nodes.agents.exceptions import ToolExecutionException
from dynamiq.nodes.node import ensure_config
from dynamiq.runnables import RunnableConfig
from dynamiq.utils import format_value
from dynamiq.utils.logger import logger

ALLOWED_MODULES = [
    "base64",
    "collections",
    "copy",
    "cmath",
    "csv",
    "datetime",
    "dynamiq",
    "functools",
    "itertools",
    "json",
    "math",
    "operator",
    "pydantic",
    "random",
    "re",
    "requests",
    "statistics",
    "time",
    "typing",
    "urllib",
    "uuid",
]


def restricted_import(name: str, globals=None, locals=None, fromlist=(), level=0) -> Any:
    """
    Restricted import function to allow importing only modules in ALLOWED_MODULES.
    """
    root_module_name = name.split(".")[0]
    if root_module_name not in ALLOWED_MODULES:
        logger.warning(f"Import of '{root_module_name}' is not allowed")
        raise ImportError(f"Import of '{root_module_name}' is not allowed")
    try:
        module = importlib.import_module(name)
        logger.debug(f"Successfully imported {name}")
        return module
    except ImportError as e:
        logger.error(f"Failed to import {name}: {str(e)}")
        raise


def get_restricted_globals() -> dict:
    """
    Return globals dict configured for restricted code execution.
    """
    return {
        "__builtins__": {
            **safe_builtins,
            **utility_builtins,
            "abs": abs,
            "all": all,
            "any": any,
            "bin": bin,
            "bool": bool,
            "chr": chr,
            "complex": complex,
            "dict": dict,
            "divmod": divmod,
            "enumerate": enumerate,
            "filter": filter,
            "float": float,
            "format": format,
            "frozenset": frozenset,
            "hex": hex,
            "int": int,
            "isinstance": isinstance,
            "issubclass": issubclass,
            "len": len,
            "list": list,
            "map": map,
            "max": max,
            "min": min,
            "next": next,
            "oct": oct,
            "ord": ord,
            "pow": pow,
            "range": range,
            "reversed": reversed,
            "round": round,
            "set": set,
            "slice": slice,
            "sorted": sorted,
            "str": str,
            "sum": sum,
            "super": super,
            "tuple": tuple,
            "type": type,
            "zip": zip,
            "__import__": restricted_import,
            "_getattr_": default_guarded_getattr,
            "_getitem_": default_guarded_getitem,
            "_getiter_": default_guarded_getiter,
        },
        "_getattr_": default_guarded_getattr,
        "_unpack_sequence_": guarded_unpack_sequence,
    }


def compile_and_execute(code: str, restricted_globals: dict) -> dict:
    """
    Compile the code using RestrictedPython and execute it in restricted_globals.
    Returns the updated restricted_globals.
    """
    try:
        byte_code = compile_restricted(code, "<inline>", "exec")
        exec(byte_code, restricted_globals)  # nosec
        return restricted_globals
    except Exception as e:
        logger.error(f"Error during restricted execution: {e}")
        raise


class PythonInputSchema(BaseModel):
    model_config = ConfigDict(extra="allow", strict=True, arbitrary_types_allowed=True)

    def to_dict(self, **kwargs) -> dict:
        return {field: format_value(value, **kwargs)[0] for field, value in self.model_extra.items()}


class Python(Node):
    """
    Node for executing Python code in a secure sandbox.
    Attributes:
      group (Literal[NodeGroup.TOOLS]): node group.
      name (str): node name.
      description (str): node description.
      code (str): Python code to execute.
    """
    group: Literal[NodeGroup.TOOLS] = NodeGroup.TOOLS
    name: str = "Python Code Executor Tool"
    description: str = (
        "The tool that executes Python code in a secure sandbox environment. "
        "All arguments are passed as a dictionary to the 'run' main function."
    )
    code: str
    input_schema: ClassVar[type[PythonInputSchema]] = PythonInputSchema

    def execute(self, input_data: PythonInputSchema, config: RunnableConfig = None, **kwargs) -> Any:
        """
        Execute the Python code.
        Args:
            input_data (PythonInputSchema): Input data for the code.
            config (RunnableConfig, optional): Execution configuration.
            **kwargs: Additional keyword arguments.
        Returns:
            Any: The result from code execution.
        """
        logger.info(f"Tool {self.name} - {self.id}: started with INPUT DATA:\n" f"{input_data.model_dump()}")
        config = ensure_config(config)
        self.run_on_node_execute_run(config.callbacks, **kwargs)
        stdout = io.StringIO()

        def safe_print(*args, **kwargs):
            print(*args, file=stdout, **kwargs)

        def guarded_write(obj, value=None):
            return value if value is not None else obj

        restricted_globals = get_restricted_globals()
        restricted_globals.update(
            {
                "_inplacevar_": self._inplacevar,
                "_iter_unpack_sequence_": guarded_unpack_sequence,
                "safe_print": safe_print,
                "_write_": guarded_write,
            }
        )

        try:
            restricted_globals = compile_and_execute(self.code, restricted_globals)
            if "run" not in restricted_globals:
                raise ValueError("The 'run' function is not defined in the provided code.")
            result = restricted_globals["run"](dict(input_data))
            if self.is_optimized_for_agents:
                result = str(result)
        except Exception as e:
            error_msg = f"Code execution error: {str(e)}"
            logger.error(error_msg)
            raise ToolExecutionException(error_msg, recoverable=True)
        logger.info(f"Tool {self.name} - {self.id}: finished with RESULT:\n" f"{str(result)[:200]}...")
        return {"content": result}

    @staticmethod
    def _inplacevar(op: str, x: Any, y: Any) -> Any:
        """
        Perform an in-place operation.
        Args:
            op (str): Operation (e.g., '+=', '-=')
            x (Any): First operand.
            y (Any): Second operand.
        Returns:
            Any: Result of the operation.
        """
        operators = {
            "+=": lambda a, b: a + b,
            "-=": lambda a, b: a - b,
            "*=": lambda a, b: a * b,
            "/=": lambda a, b: a / b,
            "//=": lambda a, b: a // b,
            "%=": lambda a, b: a % b,
            "**=": lambda a, b: a**b,
            "<<=": lambda a, b: a << b,
            ">>=": lambda a, b: a >> b,
            "&=": lambda a, b: a & b,
            "^=": lambda a, b: a ^ b,
            "|=": lambda a, b: a | b,
        }
        if op in operators:
            return operators[op](x, y)
        raise ValueError(f"Unsupported in-place operation: {op}")
