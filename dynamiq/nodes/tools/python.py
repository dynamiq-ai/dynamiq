import importlib
import io
from copy import deepcopy
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
    "io",
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
    "pandas",
    "numpy",
    "openpyxl"
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


def safe_hasattr(obj: Any, name: str) -> bool:
    """
    Safe version of hasattr that uses guarded getattr.
    """
    try:
        default_guarded_getattr(obj, name)
        return True
    except (AttributeError, TypeError):
        return False


def safe_getattr(obj: Any, name: str, default=None) -> Any:
    """
    Safe version of getattr that uses guarded getattr.
    """
    try:
        return default_guarded_getattr(obj, name)
    except (AttributeError, TypeError):
        if default is not None:
            return default
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
            "bytes": bytes,
            "callable": callable,
            "chr": chr,
            "complex": complex,
            "dict": dict,
            "dir": dir,
            "divmod": divmod,
            "enumerate": enumerate,
            "filter": filter,
            "float": float,
            "format": format,
            "frozenset": frozenset,
            "getattr": safe_getattr,
            "hasattr": safe_hasattr,
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
            "object": object,
            "oct": oct,
            "ord": ord,
            "pow": pow,
            "range": range,
            "reversed": reversed,
            "round": round,
            "set": set,
            "setattr": setattr,
            "slice": slice,
            "sorted": sorted,
            "str": str,
            "sum": sum,
            "super": super,
            "tuple": tuple,
            "type": type,
            "vars": vars,
            "zip": zip,
            "__import__": restricted_import,
            "_getattr_": default_guarded_getattr,
            "_getitem_": default_guarded_getitem,
            "_getiter_": default_guarded_getiter,
            "__metaclass__": type,
            "__name__": "__main__",
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
        return {field: format_value(value, **kwargs) for field, value in self.model_extra.items()}


class Python(Node):
    """
    Node for executing Python code in a secure sandbox.
    Attributes:
      group (Literal[NodeGroup.TOOLS]): node group.
      name (str): node name.
      description (str): node description.
      code (str): Python code to execute.
      use_multiple_params (bool): Determines how the input dictionary is passed to the Python function.
        -If True, the input dictionary is unpacked into multiple parameters.
        -If False, it is passed as a single dictionary with all parameters as keys.
    """
    group: Literal[NodeGroup.TOOLS] = NodeGroup.TOOLS
    name: str = "Python Code Executor Tool"
    description: str = """Executes Python code in a secure sandbox with restricted imports for calculations,
    data processing, and API interactions."""  # noqa: E501
    code: str
    use_multiple_params: bool = False
    input_schema: ClassVar[type[PythonInputSchema]] = PythonInputSchema
    is_files_allowed: bool = True

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
            result = (
                restricted_globals["run"](**dict(input_data))
                if self.use_multiple_params
                else restricted_globals["run"](dict(input_data))
            )
        except Exception as e:
            error_msg = f"Code execution error: {str(e)}"
            logger.error(error_msg)
            raise ToolExecutionException(error_msg, recoverable=True)

        logger.info(f"Tool {self.name} - {self.id}: finished with RESULT:\n" f"{str(result)[:200]}...")

        if isinstance(result, dict) and "content" in result:
            if self.is_optimized_for_agents:
                optimized_result = deepcopy(result)
                optimized_result["content"] = str(result["content"])
                return optimized_result
            else:
                return result
        else:
            if self.is_optimized_for_agents:
                return {"content": str(result)}
            else:
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
