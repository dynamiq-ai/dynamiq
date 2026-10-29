import ast
import builtins
import io
from copy import deepcopy
from types import ModuleType
from typing import Any, Callable, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field
from RestrictedPython import compile_restricted, safe_builtins, utility_builtins
from RestrictedPython.Eval import default_guarded_getattr, default_guarded_getitem, default_guarded_getiter
from RestrictedPython.Guards import guarded_unpack_sequence
from RestrictedPython.transformer import INSPECT_ATTRIBUTES, RestrictingNodeTransformer, copy_locations

from dynamiq.nodes import Node, NodeGroup
from dynamiq.nodes.agents.exceptions import ToolExecutionException
from dynamiq.nodes.node import ensure_config
from dynamiq.nodes.types import ActionType
from dynamiq.runnables import RunnableConfig
from dynamiq.types.cancellation import check_cancellation
from dynamiq.utils import format_value
from dynamiq.utils.lazy_imports import optional_dependency_error
from dynamiq.utils.logger import logger

# Modules exposed to restricted code by default: pure-Python standard library only. These are safe to
# offer to untrusted code because they grant no process, filesystem or native-code access.
DEFAULT_ALLOWED_MODULES = frozenset(
    {
        "base64",
        "collections",
        "copy",
        "cmath",
        "csv",
        "datetime",
        "decimal",
        "functools",
        "io",
        "itertools",
        "json",
        "math",
        "operator",
        "random",
        "re",
        "statistics",
        "string",
        "time",
        "typing",
        "uuid",
    }
)

# Heavier third-party and I/O libraries. They are not a security boundary: they expose native code and
# deserialization sinks (for example numpy.ctypeslib.ctypes and pickle-backed loaders), so a node opts
# into them explicitly through extra_allowed_modules rather than receiving them by default. yaml is left
# out on purpose because its default loader deserializes arbitrary Python objects.
EXTENDED_ALLOWED_MODULES = frozenset(
    {
        "numpy",
        "pandas",
        "requests",
        "urllib",
        "pydantic",
        "openpyxl",
        "docx",
        "pptx",
        "pdfplumber",
        "pypdf",
        "matplotlib",
        "seaborn",
        # dynamiq exposes framework internals, so it is opt-in rather than a default: it lets trusted code
        # build domain types such as Document. The attribute guard still blocks reaching os and the like
        # through a dynamiq submodule.
        "dynamiq",
    }
)

# Modules that restricted code may never import or reach, even when a caller lists them in
# extra_allowed_modules. Each grants process, filesystem, native-code or introspection access that would
# defeat the restrictions, so the import and attribute guards refuse them unconditionally.
FORBIDDEN_MODULES = frozenset(
    {
        "os",
        "sys",
        "subprocess",
        "ctypes",
        "_ctypes",
        "cffi",
        "importlib",
        "builtins",
        "socket",
        "pickle",
        "marshal",
        "shutil",
        "signal",
        "resource",
        "mmap",
        "multiprocessing",
        "threading",
        "pty",
        "fcntl",
        "gc",
        "inspect",
        "code",
        "codeop",
        "runpy",
    }
)

# Retained for backward compatibility with external importers; the nodes no longer read it.
ALLOWED_MODULES = sorted(DEFAULT_ALLOWED_MODULES | EXTENDED_ALLOWED_MODULES)

ALLOWED_DUNDER_ATTRIBUTES = {"__class__", "__name__", "__qualname__"}


class DynamiqRestrictingNodeTransformer(RestrictingNodeTransformer):
    """Override attribute checks so a short allow-list of dunders remains accessible."""

    allowed_dunder_attributes = ALLOWED_DUNDER_ATTRIBUTES

    def visit_Attribute(self, node):
        """Same as upstream implementation but permits whitelisted dunders."""
        if node.attr.startswith("_") and node.attr != "_" and node.attr not in self.allowed_dunder_attributes:
            self.error(
                node,
                f'"{node.attr}" is an invalid attribute name because it starts with "_".',
            )

        if node.attr.endswith("__roles__"):
            self.error(
                node,
                f'"{node.attr}" is an invalid attribute name because it ends with "__roles__".',
            )

        if node.attr in INSPECT_ATTRIBUTES:
            self.error(
                node,
                f'"{node.attr}" is a restricted name, that is forbidden to access in RestrictedPython.',
            )

        if isinstance(node.ctx, ast.Load):
            node = self.node_contents_visit(node)
            new_node = ast.Call(
                func=ast.Name("_getattr_", ast.Load()),
                args=[node.value, ast.Constant(node.attr)],
                keywords=[],
            )

            copy_locations(new_node, node)
            return new_node

        if isinstance(node.ctx, (ast.Store, ast.Del)):
            node = self.node_contents_visit(node)
            new_value = ast.Call(
                func=ast.Name("_write_", ast.Load()),
                args=[node.value],
                keywords=[],
            )

            copy_locations(new_value, node.value)
            node.value = new_value
            return node

        raise NotImplementedError(f"Unknown ctx type: {type(node.ctx)}")


def make_safe_print(printer: Callable[..., Any]):
    """Return a RestrictedPython-compatible print function bound to a printer callback."""

    def safe_print(*args, **print_kwargs):
        printer(*args, **print_kwargs)

    return safe_print


def guarded_write(obj, value=None):
    """RestrictedPython helper to safely propagate write operations."""
    return value if value is not None else obj


def resolve_allowed_modules(extra_allowed_modules: list[str] | None = None) -> frozenset[str]:
    """Return the effective import allow-list: the safe default plus extras, never a forbidden module.

    FORBIDDEN_MODULES always wins, so a caller cannot re-enable a dangerous module by listing it.
    """
    allowed = set(DEFAULT_ALLOWED_MODULES)
    if extra_allowed_modules:
        allowed.update(extra_allowed_modules)
    return frozenset(allowed - FORBIDDEN_MODULES)


def _make_restricted_import(allowed_modules: frozenset[str]) -> Callable[..., Any]:
    """Build an __import__ that admits only modules whose top-level package is in allowed_modules."""

    def restricted_import(name: str, globals=None, locals=None, fromlist=(), level=0) -> Any:
        if not name:
            raise ImportError("Empty module name")

        if level not in (0,):
            raise ImportError("Relative imports are not supported in the Python executor")

        root_module_name = name.split(".")[0]
        if root_module_name in FORBIDDEN_MODULES or root_module_name not in allowed_modules:
            logger.warning(f"Import of '{root_module_name}' is not allowed")
            raise ImportError(f"Import of '{root_module_name}' is not allowed")

        normalized_fromlist = fromlist or ()
        try:
            module = builtins.__import__(name, globals, locals, normalized_fromlist, level)
            logger.debug(f"Successfully imported {name}")
            return module
        except ImportError as e:
            logger.error(f"Failed to import {name}: {str(e)}")
            if optional_error := optional_dependency_error(e, name):
                raise optional_error from e
            raise

    return restricted_import


def _make_guarded_getattr(allowed_modules: frozenset[str]) -> Callable[..., Any]:
    """Build a _getattr_ that forbids reaching a module outside allowed_modules through an attribute.

    RestrictedPython's import guard only checks the "import" statement, but allow-listed modules re-export
    other modules as plain attributes (numpy.ctypeslib.ctypes, urllib.request.os). Enforcing the allow-list
    on every module-valued attribute closes that whole class without naming each path.
    """

    def guarded_getattr(obj: Any, name: str, *args: Any, **kwargs: Any) -> Any:
        value = default_guarded_getattr(obj, name, *args, **kwargs)
        if isinstance(value, ModuleType):
            root_module_name = getattr(value, "__name__", "").split(".")[0]
            if root_module_name in FORBIDDEN_MODULES or root_module_name not in allowed_modules:
                raise ImportError(f"Access to module '{getattr(value, '__name__', name)}' is not allowed")
        return value

    return guarded_getattr


def get_restricted_globals(allowed_modules: list[str] | frozenset[str] | None = None) -> dict:
    """Return a globals dict for restricted execution limited to allowed_modules.

    allowed_modules is the set of importable top-level packages; None uses DEFAULT_ALLOWED_MODULES.
    Modules in FORBIDDEN_MODULES are refused regardless. The returned import and attribute guards are
    bound to this set, so each call is independent of any other.
    """
    if allowed_modules is None:
        allowed = frozenset(DEFAULT_ALLOWED_MODULES)
    else:
        allowed = frozenset(allowed_modules) - FORBIDDEN_MODULES

    restricted_import = _make_restricted_import(allowed)
    guarded_getattr = _make_guarded_getattr(allowed)

    def safe_getattr(obj: Any, name: str, default=None) -> Any:
        """Guarded getattr builtin: the allow-list and dunder rules apply as in compiled attribute access."""
        if name in ALLOWED_DUNDER_ATTRIBUTES:
            try:
                return getattr(obj, name)
            except AttributeError:
                if default is not None:
                    return default
                raise

        try:
            return guarded_getattr(obj, name)
        except (AttributeError, TypeError):
            if default is not None:
                return default
            raise

    def safe_hasattr(obj: Any, name: str) -> bool:
        """Guarded hasattr builtin. A blocked module attribute reports as absent rather than raising."""
        try:
            guarded_getattr(obj, name)
            return True
        except (AttributeError, TypeError, ImportError):
            return False

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
            "_getattr_": guarded_getattr,
            "_getitem_": default_guarded_getitem,
            "_getiter_": default_guarded_getiter,
            "__metaclass__": type,
            "__name__": "__main__",
        },
        "_getattr_": guarded_getattr,
        "_unpack_sequence_": guarded_unpack_sequence,
    }


def compile_and_execute(code: str, restricted_globals: dict) -> dict:
    """
    Compile the code using RestrictedPython and execute it in restricted_globals.
    Returns the updated restricted_globals.
    """
    try:
        byte_code = compile_restricted(code, "<inline>", "exec", policy=DynamiqRestrictingNodeTransformer)
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
    Node for executing Python code under RestrictedPython.

    RestrictedPython limits what imported code can reach but is not a security boundary for untrusted
    input: allowed third-party libraries still expose native-code and deserialization sinks. Run code
    from an untrusted source in a sandbox node instead.

    Attributes:
      group (Literal[NodeGroup.TOOLS]): node group.
      name (str): node name.
      description (str): node description.
      code (str): Python code to execute.
      use_multiple_params (bool): Determines how the input dictionary is passed to the Python function.
        -If True, the input dictionary is unpacked into multiple parameters.
        -If False, it is passed as a single dictionary with all parameters as keys.
      extra_allowed_modules (list[str]): Top-level packages importable on top of the safe standard-library
        default, such as "numpy" or "requests". Enabling these widens the attack surface, so set it only
        for code you trust. Modules in FORBIDDEN_MODULES can never be enabled.
    """
    group: Literal[NodeGroup.TOOLS] = NodeGroup.TOOLS
    action_type: ActionType = ActionType.CODE_EXECUTION
    name: str = "python-tool"
    description: str = """Executes Python code with restricted imports for calculations, data processing,
    and API interactions. By default only the Python standard library is importable; configure
    extra_allowed_modules to enable libraries such as numpy or requests."""  # noqa: E501
    code: str
    use_multiple_params: bool = False
    extra_allowed_modules: list[str] = Field(default_factory=list)
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
        config = ensure_config(config)
        check_cancellation(config)
        self.run_on_node_execute_run(config.callbacks, **kwargs)
        stdout = io.StringIO()
        safe_print = make_safe_print(lambda *args, **print_kwargs: print(*args, file=stdout, **print_kwargs))

        restricted_globals = get_restricted_globals(resolve_allowed_modules(self.extra_allowed_modules))
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
