import ast
import builtins
import io
from collections import UserString
from copy import deepcopy
from types import BuiltinFunctionType, FunctionType, MethodType, ModuleType
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
# offer to untrusted code because they grant no process, filesystem or native-code access. Some stdlib
# modules are deliberately excluded because they resolve attributes or eval code in C, outside every
# guard: "operator" (attrgetter, methodcaller), "string" (Formatter walks dunder fields) and "typing"
# (get_type_hints and ForwardRef eval string annotations). "io" is kept for StringIO and BytesIO, but its
# file openers are refused through _FORBIDDEN_BUILTINS.
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
        "random",
        "re",
        "statistics",
        "time",
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


# Objects that are process, filesystem or code-execution primitives, refused by identity wherever a
# snippet reaches them. The compiler blocks the names eval, exec and compile, but they and the file
# openers stay reachable as attributes of allowed modules (io.open is the builtin open; io.FileIO and
# io.open_code are separate objects). collections.UserString.format delegates to str.format in library
# code, past the str.format guard, so the class itself is refused.
_FORBIDDEN_BUILTINS = frozenset(
    obj
    for obj in (
        builtins.open,
        builtins.eval,
        builtins.exec,
        builtins.compile,
        builtins.input,
        builtins.breakpoint,
        builtins.__import__,
        io.FileIO,
        getattr(io, "open_code", None),
        UserString,
    )
    if obj is not None
)


def _is_blocked_attr_name(name: str) -> bool:
    """Report whether an attribute name is refused, mirroring the compile-time attribute checks.

    The compiler rejects a private, dunder-roles or inspect name only in attribute syntax; the same name
    passed to the getattr or hasattr builtin as a string has to be rejected here too.
    """
    if not isinstance(name, str):
        return False
    if name.endswith("__roles__") or name in INSPECT_ATTRIBUTES:
        return True
    return name.startswith("_") and name != "_"


def _is_shared_write_target(obj: Any) -> bool:
    """Report whether obj is shared across runs, so mutating it would leak state between them.

    A module, and the classes and functions an allowed module defines, live in the process rather than in
    a single run's globals, so writing to them would affect every run and tenant (for example changing
    json.dumps.__defaults__). A class or function the running snippet defines has __module__ "__main__"
    and is recreated for each run, so writing to it is allowed; instances are always writable. A snippet
    cannot disguise a shared object as its own, since changing its __module__ is itself a blocked write.
    """
    if isinstance(obj, ModuleType):
        return True
    if isinstance(obj, (type, FunctionType, MethodType, BuiltinFunctionType)):
        return getattr(obj, "__module__", None) != "__main__"
    return False


def guarded_write(obj, value=None):
    """RestrictedPython write guard. Writing to a shared module, class or function is refused (see above)."""
    if _is_shared_write_target(obj):
        raise TypeError("Writing attributes on a shared module, class or function is not allowed")
    return value if value is not None else obj


def safe_setattr(obj: Any, name: str, value: Any) -> None:
    """setattr builtin: refuses a private name and any shared module, class or function target.

    The compiler rejects a private name and a shared target in attribute-write syntax, but the setattr
    builtin bypasses both, so both checks are repeated here.
    """
    if _is_blocked_attr_name(name):
        raise AttributeError(f'"{name}" is not an allowed attribute name')
    if _is_shared_write_target(obj):
        raise TypeError("Setting attributes on a shared module, class or function is not allowed")
    setattr(obj, name, value)


def resolve_allowed_modules(extra_allowed_modules: list[str] | None = None) -> frozenset[str]:
    """Return the effective import allow-list: the safe default plus extras, never a forbidden module.

    FORBIDDEN_MODULES always wins, so a caller cannot re-enable a dangerous module by listing it.
    """
    allowed = set(DEFAULT_ALLOWED_MODULES)
    if extra_allowed_modules:
        allowed.update(extra_allowed_modules)
    return frozenset(allowed - FORBIDDEN_MODULES)


def _forbidden_value_reason(value: Any, allowed_modules: frozenset[str]) -> str | None:
    """Return why value is off limits, or None if it is allowed to cross into restricted code.

    A module is refused unless its top-level package is in allowed_modules and not forbidden; a process,
    filesystem or code-execution builtin (open, eval, ...) is refused by identity; a ctypes object is
    refused by its type, so a native handle from any path (numpy.ctypeslib.load_library) cannot be used.
    Everything else passes.
    """
    if isinstance(value, ModuleType):
        root_module_name = (getattr(value, "__name__", "") or "").split(".")[0]
        if root_module_name in FORBIDDEN_MODULES or root_module_name not in allowed_modules:
            return f"module '{getattr(value, '__name__', '')}'"
        return None
    if any(value is forbidden for forbidden in _FORBIDDEN_BUILTINS):
        return "builtin"
    if (getattr(type(value), "__module__", "") or "").split(".")[0] in ("ctypes", "_ctypes"):
        return "ctypes object"
    return None


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
        except ImportError as e:
            logger.error(f"Failed to import {name}: {str(e)}")
            if optional_error := optional_dependency_error(e, name):
                raise optional_error from e
            raise

        # "from pkg.sub import name" resolves each name with a raw attribute read that the getattr guard
        # never sees, so a forbidden module or builtin could be bound directly (from numpy.ctypeslib import
        # ctypes, from io import open). Apply the same check to every imported name.
        for item in _expand_fromlist(module, normalized_fromlist):
            if _forbidden_value_reason(getattr(module, item, None), allowed_modules):
                logger.warning(f"Import of '{item}' from '{name}' is not allowed")
                raise ImportError(f"Import of '{item}' from '{name}' is not allowed")
        return module

    return restricted_import


def _expand_fromlist(module: ModuleType, fromlist: tuple[str, ...]) -> list[str]:
    """Return the attribute names a fromlist binds, expanding "*" to the module's public names."""
    names: list[str] = []
    for item in fromlist:
        if item == "*":
            public = getattr(module, "__all__", None) or [n for n in vars(module) if not n.startswith("_")]
            names.extend(public)
        else:
            names.append(item)
    return names


def _is_str_type(obj: Any) -> bool:
    """Report whether obj is a str (or str subclass, or the str type itself)."""
    return isinstance(obj, str) or (isinstance(obj, type) and issubclass(obj, str))


def _make_guarded_getattr(allowed_modules: frozenset[str]) -> Callable[..., Any]:
    """Build a _getattr_ that forbids reaching a forbidden module, builtin or ctypes object by attribute.

    RestrictedPython's import guard only checks the "import" statement, but allow-listed modules re-export
    other modules and builtins as plain attributes (numpy.ctypeslib.ctypes, urllib.request.os, io.open).
    Enforcing the allow-list on every attribute closes that whole class without naming each path. str.format
    and format_map are refused too: their replacement fields resolve attributes in C, past every guard.
    """

    def guarded_getattr(obj: Any, name: str, *args: Any, **kwargs: Any) -> Any:
        if name in ("format", "format_map") and _is_str_type(obj):
            raise AttributeError("str.format and str.format_map are not allowed; use an f-string")
        value = default_guarded_getattr(obj, name, *args, **kwargs)
        if reason := _forbidden_value_reason(value, allowed_modules):
            raise ImportError(f"Access to {reason} through '{name}' is not allowed")
        return value

    return guarded_getattr


def _make_guarded_getitem(allowed_modules: frozenset[str]) -> Callable[..., Any]:
    """Build a _getitem_ that applies the same forbidden-value check as attribute access.

    Subscription can also hand back a forbidden object (a CDLL function by ordinal, a module from a
    __dict__-like mapping), and RestrictedPython's default getitem does not look at the result.
    """

    def guarded_getitem(obj: Any, index: Any) -> Any:
        value = default_guarded_getitem(obj, index)
        if reason := _forbidden_value_reason(value, allowed_modules):
            raise ImportError(f"Access to {reason} by subscription is not allowed")
        return value

    return guarded_getitem


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
    guarded_getitem = _make_guarded_getitem(allowed)

    def safe_getattr(obj: Any, name: str, default=None) -> Any:
        """Guarded getattr builtin: the allow-list and dunder rules apply as in compiled attribute access.

        A private, roles or inspect name passed as a string is refused here, since the compiler's check
        only sees attribute syntax; this closes getattr(object, "__subclasses__") and the like.
        """
        if name not in ALLOWED_DUNDER_ATTRIBUTES and _is_blocked_attr_name(name):
            raise AttributeError(f'"{name}" is not an allowed attribute name')

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
        """Guarded hasattr builtin. A blocked or absent attribute reports as absent rather than raising."""
        try:
            safe_getattr(obj, name)
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
            "setattr": safe_setattr,
            "slice": slice,
            "sorted": sorted,
            "str": str,
            "sum": sum,
            "super": super,
            "tuple": tuple,
            "type": type,
            "zip": zip,
            "__import__": restricted_import,
            "_getattr_": guarded_getattr,
            "_getitem_": guarded_getitem,
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
