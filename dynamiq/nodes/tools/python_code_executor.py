import importlib
import inspect
import io
import json
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field
from RestrictedPython.Guards import guarded_unpack_sequence
from RestrictedPython.PrintCollector import PrintCollector

from dynamiq.nodes import Node, NodeGroup
from dynamiq.nodes.agents.exceptions import ToolExecutionException
from dynamiq.nodes.agents.utils import FileMappedInput
from dynamiq.nodes.node import ensure_config
from dynamiq.nodes.tools.python import (
    Python,
    compile_and_execute,
    get_restricted_globals,
    guarded_write,
    make_safe_print,
)
from dynamiq.runnables import RunnableConfig
from dynamiq.storages.file.base import FileInfo, FileStore
from dynamiq.utils import format_value
from dynamiq.utils.logger import logger

OPTIONAL_PRELOAD_MODULES: tuple[tuple[str, str | None], ...] = (
    ("matplotlib", None),
    ("matplotlib.pyplot", "plt"),
)


class PythonCodeExecutorFileWorkspace(BaseModel):
    """Helper exposing safe file store operations for RestrictedPython code."""

    file_store: FileStore = Field(..., description="File storage backend shared with the executor.")

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def list(self, directory: str = "", recursive: bool = True) -> list[dict[str, Any]]:
        """Return metadata for files located under directory."""
        files = self.file_store.list_files(directory=directory, recursive=recursive)
        return [self._file_info_to_dict(file_info) for file_info in files]

    def read_text(self, path: str, encoding: str = "utf-8", errors: str = "ignore") -> str:
        """Read file as text."""
        return self.file_store.retrieve(path).decode(encoding, errors=errors)

    def read_bytes(self, path: str) -> bytes:
        """Read file as bytes."""
        return self.file_store.retrieve(path)

    def write_text(
        self,
        path: str,
        content: str,
        encoding: str = "utf-8",
        metadata: dict[str, Any] | None = None,
        overwrite: bool = True,
    ) -> dict[str, Any]:
        """Write text content to storage."""
        return self._store_file(
            path=path,
            content=content.encode(encoding),
            content_type="text/plain",
            metadata=metadata,
            overwrite=overwrite,
        )

    def write_bytes(
        self,
        path: str,
        content: bytes,
        content_type: str | None = None,
        metadata: dict[str, Any] | None = None,
        overwrite: bool = True,
    ) -> dict[str, Any]:
        """Write binary content to storage."""
        return self._store_file(
            path=path, content=content, content_type=content_type, metadata=metadata, overwrite=overwrite
        )

    def write(
        self,
        path: str,
        content: str | bytes,
        content_type: str | None = None,
        metadata: dict[str, Any] | None = None,
        overwrite: bool = True,
    ) -> dict[str, Any]:
        """Write generic content to storage."""
        if isinstance(content, str):
            content_bytes = content.encode("utf-8")
            content_type = content_type or "text/plain"
        else:
            content_bytes = content
        return self._store_file(
            path=path, content=content_bytes, content_type=content_type, metadata=metadata, overwrite=overwrite
        )

    def delete(self, path: str) -> bool:
        """Delete file."""
        return self.file_store.delete(path)

    def exists(self, path: str) -> bool:
        """Check if file exists."""
        return self.file_store.exists(path)

    def describe(self, path: str, preview_bytes: int = 2048) -> dict[str, Any]:
        """Return metadata and short preview."""
        file_info = self._locate_file_info(path)
        preview = b""
        try:
            preview = self.file_store.retrieve(path)[:preview_bytes]
        except Exception:
            preview = b""

        try:
            preview_text = preview.decode("utf-8")
            preview_is_text = True
        except UnicodeDecodeError:
            preview_text = preview.hex()
            preview_is_text = False

        info_dict = self._file_info_to_dict(file_info) if file_info else {"path": path}
        info_dict.update({"preview": preview_text, "preview_is_text": preview_is_text})
        return info_dict

    def _store_file(
        self,
        path: str,
        content: bytes,
        content_type: str | None = None,
        metadata: dict[str, Any] | None = None,
        overwrite: bool = True,
    ) -> dict[str, Any]:
        file_info = self.file_store.store(
            path, content, content_type=content_type, metadata=metadata, overwrite=overwrite
        )
        return self._file_info_to_dict(file_info)

    def _locate_file_info(self, path: str) -> FileInfo | None:
        try:
            files = self.file_store.list_files(recursive=True)
        except Exception:
            return None

        for info in files:
            if info.path == path:
                return info
        return None

    @staticmethod
    def _file_info_to_dict(file_info: FileInfo | None) -> dict[str, Any]:
        if not file_info:
            return {}
        return {
            "name": file_info.name,
            "path": file_info.path,
            "size": file_info.size,
            "content_type": file_info.content_type,
            "created_at": file_info.created_at.isoformat(),
            "metadata": file_info.metadata,
        }


class PythonCodeExecutorInputSchema(BaseModel):
    """Inputs for the PythonCodeExecutor tool."""

    code: str = Field(..., description="Python code to execute. Must define a 'run' function.")
    params: dict[str, Any] = Field(
        default_factory=dict,
        description="Optional parameters forwarded to the executed code.",
    )
    files: list[Any] | FileMappedInput | None = Field(
        default=None,
        description="Files available to the executed code.",
        json_schema_extra={"map_from_storage": True},
    )

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")


class PythonCodeExecutor(Node):
    """Execute ad-hoc Python code inside RestrictedPython with file store helpers."""

    group: Literal[NodeGroup.TOOLS] = NodeGroup.TOOLS
    name: str = "PythonCodeExecutor"
    description: str = (
        "Runs dynamic Python code safely via RestrictedPython. The tool injects helper functions such as read_file(), "
        "write_file(), and list_files(), exposing the shared file store so you can read/write artifacts inside the "
        "workspace. Available libraries include: pandas, numpy, scipy, requests, pdfplumber, PyPDF2/pypdf, docx "
        "(python-docx), pptx (python-pptx), markdown, openpyxl, matplotlib (with `matplotlib.pyplot` preloaded as "
        "`plt`), and standard library utilities. Always return structured results from `run` — any `print()` output is "
        "captured separately in a `stdout` field.\n\n"
        "- Every code snippet you send to the executor MUST define a run(...) function as the entrypoint.\n"
        "- Use helper functions such as read_file(), write_file(), and list_files() provided by the code executor."
    )
    file_store: FileStore = Field(..., description="File storage backend shared with the agent.")
    is_files_allowed: bool = True
    input_schema: ClassVar[type[PythonCodeExecutorInputSchema]] = PythonCodeExecutorInputSchema

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def execute(
        self, input_data: PythonCodeExecutorInputSchema, config: RunnableConfig = None, **kwargs
    ) -> dict[str, Any]:
        code_preview = input_data.code.strip().replace("\n", "\\n")[:200]
        logger.info(
            f"Tool {self.name} - {self.id}: started with code length={len(input_data.code)} "
            f"preview='{code_preview}{'...' if len(input_data.code) > 200 else ''}'"
        )
        config = ensure_config(config)
        self.run_on_node_execute_run(config.callbacks, **kwargs)

        if not self.file_store:
            raise ToolExecutionException("PythonCodeExecutor requires a configured file_store.", recoverable=False)

        file_manager = PythonCodeExecutorFileWorkspace(file_store=self.file_store)
        helpers, helper_aliases = self._build_file_helpers(file_manager)

        stdout_collector = PrintCollector()
        safe_print = make_safe_print(stdout_collector._call_print)

        restricted_globals = get_restricted_globals()
        restricted_globals.update(helper_aliases)
        restricted_globals.update(
            {
                "_inplacevar_": Python._inplacevar,
                "_iter_unpack_sequence_": guarded_unpack_sequence,
                "safe_print": safe_print,
                "print": safe_print,
                "_write_": guarded_write,
                "_print_": stdout_collector,
                "_print": stdout_collector._call_print,
            }
        )
        restricted_globals.update(self._preload_optional_modules())

        try:
            restricted_globals = compile_and_execute(input_data.code, restricted_globals)
        except Exception as e:  # pragma: no cover - RestrictedPython already sanitizes stack
            error_msg = f"Code compilation error: {str(e)}"
            logger.error(error_msg)
            raise ToolExecutionException(error_msg, recoverable=True)

        run_callable = restricted_globals.get("run")
        if not callable(run_callable):
            raise ToolExecutionException(
                "The provided code must define a callable function named 'run'.", recoverable=True
            )

        files, mapped_inputs = self._normalize_files(input_data.files)
        context_args = self._build_context(
            files=files,
            mapped_inputs=mapped_inputs,
            file_manager=file_manager,
            stdout=stdout_collector,
            params=input_data.params,
            helpers=helpers,
        )

        args, kwargs = self._prepare_callable_arguments(run_callable, context_args)

        try:
            result = run_callable(*args, **kwargs)
        except Exception as e:  # pragma: no cover - sanitized error forwarded to the agent
            error_msg = f"Code execution error: {str(e)}"
            logger.error(error_msg)
            raise ToolExecutionException(error_msg, recoverable=True)

        stdout_value = stdout_collector().strip()
        content = self._attach_stdout(result, stdout_value)
        sanitized_content = self._sanitize_output(content)
        if stdout_value:
            logger.debug(f"Tool {self.name} - {self.id}: captured stdout ({len(stdout_value)} chars).")
        keys_repr = list(sanitized_content.keys()) if isinstance(sanitized_content, dict) else type(sanitized_content)
        logger.info(
            f"Tool {self.name} - {self.id}: finished execution with keys {keys_repr} "
            f"preview={self._preview_for_log(sanitized_content)}"
        )
        return {"content": sanitized_content}

    def _normalize_files(self, files: list[Any] | FileMappedInput | None) -> tuple[list[Any], dict[str, Any]]:
        mapped_inputs: dict[str, Any] = {}
        provided_files: list[Any] = []

        if isinstance(files, FileMappedInput):
            provided_files = list(files.files or [])
            mapped_inputs = files.input or {}
        elif isinstance(files, list):
            provided_files = files
        elif files is None:
            provided_files = []
        else:
            provided_files = [files]

        normalized_files = [self._coerce_file_reference(file_obj) for file_obj in provided_files]
        return normalized_files, mapped_inputs

    def _build_context(
        self,
        files: list[Any],
        mapped_inputs: dict[str, Any],
        file_manager: PythonCodeExecutorFileWorkspace,
        stdout: io.StringIO,
        params: dict[str, Any] | None = None,
        helpers: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        context_payload = {
            "params": params or {},
            "files": files,
            "file_inputs": mapped_inputs,
            "file_manager": file_manager,
            "workspace": file_manager,
            "helpers": helpers or {},
            "stdout": stdout,
        }
        available_args = {"context": context_payload}
        available_args.update(context_payload)
        return available_args

    def _preload_optional_modules(self) -> dict[str, Any]:
        """Expose heavier-but-common modules (e.g., matplotlib.pyplot) inside the restricted globals."""
        preloaded: dict[str, Any] = {}
        for module_path, alias in OPTIONAL_PRELOAD_MODULES:
            try:
                module = importlib.import_module(module_path)
            except Exception as exc:  # noqa: BLE001 broad to guard optional deps
                logger.debug(
                    "Optional preload module %s unavailable in python executor: %s",
                    module_path,
                    exc,
                )
                continue

            if alias:
                preloaded.setdefault(alias, module)
            else:
                root_name = module_path.split(".")[0]
                preloaded.setdefault(root_name, module)
        return preloaded

    def _build_file_helpers(
        self, file_manager: PythonCodeExecutorFileWorkspace | None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        if not file_manager:
            return {}, {}

        helper_funcs = {
            "list_files": file_manager.list,
            "read_text_file": file_manager.read_text,
            "read_bytes_file": file_manager.read_bytes,
            "write_text_file": file_manager.write_text,
            "write_bytes_file": file_manager.write_bytes,
            "write_file": file_manager.write,
            "delete_file": file_manager.delete,
            "file_exists": file_manager.exists,
            "describe_file": file_manager.describe,
        }

        alias_map = {
            "list_files": helper_funcs["list_files"],
            "read_file": helper_funcs["read_text_file"],
            "read_file_text": helper_funcs["read_text_file"],
            "read_file_bytes": helper_funcs["read_bytes_file"],
            "write_file": helper_funcs["write_file"],
            "write_text": helper_funcs["write_text_file"],
            "write_bytes": helper_funcs["write_bytes_file"],
            "delete_file": helper_funcs["delete_file"],
            "file_exists": helper_funcs["file_exists"],
            "describe_file": helper_funcs["describe_file"],
        }

        for alias, func in alias_map.items():
            helper_funcs.setdefault(alias, func)

        return helper_funcs, alias_map

    def _coerce_file_reference(self, file_obj: Any) -> Any:
        if isinstance(file_obj, str):
            fetched = self._load_file_from_store(file_obj)
            return fetched or file_obj
        return file_obj

    def _load_file_from_store(self, path: str) -> io.BytesIO | None:
        if not self.file_store:
            return None
        try:
            if not self.file_store.exists(path):
                return None
            content = self.file_store.retrieve(path)
            buffer = io.BytesIO(content)
            buffer.name = path
            try:
                info = next((f for f in self.file_store.list_files(recursive=True) if f.path == path), None)
                if info:
                    buffer.description = info.metadata.get("description", "")
                    buffer.content_type = info.content_type
            except Exception:
                buffer.description = ""
            return buffer
        except Exception:
            return None

    def _prepare_callable_arguments(self, func, context_args: dict[str, Any]) -> tuple[list[Any], dict[str, Any]]:
        signature = inspect.signature(func)
        positional_args: list[Any] = []
        keyword_args: dict[str, Any] = {}
        param_names = []
        has_var_kwargs = False

        for param in signature.parameters.values():
            param_names.append(param.name)
            if param.kind == inspect.Parameter.POSITIONAL_ONLY:
                positional_args.append(context_args.get(param.name))
            elif param.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY):
                if param.name in context_args:
                    keyword_args[param.name] = context_args[param.name]
            elif param.kind == inspect.Parameter.VAR_KEYWORD:
                has_var_kwargs = True

        if has_var_kwargs:
            for key, value in context_args.items():
                if key not in keyword_args and key not in param_names:
                    keyword_args[key] = value

        return positional_args, keyword_args

    @staticmethod
    def _attach_stdout(result: Any, stdout_value: str) -> Any:
        if stdout_value:
            if isinstance(result, dict):
                return {**result, "stdout": stdout_value}
            if result is None:
                return {"stdout": stdout_value}
            return {"result": result, "stdout": stdout_value}
        if result is None:
            return {"result": None}
        return result

    def _sanitize_output(self, content: Any) -> Any:
        """
        Attempt to convert complex data structures (pandas/numpy objects, etc.) into
        JSON-serializable data so agents and tracing layers can consume the result safely.
        """

        def default_serializer(obj: Any):
            if hasattr(obj, "to_dict"):
                try:
                    return obj.to_dict()
                except Exception as err:
                    logger.debug("Failed to serialize %s via to_dict: %s", type(obj).__name__, err)
            if hasattr(obj, "tolist"):
                try:
                    return obj.tolist()
                except Exception as err:
                    logger.debug("Failed to serialize %s via tolist: %s", type(obj).__name__, err)
            if hasattr(obj, "item"):
                try:
                    return obj.item()
                except Exception as err:
                    logger.debug("Failed to serialize %s via item(): %s", type(obj).__name__, err)
            return str(obj)

        try:
            normalized = self._stringify_keys(content)
            serialized = json.loads(json.dumps(normalized, default=default_serializer))
            return serialized
        except Exception:
            return format_value(self._stringify_keys(content))

    def _stringify_keys(self, value: Any) -> Any:
        if isinstance(value, dict):
            converted = {}
            for key, item in value.items():
                if isinstance(key, (str, int, float, bool)) or key is None:
                    new_key = key
                else:
                    new_key = str(key)
                converted[new_key] = self._stringify_keys(item)
            return converted
        if isinstance(value, list):
            return [self._stringify_keys(item) for item in value]
        if isinstance(value, tuple):
            return tuple(self._stringify_keys(item) for item in value)
        return value

    def _preview_for_log(self, value: Any, limit: int = 200) -> str:
        try:
            serialized = json.dumps(value)
        except Exception:
            serialized = str(value)
        serialized = serialized.replace("\n", "\\n")
        if len(serialized) > limit:
            return f"'{serialized[:limit]}...'"
        return f"'{serialized}'"
