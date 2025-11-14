import errno
import importlib
import inspect
import io
import json
import os
import shutil
import tempfile
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

os.environ.setdefault("MPLBACKEND", "Agg")

OPTIONAL_PRELOAD_MODULES: tuple[tuple[str, str | None], ...] = (("seaborn", "sns"),)


class PythonCodeExecutorFileWorkspace(BaseModel):
    """Helper exposing safe file store operations for RestrictedPython code."""

    file_store: FileStore = Field(..., description="File storage backend shared with the executor.")
    TEXTUAL_EXTENSIONS: ClassVar[set[str]] = {
        "txt",
        "md",
        "markdown",
        "json",
        "yaml",
        "yml",
        "csv",
        "tsv",
        "toml",
        "xml",
        "html",
        "htm",
        "css",
        "js",
        "py",
        "java",
        "ts",
        "tsx",
        "c",
        "cpp",
        "rs",
        "go",
    }

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

    def read(
        self,
        path: str,
        *,
        encoding: str = "utf-8",
        errors: str = "ignore",
        force_text: bool | None = None,
    ) -> str | bytes:
        """
        Read a file, automatically decoding text content while keeping binary files intact.
        """
        payload = self.file_store.retrieve(path)

        decode_as_text = force_text if force_text is not None else self._should_decode_as_text(path, payload)
        if not decode_as_text:
            return payload

        try:
            return payload.decode(encoding, errors=errors)
        except UnicodeDecodeError:
            return payload

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

    @classmethod
    def _should_decode_as_text(cls, path: str, payload: bytes) -> bool:
        extension = os.path.splitext(path)[1].lower().lstrip(".")
        if extension in cls.TEXTUAL_EXTENSIONS:
            return True
        if b"\x00" in payload:
            return False
        try:
            payload.decode("utf-8")
            return True
        except UnicodeDecodeError:
            return False

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
        "Runs dynamic Python code safely via RestrictedPython. "
        "The tool injects file helpers so you can operate on the "
        "shared file store without manually copying artifacts. "
        "Before your code runs, every uploaded file is mirrored into "
        "the isolated workspace (so even module-level `pd.read_csv('foo.csv')`,"
        " `pd.read_excel('bar.xlsx')`, or "
        "`open('notes.txt')` calls just work). The helpers remain the most "
        "reliable way to read/write artifacts—especially "
        "when you need metadata, binary vs text detection, or want a "
        "consistent API—so prefer read_file()/write_file(), "
        "list_files(), file_exists(), describe_file(), etc., or"
        " call them via `helpers['read_file']('foo.csv')`. Matplotlib "
        "is forced onto the 'Agg' backend so you can always render"
        " plots headlessly with `plt.savefig(...)` (no GUI popups). "
        "Available libraries include: pandas, numpy, "
        "requests, pdfplumber, PyPDF2 and pypdf, docx (python-docx), "
        "pptx (python-pptx), markdown, openpyxl, "
        "seaborn (preloaded as `sns` and the default for plotting), and the "
        "standard library. Default to seaborn for"
        " charting so plots render consistently without extra setup. Seaborn is "
        "already injected as `sns`, but explicit "
        "`import seaborn as sns` or `import matplotlib.pyplot as plt` statements "
        "are also permitted if you prefer. Write normal Python "
        "exactly as it would appear in a `.py` file—use literal "
        "newlines instead of escaping them, "
        "and avoid JSON-style quoting inside the source. Any "
        "`print()` output is captured separately "
        "in a `stdout` field, so have `run` return structured data.\n\n"
        "- Every snippet you send MUST define "
        "a callable `run` entrypoint. Prefer explicit parameters such as "
        "`def run(params=None, helpers=None, files=None, stdout=None, context=None): ...`"
        "- `run` can optionally accept params, files,"
        " file_inputs, helpers, workspace/file_manager, stdout, or a single "
        "context dict; pass only what you need in "
        "the signature and everything else is still accessible via the `context` "
        "dictionary if you include it.\n"
        "- Uploaded files are only visible through the helper "
        "functions; use write_* helpers to persist new artifacts "
        "back to the shared store.\n"
        "- RestrictedPython forbids augmented assignment "
        "on subscripts/slices (e.g., `counts[key] += 1`); use "
        "`counts[key] = counts[key] + 1` instead."
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

        def _print_factory(getattr_func=None):
            """RestrictedPython expects _print_ to be a factory; bind it to our collector."""
            stdout_collector._getattr_ = getattr_func
            return stdout_collector

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
                "_print_": _print_factory,
                "_print": stdout_collector._call_print,
            }
        )
        restricted_globals.update(self._preload_optional_modules())

        workspace_dir = tempfile.mkdtemp(prefix="dynamiq_executor_")
        original_cwd = os.getcwd()

        try:
            os.chdir(workspace_dir)

            files, mapped_inputs, workspace_paths = self._normalize_files(input_data.files)
            self._materialize_inline_files(files, workspace_dir)
            self._materialize_file_store_contents(workspace_dir, workspace_paths)

            try:
                restricted_globals = compile_and_execute(input_data.code, restricted_globals)
            except Exception as e:  # pragma: no cover - RestrictedPython already sanitizes stack
                error_msg = f"Code compilation error: {e}"
                logger.error(error_msg)
                raise ToolExecutionException(error_msg, recoverable=True)

            run_callable = restricted_globals.get("run")
            if not callable(run_callable):
                raise ToolExecutionException(
                    "The provided code must define a callable function named 'run'.", recoverable=True
                )
            context_args = self._build_context(
                files=files,
                mapped_inputs=mapped_inputs,
                file_manager=file_manager,
                stdout=stdout_collector,
                params=input_data.params,
                helpers=helpers,
            )
            context_args["local_workspace_dir"] = workspace_dir

            args, kwargs = self._prepare_callable_arguments(run_callable, context_args)

            try:
                result = run_callable(*args, **kwargs)
            except Exception as e:  # pragma: no cover - sanitized error forwarded to the agent
                friendly_file_error = self._format_file_access_error(e)
                if friendly_file_error:
                    logger.error(friendly_file_error)
                    raise ToolExecutionException(friendly_file_error, recoverable=True) from e
                error_msg = f"Code execution error: {e}"
                logger.error(error_msg)
                raise ToolExecutionException(error_msg, recoverable=True)

            stdout_value = stdout_collector().strip()
            content = self._attach_stdout(result, stdout_value)
            sanitized_content = self._sanitize_output(content)
            if stdout_value:
                logger.debug(f"Tool {self.name} - {self.id}: captured stdout ({len(stdout_value)} chars).")
            keys_repr = (
                list(sanitized_content.keys()) if isinstance(sanitized_content, dict) else type(sanitized_content)
            )
            logger.info(
                f"Tool {self.name} - {self.id}: finished execution with keys {keys_repr} "
                f"preview={self._preview_for_log(sanitized_content)}"
            )
            return {"content": sanitized_content}
        finally:
            os.chdir(original_cwd)
            shutil.rmtree(workspace_dir, ignore_errors=True)

    def _normalize_files(
        self, files: list[Any] | FileMappedInput | None
    ) -> tuple[list[Any], dict[str, Any], list[str]]:
        mapped_inputs: dict[str, Any] = {}
        provided_files: list[Any] = []
        workspace_path_set: set[str] = set()

        if isinstance(files, FileMappedInput):
            provided_files = list(files.files or [])
            mapped_inputs = files.input or {}
        elif isinstance(files, list):
            provided_files = files
        elif files is None:
            provided_files = []
        else:
            provided_files = [files]

        normalized_files = []
        for file_obj in provided_files:
            normalized_files.append(self._coerce_file_reference(file_obj))
            if isinstance(file_obj, str) and self.file_store and self.file_store.exists(file_obj):
                workspace_path_set.add(file_obj)

        return normalized_files, mapped_inputs, sorted(workspace_path_set)

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

    def _materialize_inline_files(self, files: list[Any], workspace_dir: str) -> None:
        """
        Persist in-memory file-like inputs (e.g., BytesIO objects) into the workspace filesystem.
        """
        if not files:
            return

        for idx, file_obj in enumerate(files):
            if not hasattr(file_obj, "read"):
                continue

            file_name = getattr(file_obj, "name", f"inline_file_{idx}")
            sanitized_name = os.path.normpath(file_name).lstrip(os.sep)
            if sanitized_name.startswith(".."):
                sanitized_name = sanitized_name.replace("..", "")
            sanitized_name = sanitized_name or f"inline_file_{idx}"
            destination = os.path.join(workspace_dir, sanitized_name)
            destination_dir = os.path.dirname(destination) or workspace_dir
            os.makedirs(destination_dir, exist_ok=True)

            try:
                position = None
                if hasattr(file_obj, "seek"):
                    position = file_obj.tell()
                    file_obj.seek(0)
                payload = file_obj.read()
                if isinstance(payload, str):
                    payload = payload.encode("utf-8")
                with open(destination, "wb") as handle:
                    handle.write(payload)
                logger.debug("Materialized inline file %s into isolated workspace", sanitized_name)
            except Exception as exc:
                logger.debug("Failed to materialize inline file %s: %s", sanitized_name, exc)
            finally:
                if position is not None:
                    try:
                        file_obj.seek(position)
                    except Exception as exc:
                        logger.debug("Failed to restore inline file pointer %s: %s", sanitized_name, exc)

    def _materialize_file_store_contents(self, workspace_dir: str, store_paths: list[str] | None = None) -> None:
        """
        Mirror shared file-store artifacts into the transient workspace so direct open(...) calls succeed.
        """
        if not self.file_store:
            return

        target_paths: set[str]
        if store_paths:
            target_paths = set(store_paths)
        else:
            try:
                file_infos = self.file_store.list_files(recursive=True)
                target_paths = {info.path for info in file_infos}
            except Exception as exc:
                logger.debug("Failed to enumerate file store contents for workspace sync: %s", exc)
                return

        for relative_path in sorted(target_paths):
            try:
                payload = self.file_store.retrieve(relative_path)
            except Exception as exc:
                logger.debug("Failed to retrieve %s for workspace materialization: %s", relative_path, exc)
                continue

            absolute_path = os.path.join(workspace_dir, relative_path)
            os.makedirs(os.path.dirname(absolute_path) or workspace_dir, exist_ok=True)
            try:
                with open(absolute_path, "wb") as file_handle:
                    file_handle.write(payload)
                logger.debug("Materialized %s into isolated workspace", relative_path)
            except Exception as exc:
                logger.debug("Failed to materialize %s into workspace: %s", relative_path, exc)

    def _format_file_access_error(self, exc: Exception) -> str | None:
        """
        Provide a targeted hint when user code tries to open files that only exist in the shared file store.
        """
        is_missing_file = isinstance(exc, FileNotFoundError) or getattr(exc, "errno", None) == errno.ENOENT
        if not is_missing_file:
            return None

        missing_target = getattr(exc, "filename", None)
        missing_hint = f" '{missing_target}'" if missing_target else ""
        return (
            f"File access error: could not locate{missing_hint} inside the isolated workspace. "
            "Uploaded files are only available through the injected helpers "
            "(e.g., call read_file('your_file.csv') or helpers['read_file']('your_file.csv') instead of open())."
        )

    def _preload_optional_modules(self) -> dict[str, Any]:
        """Expose heavier-but-common modules (e.g., seaborn) inside the restricted globals."""
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
            "read_file_auto": file_manager.read,
            "write_text_file": file_manager.write_text,
            "write_bytes_file": file_manager.write_bytes,
            "write_file": file_manager.write,
            "delete_file": file_manager.delete,
            "file_exists": file_manager.exists,
            "describe_file": file_manager.describe,
        }

        alias_map = {
            "list_files": helper_funcs["list_files"],
            "read_file": helper_funcs["read_file_auto"],
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
