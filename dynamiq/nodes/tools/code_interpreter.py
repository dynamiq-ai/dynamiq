import abc
import base64
import io
import json
import shlex
from pathlib import PurePosixPath
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, field_validator, model_validator
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential_jitter

from dynamiq.nodes import NodeGroup
from dynamiq.nodes.agents.exceptions import ToolExecutionException
from dynamiq.nodes.node import ConnectionNode, ensure_config
from dynamiq.nodes.types import ActionType
from dynamiq.runnables import RunnableConfig
from dynamiq.storages.file.base import FileInfo
from dynamiq.utils.logger import logger

_DESCRIPTION_TEMPLATE = """Executes Python code and shell commands in a secure cloud sandbox environment.
Provides isolated execution with package installation,
file upload/download, and persistent Python interpreter sessions for complex data analysis and system operations.

-Key Capabilities:-
- Execute Python code with stateful interpreter (variables persist between executions)
- Run shell commands and system operations with file system access
- Upload and process files (CSV, JSON, images) with persistent storage
- Maintain sandbox sessions across multiple executions for workflows

-Usage Strategy:-
Always use print() statements to display results - code without output will fail.
Specify required packages in the 'packages' parameter.
 Variables and imports persist between Python executions in the same session.

-File Management Strategy:-
- Uploaded files are automatically saved to the {home_dir}/input directory
- If file name contains path (e.g., "data/file.csv"), it will be saved as {home_dir}/input/data/file.csv
- If file name is just a filename (e.g., "file.csv"), it will be saved as {home_dir}/input/file.csv
- If file name starts with "/" (e.g., "/file.csv"), it will be saved to the root directory as "/file.csv"
- To return files back to the user, save them in the {home_dir}/output directory
- Files saved in {home_dir}/output will be automatically collected and returned
- Use absolute paths like {home_dir}/output/filename.ext when saving files
- Access uploaded files from {home_dir}/input/filename.ext in your code

-Parameter Guide:-
- packages: Comma-separated list of Python packages to install
- python: Python code to execute (must include print statements)
- shell_command: Shell commands to run in the sandbox
- files: Binary files to upload for processing (saved to {home_dir}/input)

-Examples:-
- Data analysis: {"python": "import pandas as pd\\ndf = pd.read_csv('{home_dir}/input/data.csv')\\nprint(df.head())"}
- Next execution: {"python": "print(df.describe())"}
- File processing: {"packages": "requests",
"python": "import requests\\nresponse = requests.get('https://api.example.com')\\nprint(response.json())"}
- System operations: {"shell_command": "ls -la {home_dir}/input && ls -la {home_dir}/output"}
- File generation: {"python": "import pandas as pd\\ndf = pd.DataFrame({'A': [1,2,3], 'B': [4,5,6]})\\n\
df.to_csv('{home_dir}/output/result.csv')\\nprint('File saved to {home_dir}/output/result.csv')"}
- File upload scenarios:
  * "file.csv" → saved as {home_dir}/input/file.csv
  * "data/file.csv" → saved as {home_dir}/input/data/file.csv
  * "/file.csv" → saved as /file.csv (root directory)"""

DESCRIPTION_SANDBOX_INTERPRETER = _DESCRIPTION_TEMPLATE.replace("{home_dir}", "/home/user")


def detect_mime_type(file_content: bytes, file_path: str) -> str:
    """
    Detect MIME type using magic numbers and file extension.

    Args:
        file_content: The raw file content as bytes
        file_path: The file path to extract extension from

    Returns:
        str: The detected MIME type
    """
    magic_signatures = {
        # Images
        b"\x89PNG\r\n\x1a\n": "image/png",
        b"\xff\xd8\xff": "image/jpeg",
        b"GIF87a": "image/gif",
        b"GIF89a": "image/gif",
        b"RIFF": "image/webp",
        b"BM": "image/bmp",
        b"\x00\x00\x01\x00": "image/x-icon",
        # Documents
        b"%PDF": "application/pdf",
        b"PK\x03\x04": "application/zip",
        b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1": "application/vnd.ms-office",
        # Text/Data
        b"{\n": "application/json",
        b'{"': "application/json",
        b"[\n": "application/json",
        b"[{": "application/json",
    }

    for signature, mime_type in magic_signatures.items():
        if file_content.startswith(signature):
            if signature == b"RIFF" and len(file_content) > 12:
                if file_content[8:12] == b"WEBP":
                    return "image/webp"
                else:
                    continue
            return mime_type

    extension = file_path.lower().split(".")[-1] if "." in file_path else ""

    extension_map = {
        "png": "image/png",
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "gif": "image/gif",
        "webp": "image/webp",
        "bmp": "image/bmp",
        "ico": "image/x-icon",
        "svg": "image/svg+xml",
        "pdf": "application/pdf",
        "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "xls": "application/vnd.ms-excel",
        "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "doc": "application/msword",
        "pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        "ppt": "application/vnd.ms-powerpoint",
        "txt": "text/plain",
        "csv": "text/csv",
        "json": "application/json",
        "xml": "application/xml",
        "html": "text/html",
        "htm": "text/html",
        "css": "text/css",
        "js": "application/javascript",
        "md": "text/markdown",
        "zip": "application/zip",
        "tar": "application/x-tar",
        "gz": "application/gzip",
        "rar": "application/vnd.rar",
    }

    return extension_map.get(extension, "application/octet-stream")


def should_use_data_uri(mime_type: str) -> bool:
    """
    Determine if a file should be returned as a data URI.

    Args:
        mime_type: The MIME type of the file

    Returns:
        bool: True if should use data URI format
    """
    # Use data URIs for images and other web-renderable content
    data_uri_types = [
        "image/",
        "text/html",
        "text/css",
        "application/javascript",
        "image/svg+xml",
    ]

    return any(mime_type.startswith(prefix) for prefix in data_uri_types)


def generate_fallback_filename(file: bytes | io.BytesIO) -> str:
    """
    Generate a unique fallback filename for uploaded files.

    Args:
        file: File content as bytes or BytesIO object.

    Returns:
        str: A unique filename based on the object's id.
    """
    return f"uploaded_file_{id(file)}.bin"


def generate_file_description(file: bytes | io.BytesIO, length: int = 20) -> str:
    """
    Generate a description for a file based on its content.

    Args:
        file: File content as bytes or BytesIO object.
        length: Maximum number of bytes to include in the description.

    Returns:
        str: A description of the file's content or existing description.
    """
    if description := getattr(file, "description", None):
        return description

    file_content = file.getbuffer()[:length] if isinstance(file, io.BytesIO) else file[:length]
    return f"File starting with: {file_content.hex()}"


def handle_file_upload(files: list[bytes | io.BytesIO | FileInfo]) -> list[io.BytesIO]:
    """
    Handles file uploading and converts all inputs to BytesIO objects.

    Args:
        files: List of file objects to upload.

    Returns:
        list[io.BytesIO]: List of BytesIO objects with file data.

    Raises:
        ValueError: If invalid file data type is provided.
    """
    files_data = []
    for file in files:
        if isinstance(file, io.BytesIO):
            files_data.append(file)
        elif isinstance(file, bytes):
            file_name = getattr(file, "name", generate_fallback_filename(file))
            bytes_io = io.BytesIO(file)
            bytes_io.name = file_name
            files_data.append(bytes_io)
        elif isinstance(file, FileInfo):
            bytes_io = io.BytesIO(file.content)
            bytes_io.name = file.name
            files_data.append(bytes_io)
        else:
            raise ValueError(f"Error: Invalid file data type: {type(file)}. " f"Expected bytes or BytesIO or FileInfo.")

    return files_data


class SandboxCreationErrorHandling(BaseModel):
    max_retries: int = Field(default=5, description="Maximum number of creation attempts on rate-limit.")
    initial_wait_seconds: float = Field(default=2.0, description="Initial wait before retry.")
    max_wait_seconds: float = Field(default=32.0, description="Max wait used by exponential jitter backoff.")
    exponential_base: float = Field(
        default=2.0, description="Exponential base for backoff (2.0 means doubling each time)."
    )
    jitter: float = Field(default=1.0, description="Jitter factor to add randomness to retry timing.")


class CodeInterpreterInputSchema(BaseModel):
    """Input schema for sandbox interpreter tool."""

    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)

    packages: str = Field(default="", description="Comma-separated pip packages to install.")
    shell_command: str = Field(default="", description="Shell command to execute.")
    python: str = Field(default="", description="Python code to execute.")
    download_files: list[str] = Field(default_factory=list, description="Exact file paths to fetch as base64.")
    files: list[io.BytesIO] | None = Field(
        default=None,
        description="Files to upload to the sandbox.",
        json_schema_extra={"is_accessible_to_agent": False, "map_from_storage": True},
    )
    params: dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary variables to inject as Python globals before executing 'python'.",
        json_schema_extra={"is_accessible_to_agent": False},
    )
    env: dict[str, str] = Field(
        default_factory=dict,
        description="Environment variables for shell commands.",
        json_schema_extra={"is_accessible_to_agent": False},
    )
    cwd: str | None = Field(default=None, description="Working directory for shell commands.")
    timeout: int | None = Field(default=None, description="Override sandbox timeout for this execution (seconds)")

    @model_validator(mode="after")
    def validate_execution_commands(self):
        """Validate that either shell command, python code, or download files is specified."""
        if not self.shell_command and not self.python and not self.download_files:
            raise ValueError("shell_command, python code, or download_files has to be specified.")
        return self

    @field_validator("files", mode="before")
    @classmethod
    def files_validator(cls, files: list[bytes | io.BytesIO | FileInfo], **kwargs):
        """Validate and process files."""
        if files in (None, [], ()):
            return None

        return handle_file_upload(files)


class BaseCodeInterpreterTool(ConnectionNode, abc.ABC):
    """
    Abstract base class for sandbox interpreter tools.

    Provides shared logic for executing code and managing files in a sandbox environment.
    Subclasses must implement the abstract methods to delegate to their specific SDK.

    Attributes:
        group: The node group identifier.
        name: The unique name of the tool.
        description: Detailed usage instructions and capabilities.
        installed_packages: Pre-installed packages in the sandbox.
        files: Files to be uploaded.
        persistent_sandbox: Whether to maintain sandbox between executions.
        is_files_allowed: Whether file uploads are permitted.
        _sandbox: Internal sandbox instance for persistent mode.
    """

    _rate_limit_exception: ClassVar[type[Exception]]

    group: Literal[NodeGroup.TOOLS] = NodeGroup.TOOLS
    action_type: ActionType = ActionType.CODE_EXECUTION
    name: str = "sandbox-interpreter-tool"
    description: str = DESCRIPTION_SANDBOX_INTERPRETER
    base_path: str = Field(default="/home/user", description="Base path in the sandbox filesystem.")
    installed_packages: list = Field(default_factory=list, description="Pre-installed packages in the sandbox.")
    files: list[io.BytesIO] | None = None
    persistent_sandbox: bool = False
    timeout: int = Field(default=600, description="Sandbox timeout in seconds (default: 600 seconds)")
    is_files_allowed: bool = True
    creation_error_handling: SandboxCreationErrorHandling = Field(default_factory=SandboxCreationErrorHandling)

    _sandbox: Any = PrivateAttr(default=None)

    @property
    def input_dir(self) -> str:
        """Directory for uploaded input files."""
        return f"{self.base_path}/input"

    @property
    def output_dir(self) -> str:
        """Directory for output files."""
        return f"{self.base_path}/output"

    input_schema: ClassVar[type[CodeInterpreterInputSchema]] = CodeInterpreterInputSchema

    @abc.abstractmethod
    def _create_sandbox(self) -> Any:
        """Create a new sandbox instance."""
        ...

    @abc.abstractmethod
    def _get_sandbox_id(self, sandbox: Any) -> str:
        """Return the sandbox ID string."""
        ...

    @abc.abstractmethod
    def _get_sandbox_host(self, sandbox: Any) -> str | None:
        """Return the sandbox host for tracing. Return None if unavailable."""
        ...

    @abc.abstractmethod
    def _execute_python_code(
        self, code: str, sandbox: Any, params: dict | None = None, timeout: int | None = None
    ) -> str:
        """Execute Python code. Return stdout. Raise ToolExecutionException on error."""
        ...

    @abc.abstractmethod
    def _execute_shell_command(
        self, command: str, sandbox: Any, env: dict | None = None, cwd: str | None = None, timeout: int | None = None
    ) -> str:
        """Execute shell command. Return stdout. Raise ToolExecutionException on error."""
        ...

    @abc.abstractmethod
    def _install_packages(self, sandbox: Any, packages: str) -> None:
        """Install pip packages. Raise ToolExecutionException on failure."""
        ...

    @abc.abstractmethod
    def _upload_file_to_sandbox(self, file: io.BytesIO, target_path: str, sandbox: Any) -> str:
        """Upload a file to the given target_path. Return the actual uploaded path."""
        ...

    @abc.abstractmethod
    def _download_file_bytes(self, file_path: str, sandbox: Any) -> bytes:
        """Download a file as raw bytes."""
        ...

    @abc.abstractmethod
    def _run_shell_command(self, command: str, sandbox: Any) -> tuple[int, str]:
        """Run a shell command. Return (exit_code, stdout). Used for mkdir, find, etc."""
        ...

    @abc.abstractmethod
    def _destroy_sandbox(self, sandbox: Any) -> None:
        """Kill/delete the sandbox."""
        ...

    @abc.abstractmethod
    def close(self) -> None:
        """Close the persistent sandbox."""
        ...

    def __init__(self, **kwargs):
        """Initialize the sandbox interpreter tool."""
        super().__init__(**kwargs)
        if self.base_path != "/home/user":
            self.description = _DESCRIPTION_TEMPLATE.replace("{home_dir}", self.base_path)
        if self.persistent_sandbox and self.connection.api_key:
            self._initialize_persistent_sandbox()
        else:
            logger.debug(f"Tool {self.name} - {self.id}: Will initialize sandbox on each execute")

    @property
    def to_dict_exclude_params(self) -> set:
        """
        Get parameters to exclude from dictionary representation.

        Returns:
            set: Set of parameters to exclude.
        """
        return super().to_dict_exclude_params | {"files": True}

    def to_dict(self, **kwargs) -> dict[str, Any]:
        """
        Convert instance to dictionary format.

        Args:
            **kwargs: Additional keyword arguments.

        Returns:
            dict[str, Any]: Dictionary representation of the instance.
        """
        data = super().to_dict(**kwargs)
        if self.files:
            data["files"] = [{"name": getattr(f, "name", f"file_{i}")} for i, f in enumerate(self.files)]
        return data

    def _initialize_persistent_sandbox(self):
        """Initialize the persistent sandbox, install packages, and upload initial files."""
        logger.debug(f"Tool {self.name} - {self.id}: " f"Initializing Persistent Sandbox with timeout {self.timeout}s")
        self._sandbox = self._create_sandbox_with_retry()
        self._install_default_packages(self._sandbox)
        if self.files:
            self._upload_files(files=self.files, sandbox=self._sandbox)

    def _install_default_packages(self, sandbox: Any) -> None:
        """Install the default packages in the specified sandbox."""
        if self.installed_packages:
            for package in self.installed_packages:
                self._install_packages(sandbox, package)

    def _upload_files(self, files: list[io.BytesIO], sandbox: Any) -> str:
        """
        Upload multiple files to the sandbox and return details for each file.

        Args:
            files: List of file data objects to upload.
            sandbox: The sandbox instance to upload files to.

        Returns:
            str: Details of uploaded files.
        """
        upload_details = []
        for file in files:
            uploaded_path = self._upload_file(file, sandbox)
            file_name = getattr(file, "name", "unknown_file")
            upload_details.append(
                {
                    "original_name": file_name,
                    "description": getattr(file, "description", ""),
                    "uploaded_path": uploaded_path,
                }
            )
        self._update_description_with_files(upload_details)
        return "\n".join([f"{file['original_name']} -> {file['uploaded_path']}" for file in upload_details])

    def _resolve_param_value(self, value: Any, sandbox: Any) -> Any:
        """Recursively walk a param value, uploading any files and replacing them with paths."""
        if isinstance(value, FileInfo):
            return self._upload_file(value.to_bytesio(), sandbox)
        if isinstance(value, io.BytesIO):
            return self._upload_file(value, sandbox)
        if isinstance(value, list):
            return [self._resolve_param_value(item, sandbox) for item in value]
        if isinstance(value, tuple):
            return tuple(self._resolve_param_value(item, sandbox) for item in value)
        if isinstance(value, dict):
            return {k: self._resolve_param_value(v, sandbox) for k, v in value.items()}
        return value

    def _upload_file(self, file: io.BytesIO, sandbox: Any = None) -> str:
        """
        Upload a single file to the specified sandbox and return the uploaded path.

        Args:
            file: The file data to upload.
            sandbox: The sandbox instance to upload to.

        Returns:
            str: The path where the file was uploaded.

        Raises:
            ValueError: If sandbox instance is not provided.
        """
        if not sandbox:
            raise ValueError("Sandbox instance is required for file upload.")

        file_name = getattr(file, "name", "unknown_file")

        if "/" in file_name:
            dir_path = "/".join(file_name.split("/")[:-1])
            self._run_shell_command(f"mkdir -p {self.input_dir}/{shlex.quote(dir_path)}", sandbox)

        # Reset file position to beginning
        file.seek(0)

        target_path = (
            f"{self.input_dir}/{file_name}" if not file_name.startswith("/") else f"{self.input_dir}{file_name}"
        )
        return self._upload_file_to_sandbox(file, target_path, sandbox)

    def _update_description_with_files(self, upload_details: list[dict]) -> None:
        """
        Update the tool description with detailed information about the uploaded files.

        Args:
            upload_details: List of dictionaries containing file upload details.
        """
        if upload_details:
            self.description = self.description.strip().replace("</tool_description>", "")
            self.description += "\n\n**Uploaded Files Details:**"
            for file_info in upload_details:
                self.description += (
                    f"\n- **Original File Name**: {file_info['original_name']}\n"
                    f"  **Description**: {file_info['description']}\n"
                    f"  **Uploaded Path**: {file_info['uploaded_path']}\n"
                )
            self.description += "\n</tool_description>"

    def _download_files(self, file_paths: list[str], sandbox: Any = None) -> dict[str, str]:
        """
        Download files from sandbox and return them with proper MIME types and data URIs.

        Args:
            file_paths: List of file paths to download.
            sandbox: The sandbox instance to download from.

        Returns:
            dict[str, str]: Dictionary mapping file paths to base64 or data URI content.

        Raises:
            ValueError: If sandbox instance is not provided.
        """
        if not sandbox:
            raise ValueError("Sandbox instance is required for file download.")

        downloaded_files = {}
        for file_path in file_paths:
            try:
                file_bytes = self._download_file_bytes(file_path, sandbox)

                base64_content = base64.b64encode(file_bytes).decode("utf-8")

                downloaded_files[file_path] = base64_content

            except Exception as e:
                logger.warning(f"Tool {self.name} - {self.id}: Failed to download {file_path}: {e}")
                downloaded_files[file_path] = f"Error: {str(e)}"

        return downloaded_files

    def _is_simple_structure(self, obj: Any, max_depth: int = 3) -> bool:
        """
        Check if object contains only simple, serializable types.

        Args:
            obj: The object to check.
            max_depth: Maximum depth to check for nested structures.

        Returns:
            bool: True if object contains only simple types.
        """
        if max_depth <= 0:
            return False
        if isinstance(obj, (str, int, float, bool, type(None))):
            return True
        elif isinstance(obj, list):
            return all(self._is_simple_structure(item, max_depth - 1) for item in obj[:10])  # Limit list size
        elif isinstance(obj, dict):
            return all(
                isinstance(k, str) and self._is_simple_structure(v, max_depth - 1)
                for k, v in list(obj.items())[:10]  # Limit dict size
            )
        else:
            return False

    def _collect_output_files(self, sandbox: Any, base_dir: str = "") -> dict[str, str]:
        """
        Collect common output files from the output directory.

        Args:
            sandbox: The sandbox instance to collect files from.
            base_dir: Base directory to search for files.

        Returns:
            dict[str, str]: Dictionary mapping file paths to base64 or data URI content.
        """
        try:
            collected_files = {}

            search_dirs = [self.output_dir]

            for search_dir in search_dirs:
                check_cmd = f"test -d {shlex.quote(search_dir)} && echo exists"
                check_exit_code, check_stdout = self._run_shell_command(check_cmd, sandbox)

                if check_exit_code != 0 or "exists" not in check_stdout:
                    continue

                max_depth = "3"
                cmd = (
                    f"cd {shlex.quote(search_dir)} && find . -maxdepth {max_depth} "
                    f"-type f -printf '%P\\n' 2>/dev/null | head -20"
                )
                exit_code, stdout = self._run_shell_command(cmd, sandbox)

                if exit_code != 0 or not stdout.strip():
                    continue

                file_paths = [f for f in stdout.splitlines() if f.strip()]
                if file_paths:
                    abs_paths = [str(PurePosixPath(search_dir) / p) for p in file_paths]
                    files = self._download_files(abs_paths, sandbox=sandbox)
                    collected_files.update(files)

            return collected_files

        except Exception as e:
            logger.warning(f"Failed to collect output files: {e}")
            return {}

    def execute(
        self, input_data: CodeInterpreterInputSchema, config: RunnableConfig | None = None, **kwargs
    ) -> dict[str, Any]:
        """
        Execute the requested action based on the input data.

        Args:
            input_data: The input schema containing execution parameters.
            config: Optional runnable configuration.
            **kwargs: Additional keyword arguments.

        Returns:
            dict[str, Any]: Dictionary containing execution results.

        Raises:
            ToolExecutionException: If execution fails or invalid input provided.
        """
        logger.info(f"Tool {self.name} - {self.id}: started with input:\n" f"{str(input_data.model_dump())[:300]}")
        config = ensure_config(config)

        if self.persistent_sandbox and self._sandbox:
            sandbox = self._sandbox
        else:
            sandbox = self._create_sandbox_with_retry()
            self._install_default_packages(sandbox)
            if self.files:
                self._upload_files(files=self.files, sandbox=sandbox)

        tool_data = {
            "tool_session_id": self._get_sandbox_id(sandbox),
            "tool_session_host": self._get_sandbox_host(sandbox),
        }
        self.run_on_node_execute_run(
            config.callbacks,
            tool_data=tool_data,
            **kwargs,
        )

        if sandbox and self.is_files_allowed:
            try:
                self._run_shell_command(f"mkdir -p {self.input_dir} {self.output_dir}", sandbox)
                logger.debug(f"Created {self.input_dir} and {self.output_dir} directories")
            except Exception as e:
                logger.warning(f"Failed to create directories: {e}")

        if input_data.timeout and sandbox:
            try:
                self.set_timeout(input_data.timeout)
                logger.debug(f"Set per-call timeout to {input_data.timeout}s")
            except Exception as e:
                logger.warning(f"Failed to set per-call timeout: {e}")

        try:
            content = {}

            if files := input_data.files:
                content["files_uploaded"] = self._upload_files(files=files, sandbox=sandbox)

            if packages := input_data.packages:
                self._install_packages(sandbox=sandbox, packages=packages)
                content["packages_installation"] = f"Installed packages: {input_data.packages}"

            if shell_command := input_data.shell_command:
                content["shell_command_execution"] = self._execute_shell_command(
                    shell_command,
                    sandbox=sandbox,
                    env=input_data.env,
                    cwd=input_data.cwd or self.output_dir,
                    timeout=input_data.timeout,
                )

            if python := input_data.python:
                content["code_execution"] = self._execute_python_code(
                    python,
                    sandbox=sandbox,
                    params=input_data.params,
                    timeout=input_data.timeout,
                )

            if download_files := input_data.download_files:
                downloaded_files = self._download_files(download_files, sandbox=sandbox)
                content.setdefault("files", {}).update(downloaded_files)

            if shell_command or python:
                collected_files = self._collect_output_files(sandbox)
                if collected_files:
                    content.setdefault("files", {}).update(collected_files)

            if not (packages or files or shell_command or python or download_files):
                raise ToolExecutionException(
                    "Error: Invalid input data. Please provide packages, files, shell_command, "
                    "python code, or download_files.",
                    recoverable=True,
                )

            if python and not content.get("code_execution") and not content.get("files"):
                raise ToolExecutionException(
                    "Error: No output from Python execution. "
                    "Please use 'print()' to display the result of your Python code.",
                    recoverable=True,
                )

        finally:
            if not self.persistent_sandbox:
                logger.debug(f"Tool {self.name} - {self.id}: Closing Sandbox")
                self._destroy_sandbox(sandbox)

        if self.is_optimized_for_agents:
            result_text = ""

            if code_execution := content.get("code_execution"):
                result_text += "## Output\n\n" + code_execution + "\n\n"

            if shell_command_execution := content.get("shell_command_execution"):
                result_text += "## Shell Output\n\n" + shell_command_execution + "\n\n"

            all_files = content.get("files", {})

            uploaded_files = set()
            if files_uploaded := content.get("files_uploaded"):
                for line in files_uploaded.split("\n"):
                    if " -> " in line:
                        uploaded_path = line.split(" -> ")[1].strip()
                        uploaded_files.add(uploaded_path)

            new_files = {k: v for k, v in all_files.items() if k not in uploaded_files}

            # Convert files to BytesIO objects for proper storage handling
            files_bytesio = []
            if new_files:
                result_text += "## Generated Files (ready for download)\n\n"
                for file_path, file_content in new_files.items():
                    if file_content.startswith("Error:"):
                        result_text += f"- {file_path}: {file_content}\n"
                    else:
                        try:
                            # Decode content to bytes
                            if file_content.startswith("data:"):
                                # Handle data URI format
                                mime_part = file_content.split(";")[0].replace("data:", "")
                                base64_part = file_content.split(",", 1)[1]
                                content_bytes = base64.b64decode(base64_part)
                                content_type = mime_part
                            else:
                                # Handle plain base64 content
                                content_bytes = base64.b64decode(file_content)
                                content_type = detect_mime_type(content_bytes, file_path)

                            file_name = file_path.split("/")[-1]
                            file_size = len(content_bytes)
                            result_text += f"- **{file_name}** ({file_size:,} bytes, {content_type})\n"

                            # Create BytesIO object with metadata
                            file_bytesio = io.BytesIO(content_bytes)
                            file_bytesio.name = file_name
                            file_bytesio.description = f"Generated file from sandbox: {file_path}"
                            file_bytesio.content_type = content_type

                            # Ensure the BytesIO object is positioned at the beginning for reading
                            file_bytesio.seek(0)

                            files_bytesio.append(file_bytesio)

                        except (base64.binascii.Error, ValueError, Exception) as e:
                            error_msg = f"Failed to decode file {file_path}: {str(e)}"
                            result_text += f"- {file_path}: {error_msg}\n"
                            logger.warning(f"Tool {self.name} - {self.id}: {error_msg}")

                result_text += "\n"

            if packages_installation := content.get("packages_installation"):
                packages = packages_installation.replace("Installed packages: ", "")
                if packages:
                    result_text += f"*Packages installed: {packages}*\n\n"

            if files_uploaded := content.get("files_uploaded"):
                files_list = []
                for line in files_uploaded.split("\n"):
                    if " -> " in line:
                        file_name = line.split(" -> ")[0].strip()
                        files_list.append(file_name)
                if files_list:
                    result_text += f"*Files uploaded: {', '.join(files_list)}*\n"
                    result_text += f"Note: Uploaded files are available under {self.input_dir}. "
            logger.info(f"Tool {self.name} - {self.id}: finished with result:\n" f"{str(result_text)[:200]}...")

            return {"content": result_text, "files": files_bytesio}

        if code_execution := content.get("code_execution"):
            try:
                parsed = json.loads(code_execution)
                if isinstance(parsed, (dict, list)):
                    content["code_execution"] = parsed
            except (json.JSONDecodeError, ValueError):
                pass

        logger.info(f"Tool {self.name} - {self.id}: finished with result:\n" f"{str(content)[:200]}...")
        return {"content": content}

    def _create_sandbox_with_retry(self) -> Any:
        """Create sandbox with tenacity retry on rate-limit responses.

        Uses exponential backoff strategy for rate limit errors with configuration
        from the node's error_handling settings.
        """

        @retry(
            retry=retry_if_exception_type(self._rate_limit_exception),
            stop=stop_after_attempt(self.creation_error_handling.max_retries),
            wait=wait_exponential_jitter(
                initial=self.creation_error_handling.initial_wait_seconds,
                max=self.creation_error_handling.max_wait_seconds,
                exp_base=self.creation_error_handling.exponential_base,
                jitter=self.creation_error_handling.jitter,
            ),
            reraise=True,
        )
        def create_sandbox():
            try:
                sandbox = self._create_sandbox()
                logger.debug(f"Tool {self.name} - {self.id}: Successfully created sandbox")
                return sandbox
            except self._rate_limit_exception:
                logger.warning(
                    f"Tool {self.name} - {self.id}: Sandbox creation rate-limited. "
                    f"Retrying with exponential backoff."
                )
                raise
            except Exception:
                raise

        return create_sandbox()

    def set_timeout(self, timeout: int) -> None:
        """
        Update the timeout for the sandbox during runtime.

        Args:
            timeout: New timeout value in seconds.
        """
        self.timeout = timeout
