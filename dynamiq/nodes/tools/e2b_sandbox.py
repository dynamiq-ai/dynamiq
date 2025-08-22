import io
import shlex
from typing import Any, ClassVar, Literal

from e2b_code_interpreter import Sandbox
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from dynamiq.connections import E2B as E2BConnection
from dynamiq.nodes import NodeGroup
from dynamiq.nodes.agents.exceptions import ToolExecutionException
from dynamiq.nodes.node import ConnectionNode, ensure_config
from dynamiq.runnables import RunnableConfig
from dynamiq.utils.logger import logger

DESCRIPTION_E2B = """Executes Python code and shell commands in a secure cloud sandbox environment.
Provides isolated execution with package installation,
file upload, and persistent Python interpreter sessions for complex data analysis and system operations.

-Key Capabilities:-
- Execute Python code with stateful interpreter (variables persist between executions)
- Run shell commands and system operations with file system access
- Upload and process files (CSV, JSON, images) with persistent storage
- Maintain sandbox sessions across multiple executions for workflows

-Usage Strategy:-
Always use print() statements to display results - code without output will fail.
Specify required packages in the 'packages' parameter.
 Variables and imports persist between Python executions in the same session.

-Parameter Guide:-
- packages: Comma-separated list of Python packages to install
- python: Python code to execute (must include print statements)
- shell_command: Shell commands to run in the sandbox
- files: Binary files to upload for processing

-Examples:-
- Data analysis: {"python": "import pandas as pd\\ndf = pd.read_csv('/home/user/data/file.csv')\\nprint(df.head())"}
- Next execution: {"python": "print(df.describe())"}
- File processing: {"packages": "requests",
"python": "import requests\\nresponse = requests.get('https://api.example.com')\\nprint(response.json())"}
- System operations: {"shell_command": "ls -la /home/user && df -h"}"""


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


class FileData(BaseModel):
    """Model for file data with metadata."""
    data: bytes
    name: str
    description: str


def handle_file_upload(files: list[bytes | io.BytesIO | FileData]) -> list[FileData]:
    """
    Handles file uploading with additional metadata.

    Args:
        files: List of file objects to upload.

    Returns:
        list[FileData]: List of processed file data objects.

    Raises:
        ValueError: If invalid file data type is provided.
    """
    files_data = []
    for file in files:
        if isinstance(file, FileData):
            files_data.append(file)
        elif isinstance(file, bytes | io.BytesIO):
            file_name = getattr(file, "name", generate_fallback_filename(file))
            description = getattr(file, "description", generate_file_description(file))
            files_data.append(
                FileData(
                    data=file.getvalue() if isinstance(file, io.BytesIO) else file,
                    name=file_name,
                    description=description,
                )
            )
        else:
            raise ValueError(f"Error: Invalid file data type: {type(file)}. " f"Expected bytes or BytesIO or FileData.")

    return files_data


class E2BInterpreterInputSchema(BaseModel):
    """Input schema for E2B interpreter tool."""

    model_config = ConfigDict(extra="allow")

    packages: str = Field(default="", description="Comma-separated pip packages to install.")
    shell_command: str = Field(default="", description="Shell command to execute.")
    python: str = Field(default="", description="Python code to execute.")
    files: list[FileData] | None = Field(
        default=None,
        description="Files to upload to the sandbox.",
        json_schema_extra={"is_accessible_to_agent": False},
    )
    params: dict[str, Any] = Field(
        default_factory=dict, description="Arbitrary variables to inject as Python globals before executing 'python'."
    )
    env: dict[str, str] = Field(default_factory=dict, description="Environment variables for shell commands.")
    cwd: str = Field(default="/home/user", description="Working directory for shell commands.")
    timeout: int | None = Field(default=None, description="Override sandbox timeout for this execution (seconds)")

    @model_validator(mode="after")
    def validate_execution_commands(self):
        """Validate that either shell command or python code is specified."""
        if not self.shell_command and not self.python:
            raise ValueError("shell_command or python code has to be specified.")
        return self

    @field_validator("files", mode="before")
    @classmethod
    def files_validator(cls, files):
        """Validate and process files."""
        if files in (None, [], ()):
            return None
        return handle_file_upload(files)


class E2BInterpreterTool(ConnectionNode):
    """
    A tool for executing code and managing files in an E2B sandbox environment.

    This tool provides a secure execution environment for running Python code,
    shell commands, and managing file operations.

    Attributes:
        group: The node group identifier.
        name: The unique name of the tool.
        description: Detailed usage instructions and capabilities.
        connection: Configuration for E2B connection.
        installed_packages: Pre-installed packages in the sandbox.
        files: Files to be uploaded.
        persistent_sandbox: Whether to maintain sandbox between executions.
        is_files_allowed: Whether file uploads are permitted.
        _sandbox: Internal sandbox instance for persistent mode.
    """

    group: Literal[NodeGroup.TOOLS] = NodeGroup.TOOLS
    name: str = "E2b Code Interpreter Tool"
    description: str = DESCRIPTION_E2B
    connection: E2BConnection
    installed_packages: list = []
    files: list[FileData] | None = None
    persistent_sandbox: bool = True
    timeout: int = Field(default=600, description="Sandbox timeout in seconds (default: 600 seconds)")
    _sandbox: Sandbox | None = None
    is_files_allowed: bool = True
    input_schema: ClassVar[type[E2BInterpreterInputSchema]] = E2BInterpreterInputSchema

    def __init__(self, **kwargs):
        """Initialize the E2B interpreter tool."""
        super().__init__(**kwargs)
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
        self._sandbox = Sandbox(api_key=self.connection.api_key, timeout=self.timeout)
        self._install_default_packages(self._sandbox)
        if self.files:
            self._upload_files(files=self.files, sandbox=self._sandbox)

    def _install_default_packages(self, sandbox: Sandbox) -> None:
        """Install the default packages in the specified sandbox."""
        if self.installed_packages:
            for package in self.installed_packages:
                self._install_packages(sandbox, package)

    def _install_packages(self, sandbox: Sandbox, packages: str) -> None:
        """
        Install the specified packages in the given sandbox.

        Args:
            sandbox: The sandbox instance to install packages in.
            packages: Comma-separated string of package names.

        Raises:
            ToolExecutionException: If package installation fails.
        """
        if packages:
            logger.debug(f"Tool {self.name} - {self.id}: Installing packages: {packages}")
            try:
                process = sandbox.commands.run(f"pip install -qq {' '.join(packages.split(','))}")
            except Exception as e:
                raise ToolExecutionException(f"Error during package installation: {e}", recoverable=True)

            if process.exit_code != 0:
                raise ToolExecutionException(f"Error during package installation: {process.stderr}", recoverable=True)

    def _upload_files(self, files: list[FileData], sandbox: Sandbox) -> str:
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
            upload_details.append(
                {
                    "original_name": file.name,
                    "description": file.description,
                    "uploaded_path": uploaded_path,
                }
            )
        self._update_description_with_files(upload_details)
        return "\n".join([f"{file['original_name']} -> {file['uploaded_path']}" for file in upload_details])

    def _upload_file(self, file: FileData, sandbox: Sandbox | None = None) -> str:
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

        target_path = f"/home/user/data/{file.name}"
        dir_path = "/".join(target_path.split("/")[:-1])
        sandbox.commands.run(f"mkdir -p {shlex.quote(dir_path)}")

        file_like_object = io.BytesIO(file.data)
        file_like_object.name = file.name

        uploaded_info = sandbox.files.write(target_path, file_like_object)
        logger.debug(f"Tool {self.name} - {self.id}: Uploaded file info: {uploaded_info}")

        return uploaded_info.path

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

    def get_custom_vars(self, params: dict[str, Any]) -> str:
        """
        Generate custom variable assignment code from parameters.

        Args:
            params: Dictionary of variables to inject into the execution environment.

        Returns:
            str: Python code string for variable assignments.
        """
        if not params:
            return ""

        vars_code = "\n# Tool params variables injected by framework\n"
        for key, value in params.items():
            if isinstance(value, str):
                vars_code += f'{key} = "{value}"\n'
            elif isinstance(value, (int, float, bool)) or value is None:
                vars_code += f"{key} = {value}\n"
            elif isinstance(value, (list, dict)):
                vars_code += f"{key} = {value}\n"
            else:
                vars_code += f'{key} = "{str(value)}"\n'

        return vars_code

    def prepare_agent_output(self, content: dict[str, Any]) -> str:
        """
        Prepare formatted output for agent consumption.

        Args:
            content: Dictionary containing execution results.

        Returns:
            str: Formatted text output for agents.
        """
        result_text = ""

        if code_execution := content.get("code_execution"):
            result_text += "## Output\n\n" + code_execution + "\n\n"

        if shell_command_execution := content.get("shell_command_execution"):
            result_text += "## Shell Output\n\n" + shell_command_execution + "\n\n"

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
                result_text += f"*Files uploaded: {', '.join(files_list)}*\n\n"

        return result_text

    def _execute_python_code(self, code: str, sandbox: Sandbox | None = None, params: dict = None) -> str:
        """
        Execute Python code in the specified sandbox with persistent session state.

        Args:
            code: The Python code to execute.
            sandbox: The sandbox instance to execute code in.
            params: Variables to inject into the execution environment.

        Returns:
            str: The output from code execution.

        Raises:
            ValueError: If sandbox instance is not provided.
            ToolExecutionException: If code execution fails.
        """
        if not sandbox:
            raise ValueError("Sandbox instance is required for code execution.")

        if params:
            vars_code = self.get_custom_vars(params)
            code = vars_code + "\n" + code

        try:
            logger.info(f"Executing Python code: {code}")
            execution = sandbox.run_code(code)
            output_parts = []

            if execution.text:
                output_parts.append(execution.text)

            if execution.error:
                logger.debug(
                        f"Tool {self.name}: Error in persistent session: " f"{execution.error}"
                    )
                raise ToolExecutionException(f"Error during Python code execution: {execution.error}", recoverable=True)

            if hasattr(execution, "logs") and execution.logs:
                if hasattr(execution.logs, "stdout") and execution.logs.stdout:
                    for log in execution.logs.stdout:
                        output_parts.append(log)
                if hasattr(execution.logs, "stderr") and execution.logs.stderr:
                    for log in execution.logs.stderr:
                        output_parts.append(f"[stderr] {log}")

            return "\n".join(output_parts) if output_parts else ""

        except Exception as e:
            raise ToolExecutionException(f"Error during Python code execution: {e}", recoverable=True)

    def _execute_shell_command(
        self, command: str, sandbox: Sandbox | None = None, env: dict | None = None, cwd: str | None = None
    ) -> str:
        """
        Execute a shell command in the specified sandbox.

        Args:
            command: The shell command to execute.
            sandbox: The sandbox instance to execute command in.
            env: Environment variables for the command.
            cwd: Working directory for the command.

        Returns:
            str: The output from command execution.

        Raises:
            ValueError: If sandbox instance is not provided.
            ToolExecutionException: If command execution fails.
        """
        if not sandbox:
            raise ValueError("Sandbox instance is required for command execution.")

        try:
            process = sandbox.commands.run(command, background=True, envs=env or {}, cwd=cwd or "/home/user")
        except Exception as e:
            raise ToolExecutionException(f"Error during shell command execution: {e}", recoverable=True)

        output = process.wait()
        if output.exit_code != 0:
            raise ToolExecutionException(f"Error during shell command execution: {output.stderr}", recoverable=True)
        return output.stdout

    def execute(
        self, input_data: E2BInterpreterInputSchema, config: RunnableConfig | None = None, **kwargs
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
        logger.info(f"Tool {self.name} - {self.id}: started with input:\n" f"{input_data.model_dump()}")
        config = ensure_config(config)
        self.run_on_node_execute_run(config.callbacks, **kwargs)

        if self.persistent_sandbox and self._sandbox:
            sandbox = self._sandbox
        else:
            sandbox = Sandbox(api_key=self.connection.api_key, timeout=self.timeout)
            self._install_default_packages(sandbox)
            if self.files:
                self._upload_files(files=self.files, sandbox=sandbox)

        if input_data.timeout and sandbox:
            try:
                sandbox.set_timeout(input_data.timeout)
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
                    shell_command, sandbox=sandbox, env=input_data.env, cwd=input_data.cwd
                )

            if python := input_data.python:
                content["code_execution"] = self._execute_python_code(python, sandbox=sandbox, params=input_data.params)

            if not (packages or files or shell_command or python):
                raise ToolExecutionException(
                    "Error: Invalid input data. Please provide packages, files, shell_command, " "or python code.",
                    recoverable=True,
                )

            if python and not content.get("code_execution"):
                raise ToolExecutionException(
                    "Error: No output from Python execution. "
                    "Please use 'print()' to display the result of your Python code.",
                    recoverable=True,
                )

        finally:
            if not self.persistent_sandbox:
                logger.debug(f"Tool {self.name} - {self.id}: Closing Sandbox")
                sandbox.kill()

        if self.is_optimized_for_agents:
            result_text = self.prepare_agent_output(content)
            logger.info(f"Tool {self.name} - {self.id}: finished with result:\n" f"{str(result_text)[:200]}...")
            return {"content": result_text}

        return {"content": content}

    def set_timeout(self, timeout: int) -> None:
        """
        Update the timeout for the sandbox during runtime.

        Args:
            timeout: New timeout value in seconds.
        """
        self.timeout = timeout
        if self._sandbox and self.persistent_sandbox:
            try:
                self._sandbox.set_timeout(timeout)
                logger.debug(f"Tool {self.name} - {self.id}: Updated sandbox timeout to {timeout}s")
            except Exception as e:
                logger.warning(f"Tool {self.name} - {self.id}: Failed to update sandbox timeout: {e}")

    def close(self) -> None:
        """Close the persistent sandbox if it exists."""
        if self._sandbox and self.persistent_sandbox:
            logger.debug(f"Tool {self.name} - {self.id}: Closing Sandbox")
            self._sandbox.kill()
            self._sandbox = None
