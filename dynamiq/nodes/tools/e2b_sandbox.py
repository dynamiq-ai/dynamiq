import io
from hashlib import sha256
from typing import Any, ClassVar, Literal

from e2b_code_interpreter import Sandbox
from pydantic import BaseModel, Field, field_validator, model_validator

from dynamiq.connections import E2B as E2BConnection
from dynamiq.nodes import NodeGroup
from dynamiq.nodes.agents.exceptions import ToolExecutionException
from dynamiq.nodes.node import ConnectionNode, ensure_config
from dynamiq.runnables import RunnableConfig
from dynamiq.utils.logger import logger

DESCRIPTION_E2B = """Executes Python code and shell commands in a secure cloud sandbox environment.
Provides isolated execution with package installation,
file upload/download, and persistent sessions for complex data analysis and system operations.

-Key Capabilities:-
- Execute Python code with full library support and package installation
- Run shell commands and system operations with file system access
- Upload and process files (CSV, JSON, images) with persistent storage
- Maintain sandbox sessions across multiple executions for workflows

-Usage Strategy:-
Always use print() statements to display results - code without output will fail. Specify required packages in the 'packages' parameter. Handle errors gracefully and validate code syntax before execution.

-Parameter Guide:-
- packages: Comma-separated list of Python packages to install
- python: Python code to execute (must include print statements)
- shell_command: Shell commands to run in the sandbox
- files: Binary files to upload for processing

-Examples:-
- Data analysis: {"packages": "pandas,numpy", "python": "import pandas as pd\\ndata = pd.read_csv('data.csv')\\nprint(data.describe())"}
- File processing: {"packages": "requests", "python": "import requests\\nresponse = requests.get('https://api.example.com')\\nprint(response.json())"}
- System operations: {"shell_command": "ls -la /home/user && df -h"}"""  # noqa: E501


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
    data: bytes
    name: str
    description: str


def handle_file_upload(files: list[bytes | io.BytesIO | FileData]) -> list[FileData]:
    """Handles file uploading with additional metadata."""
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
            raise ValueError(f"Error: Invalid file data type: {type(file)}. Expected bytes or BytesIO or FileData.")

    return files_data


class E2BInterpreterInputSchema(BaseModel):
    packages: str = Field(default="", description="Parameter to provide packages to install.")
    shell_command: str = Field(default="", description="Parameter to provide shell command to execute.")
    python: str = Field(default="", description="Parameter to provide python code to execute.")

    files: list[FileData] = Field(
        default=None,
        description="Parameter to provide files for uploading to the sandbox.",
        json_schema_extra={"is_accessible_to_agent": False},
    )

    @model_validator(mode="after")
    def validate_execution_commands(self):
        """Validate that either shell command or python code is specified"""
        if not self.shell_command and not self.python:
            raise ValueError("shell_command or python code has to be specified.")
        return self

    @field_validator("files", mode="before")
    @classmethod
    def files_validator(cls, files: list[bytes | io.BytesIO | FileData]) -> list[FileData]:
        return handle_file_upload(files)


class E2BInterpreterTool(ConnectionNode):
    """
    A tool for executing code and managing files in an E2B sandbox environment.

    This tool provides a secure execution environment for running Python code,
    shell commands, and managing file operations.

    Attributes:
        group (Literal[NodeGroup.TOOLS]): The node group identifier.
        name (str): The unique name of the tool.
        description (str): Detailed usage instructions and capabilities.
        connection (E2BConnection): Configuration for E2B connection.
        installed_packages (List[str]): Pre-installed packages in the sandbox.
        files (Optional[List[Union[io.BytesIO, bytes]]]): Files to be uploaded.
        persistent_sandbox (bool): Whether to maintain sandbox between executions.
        is_files_allowed (bool): Whether file uploads are permitted.
        _sandbox (Optional[Sandbox]): Internal sandbox instance for persistent mode.
    """

    group: Literal[NodeGroup.TOOLS] = NodeGroup.TOOLS
    name: str = "E2b Code Interpreter Tool"
    description: str = DESCRIPTION_E2B
    connection: E2BConnection
    installed_packages: list = []
    files: list[FileData] | None = None
    persistent_sandbox: bool = True
    _sandbox: Sandbox | None = None
    is_files_allowed: bool = True
    input_schema: ClassVar[type[E2BInterpreterInputSchema]] = E2BInterpreterInputSchema

    @field_validator("files", mode="before")
    @classmethod
    def files_validator(cls, files: list[bytes | io.BytesIO | FileData]) -> list[FileData]:
        return handle_file_upload(files)

    def __init__(self, **kwargs):
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
            Dict[str, Any]: Dictionary representation of the instance.
        """
        data = super().to_dict(**kwargs)
        if self.files:
            data["files"] = [{"name": getattr(f, "name", f"file_{i}")} for i, f in enumerate(self.files)]
        return data

    def _initialize_persistent_sandbox(self):
        """Initializes the persistent sandbox, installs packages, and uploads initial files."""
        logger.debug(f"Tool {self.name} - {self.id}: Initializing Persistent Sandbox")
        self._sandbox = Sandbox(api_key=self.connection.api_key)
        self._install_default_packages(self._sandbox)
        if self.files:
            self._upload_files(files=self.files, sandbox=self._sandbox)

    def _install_default_packages(self, sandbox: Sandbox) -> None:
        """Installs the default packages in the specified sandbox."""
        if self.installed_packages:
            for package in self.installed_packages:
                self._install_packages(sandbox, package)

    def _install_packages(self, sandbox: Sandbox, packages: str) -> None:
        """Installs the specified packages in the given sandbox."""
        if packages:
            logger.debug(f"Tool {self.name} - {self.id}: Installing packages: {packages}")
            try:
                process = sandbox.commands.run(f"pip install -qq {' '.join(packages.split(','))}")
            except Exception as e:
                raise ToolExecutionException(f"Error during package installation: {e}", recoverable=True)

            if process.exit_code != 0:
                raise ToolExecutionException(f"Error during package installation: {process.stderr}", recoverable=True)

    def _upload_files(self, files: list[FileData], sandbox: Sandbox) -> str:
        """Uploads multiple files to the sandbox and returns details for each file."""
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
        """Uploads a single file to the specified sandbox and returns the uploaded path."""
        if not sandbox:
            raise ValueError("Sandbox instance is required for file upload.")

        # Handle the file types (bytes or io.BytesIO)
        file_like_object = io.BytesIO(file.data)
        file_like_object.name = file.name

        # Upload the file to the sandbox
        uploaded_info = sandbox.files.write(f"/home/user/{file.name}", file_like_object)
        logger.debug(f"Tool {self.name} - {self.id}: Uploaded file info: {uploaded_info}")

        return uploaded_info.path

    def _update_description_with_files(self, upload_details: list[dict]) -> None:
        """Updates the tool description with detailed information about the uploaded files."""
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

    def _execute_python_code(self, code: str, sandbox: Sandbox | None = None) -> str:
        """Executes Python code in the specified sandbox."""
        if not sandbox:
            raise ValueError("Sandbox instance is required for code execution.")
        code_hash = sha256(code.encode()).hexdigest()
        filename = f"/home/user/{code_hash}.py"
        sandbox.files.write(filename, code)
        try:
            process = sandbox.commands.run(f"python3 {filename}")
        except Exception as e:
            raise ToolExecutionException(f"Error during Python code execution: {e}", recoverable=True)

        if not (process.stdout or process.stderr):
            raise ToolExecutionException(
                "Error: No output. Please use 'print()' to display the result of your Python code.",
                recoverable=True,
            )
        if process.exit_code != 0:
            raise ToolExecutionException(f"Error during Python code execution: {process.stderr}", recoverable=True)
        return process.stdout

    def _execute_shell_command(self, command: str, sandbox: Sandbox | None = None) -> str:
        """Executes a shell command in the specified sandbox."""
        if not sandbox:
            raise ValueError("Sandbox instance is required for command execution.")

        try:
            process = sandbox.commands.run(command, background=True)
        except Exception as e:
            raise ToolExecutionException(f"Error during shell command execution: {e}", recoverable=True)

        output = process.wait()
        if output.exit_code != 0:
            raise ToolExecutionException(f"Error during shell command execution: {output.stderr}", recoverable=True)
        return output.stdout

    def execute(
        self, input_data: E2BInterpreterInputSchema, config: RunnableConfig | None = None, **kwargs
    ) -> dict[str, Any]:
        """Executes the requested action based on the input data."""
        logger.info(f"Tool {self.name} - {self.id}: started with input:\n{input_data.model_dump()}")
        config = ensure_config(config)
        self.run_on_node_execute_run(config.callbacks, **kwargs)

        if self.persistent_sandbox:
            sandbox = self._sandbox
        else:
            sandbox = Sandbox(api_key=self.connection.api_key)
            self._install_default_packages(sandbox)
            if self.files:
                self._upload_files(files=self.files, sandbox=sandbox)

        try:
            content = {}
            if files := input_data.files:
                content["files_installation"] = self._upload_files(files=files, sandbox=sandbox)
            if packages := input_data.packages:
                self._install_packages(sandbox=sandbox, packages=packages)
                content["packages_installation"] = f"Installed packages: {input_data.packages}"
            if shell_command := input_data.shell_command:
                content["shell_command_execution"] = self._execute_shell_command(shell_command, sandbox=sandbox)
            if python := input_data.python:
                content["code_execution"] = self._execute_python_code(python, sandbox=sandbox)
            if not (packages or files or shell_command or python):
                raise ToolExecutionException(
                    "Error: Invalid input data. Please provide packages, files, shell_command, or python code.",
                    recoverable=True,
                )

        finally:
            if not self.persistent_sandbox:
                logger.debug(f"Tool {self.name} - {self.id}: Closing Sandbox")
                sandbox.kill()

        if self.is_optimized_for_agents:
            result = ""
            if files_installation := content.get("files_installation"):
                result += "## Files Installation\n\n" + files_installation + "\n\n"
            if packages_installation := content.get("packages_installation"):
                result += "## Package Installation\n\n" + packages_installation + "\n\n"
            if shell_command_execution := content.get("shell_command_execution"):
                result += "## Shell Command Execution\n\n" + shell_command_execution + "\n\n"
            if code_execution := content.get("code_execution"):
                result += "## Code Execution\n\n" + code_execution + "\n\n"
            if result:
                content = result

        logger.info(f"Tool {self.name} - {self.id}: finished with result:\n{str(content)[:200]}...")
        return {"content": content}

    def close(self) -> None:
        """Closes the persistent sandbox if it exists."""
        if self._sandbox and self.persistent_sandbox:
            logger.debug(f"Tool {self.name} - {self.id}: Closing Sandbox")
            self._sandbox.kill()
            self._sandbox = None
