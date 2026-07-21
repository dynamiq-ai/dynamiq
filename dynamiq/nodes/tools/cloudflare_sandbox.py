import io
import shlex
from typing import Any, ClassVar
from uuid import uuid4

from dynamiq.connections import Cloudflare as CloudflareConnection
from dynamiq.connections.cloudflare_sandbox import (
    CloudflareExecResult,
    CloudflareRateLimitError,
    CloudflareSandboxInstance,
)
from dynamiq.nodes.agents.exceptions import ToolExecutionException
from dynamiq.nodes.tools.code_interpreter import BaseCodeInterpreterTool
from dynamiq.utils.logger import logger

DESCRIPTION_CLOUDFLARE_INTERPRETER = """Executes Python code and shell commands in a secure cloud sandbox environment.
Provides isolated execution with package installation,
file upload/download, and a persistent filesystem for complex data analysis and system operations.

-Key Capabilities:-
- Execute Python code (each execution runs in a fresh python3 process)
- Run shell commands and system operations with file system access
- Upload and process files (CSV, JSON, images) with persistent storage
- Files persist across executions for the lifetime of the sandbox

-Usage Strategy:-
Always use print() statements to display results - code without output will fail.
Specify required packages in the 'packages' parameter.
 Variables do NOT persist between Python executions - save intermediate results to files under /workspace.

-File Management Strategy:-
- Uploaded files are automatically saved to the /workspace/input directory
- If file name contains path (e.g., "data/file.csv"), it will be saved as /workspace/input/data/file.csv
- If file name is just a filename (e.g., "file.csv"), it will be saved as /workspace/input/file.csv
- To return files back to the user, save them in the /workspace/output directory
- Files saved in /workspace/output will be automatically collected and returned
- Use absolute paths like /workspace/output/filename.ext when saving files
- Access uploaded files from /workspace/input/filename.ext in your code
- All file paths must stay inside /workspace

-Parameter Guide:-
- packages: Comma-separated list of Python packages to install
- python: Python code to execute (must include print statements)
- shell_command: Shell commands to run in the sandbox
- files: Binary files to upload for processing (saved to /workspace/input)

-Examples:-
- Data analysis: {"python": "import pandas as pd\\ndf = pd.read_csv('/workspace/input/data.csv')\\nprint(df.head())"}
- File processing: {"packages": "requests",
"python": "import requests\\nresponse = requests.get('https://api.example.com')\\nprint(response.json())"}
- System operations: {"shell_command": "ls -la /workspace/input && ls -la /workspace/output"}
- File generation: {"python": "import pandas as pd\\ndf = pd.DataFrame({'A': [1,2,3], 'B': [4,5,6]})\\n\
df.to_csv('/workspace/output/result.csv')\\nprint('File saved to /workspace/output/result.csv')"}"""


class CloudflareInterpreterTool(BaseCodeInterpreterTool):
    """Cloudflare Sandboxes implementation of the sandbox interpreter tool.

    Talks plain HTTP to a deployed Cloudflare sandbox bridge Worker (see
    ``dynamiq.connections.cloudflare_sandbox``). Python code is written to a
    temporary file inside the sandbox and executed with ``python3``, so each
    execution runs in a fresh process; the /workspace filesystem persists for
    the sandbox lifetime.
    """

    name: str = "cloudflare-code-interpreter-tool"
    description: str = DESCRIPTION_CLOUDFLARE_INTERPRETER
    base_path: str = "/workspace"
    connection: CloudflareConnection
    _sandbox: CloudflareSandboxInstance | None = None
    _rate_limit_exception: ClassVar[type[Exception]] = CloudflareRateLimitError

    def _create_sandbox(self) -> CloudflareSandboxInstance:
        client = self.connection.get_client()
        return CloudflareSandboxInstance(client, client.create_sandbox())

    def _get_sandbox_id(self, sandbox: CloudflareSandboxInstance) -> str:
        return sandbox.sandbox_id

    def _get_sandbox_host(self, sandbox: CloudflareSandboxInstance) -> str | None:
        return None

    def _raise_on_failure(self, result: CloudflareExecResult, action: str) -> str:
        if result.exit_code != 0:
            raise ToolExecutionException(f"Error during {action}: {result.stderr or result.stdout}", recoverable=True)
        output_parts = []
        if result.stdout:
            output_parts.append(result.stdout)
        if result.stderr:
            output_parts.append(f"[stderr] {result.stderr}")
        return "\n".join(output_parts)

    def _execute_python_code(
        self, code: str, sandbox: Any, params: dict | None = None, timeout: int | None = None
    ) -> str:
        if not sandbox:
            raise ValueError("Sandbox instance is required for code execution.")

        if params:
            vars_code = "\n# Tool params variables\n"
            for key, value in params.items():
                resolved = self._resolve_param_value(value, sandbox)
                vars_code += f"{key} = {repr(resolved)}\n"
            code = vars_code + "\n" + code

        script_path = f"{self.base_path}/.dynamiq/script_{uuid4().hex}.py"
        try:
            logger.info(f"Executing Python code: {code}")
            sandbox.write_file(script_path, code.encode("utf-8"))
            result = sandbox.exec(f"python3 {shlex.quote(script_path)}", timeout=timeout or self.timeout)
            return self._raise_on_failure(result, "Python code execution")
        except ToolExecutionException:
            raise
        except Exception as e:
            raise ToolExecutionException(f"Error during Python code execution: {e}", recoverable=True)
        finally:
            try:
                sandbox.exec(f"rm -f {shlex.quote(script_path)}")
            except Exception as e:
                logger.debug(f"Tool {self.name} - {self.id}: Failed to clean up {script_path}: {e}")

    def _execute_shell_command(
        self,
        command: str,
        sandbox: Any,
        env: dict | None = None,
        cwd: str | None = None,
        timeout: int | None = None,
    ) -> str:
        if not sandbox:
            raise ValueError("Sandbox instance is required for command execution.")

        try:
            result = sandbox.exec(command, cwd=cwd, env=env or None, timeout=timeout or self.timeout)
        except ToolExecutionException:
            raise
        except Exception as e:
            raise ToolExecutionException(f"Error during shell command execution: {e}", recoverable=True)

        return self._raise_on_failure(result, "shell command execution")

    def _install_packages(self, sandbox: Any, packages: str) -> None:
        if packages:
            logger.debug(f"Tool {self.name} - {self.id}: Installing packages: {packages}")
            try:
                result = sandbox.exec(f"pip install -qq {' '.join(packages.split(','))}", timeout=self.timeout)
            except Exception as e:
                raise ToolExecutionException(f"Error during package installation: {e}", recoverable=True)

            if result.exit_code != 0:
                raise ToolExecutionException(
                    f"Error during package installation: {result.stderr or result.stdout}", recoverable=True
                )

    def _upload_file_to_sandbox(self, file: io.BytesIO, target_path: str, sandbox: Any) -> str:
        sandbox.write_file(target_path, file.read())
        logger.debug(f"Tool {self.name} - {self.id}: Uploaded file to: {target_path}")
        return target_path

    def _download_file_bytes(self, file_path: str, sandbox: Any) -> bytes:
        return sandbox.read_file(file_path)

    def _run_shell_command(self, command: str, sandbox: Any) -> tuple[int, str]:
        result = sandbox.exec(command)
        return result.exit_code if result.exit_code is not None else 1, result.stdout or ""

    def _reconnect_sandbox(self, sandbox_id: str) -> CloudflareSandboxInstance:
        client = self.connection.get_client()
        sandbox = CloudflareSandboxInstance(client, sandbox_id)
        if not client.is_running(sandbox_id):
            raise ValueError(f"Cloudflare sandbox {sandbox_id} is not reachable")
        return sandbox

    def _destroy_sandbox(self, sandbox: Any) -> None:
        sandbox.destroy()

    def close(self) -> None:
        """Close the persistent sandbox if it exists."""
        if self._sandbox and self.persistent_sandbox:
            logger.debug(f"Tool {self.name} - {self.id}: Closing Sandbox")
            try:
                self._sandbox.destroy()
            except Exception as e:
                logger.warning(f"Tool {self.name} - {self.id}: Failed to destroy sandbox: {e}")
            self._sandbox = None
