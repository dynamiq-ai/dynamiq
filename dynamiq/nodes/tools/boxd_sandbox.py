import io
import shlex
import uuid
from typing import Any, ClassVar

from boxd import Machine, RateLimitError
from pydantic import Field

from dynamiq.connections import Boxd as BoxdConnection
from dynamiq.nodes.agents.exceptions import ToolExecutionException
from dynamiq.nodes.tools.code_interpreter import (
    DESCRIPTION_SANDBOX_INTERPRETER,
    BaseCodeInterpreterTool,
    shell_double_quote,
    split_pip_requirements,
    tail_of_output,
)
from dynamiq.sandboxes.boxd import (
    BOXD_HOME,
    DEFAULT_MACHINE_NAME_PREFIX,
    boxd_shell_argv,
    connect_boxd_machine,
    create_boxd_machine,
)
from dynamiq.utils.logger import logger

# A machine created for a single call is deleted at the end of the call. This timer deletes it as well if the call
# never gets there, for example when the process dies. It covers creation, package installs and the command.
MIN_CALL_MACHINE_DELETE_AFTER = 3600
CALL_MACHINE_DELETE_AFTER_TIMEOUTS = 4

# Each Python execution runs in a new process, so the description does not promise variables that persist.
_STATELESS_PYTHON_LINES = {
    "and persistent Python interpreter sessions for complex data analysis and system operations.": (
        "and a Python process per execution for data analysis and system operations."
    ),
    "- Execute Python code with stateful interpreter (variables persist between executions)": (
        "- Execute Python code; each execution runs in a new Python process"
    ),
    " Variables and imports persist between Python executions in the same session.": (
        " Each Python execution starts a new process: write results you need later to files."
    ),
    '- Next execution: {"python": "print(df.describe())"}': (
        '- Next execution: {"python": "import pandas as pd\\ndf = pd.read_csv(\'/home/user/input/data.csv\')\\n'
        'print(df.describe())"}'
    ),
}


def _boxd_description(home_dir: str) -> str:
    description = DESCRIPTION_SANDBOX_INTERPRETER
    for old, new in _STATELESS_PYTHON_LINES.items():
        description = description.replace(old, new)
    return description.replace("/home/user", home_dir)


DESCRIPTION_BOXD_INTERPRETER = _boxd_description(BOXD_HOME)


def _output_with_stderr(stdout: str, stderr: str) -> str:
    output_parts = [part for part in (stdout, f"[stderr] {stderr}" if stderr else "") if part]
    return "\n".join(output_parts)


class BoxdInterpreterTool(BaseCodeInterpreterTool):
    """boxd implementation of the sandbox interpreter tool.

    Runs Python and shell commands in an isolated boxd machine. Packages go to a virtualenv on the machine, which
    the Python code and shell commands use.
    """

    name: str = "boxd-code-interpreter-tool"
    description: str = DESCRIPTION_BOXD_INTERPRETER
    base_path: str = BOXD_HOME
    connection: BoxdConnection
    timeout: int = Field(
        default=600,
        description="Seconds to wait for a machine to accept commands, and the limit of a command without its own.",
    )
    snapshot: str | None = Field(
        default=None,
        description="Name of a boxd snapshot to create machines from. If None, the default image is used.",
    )
    labels: dict[str, str] | None = Field(default=None, description="Custom labels attached to new machines.")
    name_prefix: str = Field(
        default=DEFAULT_MACHINE_NAME_PREFIX,
        description="Prefix of machine names. A random suffix makes each name unique.",
    )
    vcpu: int | None = Field(default=None, description="vCPUs of new machines. If None, the org default is used.")
    memory: str | None = Field(
        default=None, description="Memory of new machines, for example '8G'. If None, the org default is used."
    )
    auto_suspend_timeout: int | None = Field(
        default=0,
        description=(
            "Seconds without network traffic before a machine goes to standby. 0 (the default) disables it. "
            "If None, the org default is used."
        ),
    )
    delete_after: int | None = Field(
        default=None,
        description=(
            "Seconds after creation when boxd deletes a machine and its disk, even while it is in use. 0 sets no "
            "timer. If None, a persistent machine has no timer, and a machine for a single call gets one hour or "
            "four times timeout, whichever is longer."
        ),
    )
    _sandbox: Machine | None = None
    _rate_limit_exception: ClassVar[type[Exception]] = RateLimitError

    def __init__(self, **kwargs):
        """Initialize the tool, pointing the default description at a custom base path."""
        super().__init__(**kwargs)
        if self.base_path != BOXD_HOME and self.description == DESCRIPTION_BOXD_INTERPRETER:
            self.description = _boxd_description(self.base_path)

    @property
    def _client(self) -> Any:
        return self.connection.get_client()

    def _resolve_delete_after(self) -> int | None:
        if self.delete_after is not None:
            return self.delete_after or None
        if self.persistent_sandbox:
            return None
        return max(MIN_CALL_MACHINE_DELETE_AFTER, CALL_MACHINE_DELETE_AFTER_TIMEOUTS * self.timeout)

    def _create_sandbox(self) -> Machine:
        return create_boxd_machine(
            self.connection,
            name_prefix=self.name_prefix,
            snapshot=self.snapshot,
            labels=self.labels,
            vcpu=self.vcpu,
            memory=self.memory,
            auto_suspend_timeout=self.auto_suspend_timeout,
            delete_after=self._resolve_delete_after(),
            ready_timeout=self.timeout,
        )

    def _get_sandbox_id(self, sandbox: Machine) -> str:
        return sandbox.id

    def _get_sandbox_host(self, sandbox: Machine) -> str | None:
        return None

    def _exec(
        self,
        sandbox: Machine,
        command: str,
        env: dict | None = None,
        cwd: str | None = None,
        timeout: int | None = None,
    ):
        # A call without a timeout of its own gets the tool's, so a hung command never blocks the agent for good.
        return self._client.machines.exec(
            sandbox.id, boxd_shell_argv(command), env=env, cwd=cwd, timeout=timeout or self.timeout
        )

    def _execute_python_code(
        self, code: str, sandbox: Machine, params: dict | None = None, timeout: int | None = None
    ) -> str:
        if not sandbox:
            raise ValueError("Sandbox instance is required for code execution.")

        if params:
            vars_code = "\n# Tool params variables\n"
            for key, value in params.items():
                resolved = self._resolve_param_value(value, sandbox)
                vars_code += f"{key} = {repr(resolved)}\n"
            code = vars_code + "\n" + code

        # A path on the machine, not on this host, with a random name.
        script_path = f"/tmp/dynamiq-{uuid.uuid4().hex}.py"  # nosec B108
        try:
            logger.debug(f"Executing Python code: {code}")
            self._client.machines.files.upload(sandbox.id, script_path, code)
            quoted_path = shlex.quote(script_path)
            result = self._exec(
                sandbox,
                f"python {quoted_path}; status=$?; rm -f {quoted_path}; exit $status",
                cwd=self.base_path,
                timeout=timeout,
            )
        except Exception as e:
            raise ToolExecutionException(f"Error during Python code execution: {e}", recoverable=True)

        if result.exit_code != 0:
            raise ToolExecutionException(
                f"Error during Python code execution: {tail_of_output(result.stderr or result.stdout or '')}",
                recoverable=True,
            )
        return _output_with_stderr(result.stdout, result.stderr)

    def _execute_shell_command(
        self,
        command: str,
        sandbox: Machine,
        env: dict | None = None,
        cwd: str | None = None,
        timeout: int | None = None,
    ) -> str:
        if not sandbox:
            raise ValueError("Sandbox instance is required for command execution.")

        try:
            result = self._exec(sandbox, command, env=env or None, cwd=cwd, timeout=timeout)
        except Exception as e:
            raise ToolExecutionException(f"Error during shell command execution: {e}", recoverable=True)

        if result.exit_code != 0:
            pairs = (("stderr", result.stderr), ("stdout", result.stdout))
            streams = [f"{name}: {tail_of_output(value)}" for name, value in pairs if value]
            message = " ".join(streams) or "No output was produced."
            raise ToolExecutionException(
                f"Command failed with exit code {result.exit_code}. {message}", recoverable=True
            )
        return result.stdout

    def _install_packages(self, sandbox: Machine, packages: str) -> None:
        if packages:
            logger.debug(f"Tool {self.name} - {self.id}: Installing packages: {packages}")
            # Quoted: the command runs in a shell, where an unquoted pin like "pandas>=2.0" is a redirection.
            requirements = [shell_double_quote(requirement) for requirement in split_pip_requirements(packages)]
            if not requirements:
                return
            try:
                result = self._exec(sandbox, f"uv pip install -q {' '.join(requirements)}")
            except Exception as e:
                raise ToolExecutionException(f"Error during package installation: {e}", recoverable=True)

            if result.exit_code != 0:
                raise ToolExecutionException(
                    f"Error during package installation: {tail_of_output(result.stderr or result.stdout or '')}",
                    recoverable=True,
                )

    def _upload_file_to_sandbox(self, file: io.BytesIO, target_path: str, sandbox: Machine) -> str:
        self._client.machines.files.upload(sandbox.id, target_path, file.read())
        logger.debug(f"Tool {self.name} - {self.id}: Uploaded file to: {target_path}")
        return target_path

    def _download_file_bytes(self, file_path: str, sandbox: Machine) -> bytes:
        return self._client.machines.files.download(sandbox.id, file_path)

    def _run_shell_command(self, command: str, sandbox: Machine) -> tuple[int, str]:
        result = self._exec(sandbox, command)
        return result.exit_code, result.stdout or ""

    def _reconnect_sandbox(self, sandbox_id: str) -> Machine:
        return connect_boxd_machine(self.connection, sandbox_id, ready_timeout=self.timeout)

    def _destroy_sandbox(self, sandbox: Machine) -> None:
        self._client.machines.delete(sandbox.id)

    def close(self) -> None:
        """Delete the persistent machine if it exists."""
        if self._sandbox and self.persistent_sandbox:
            logger.debug(f"Tool {self.name} - {self.id}: Closing Sandbox")
            try:
                self._client.machines.delete(self._sandbox.id)
            except Exception as e:
                logger.warning(f"Tool {self.name} - {self.id}: Failed to delete sandbox: {e}")
            self._sandbox = None
