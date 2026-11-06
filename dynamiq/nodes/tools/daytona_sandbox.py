import io
from typing import Any, ClassVar

from daytona import CreateSandboxFromSnapshotParams, DaytonaRateLimitError, SandboxState
from pydantic import Field

from dynamiq.connections import Daytona as DaytonaConnection
from dynamiq.nodes.agents.exceptions import ToolExecutionException
from dynamiq.nodes.tools.code_interpreter import (
    DESCRIPTION_SANDBOX_INTERPRETER,
    BaseCodeInterpreterTool,
    shell_double_quote,
    split_pip_requirements,
    tail_of_output,
)
from dynamiq.utils.logger import logger


class DaytonaInterpreterTool(BaseCodeInterpreterTool):
    """Daytona implementation of the sandbox interpreter tool."""

    name: str = "daytona-code-interpreter-tool"
    description: str = DESCRIPTION_SANDBOX_INTERPRETER
    base_path: str = "/home/daytona"
    connection: DaytonaConnection
    auto_stop_interval: int | None = Field(
        default=None,
        description=(
            "Daytona auto-stop interval in minutes. If None, derived from "
            "``timeout`` so persistent sandboxes stop themselves on idle."
        ),
    )
    _sandbox: Any = None
    _rate_limit_exception: ClassVar[type[Exception]] = DaytonaRateLimitError

    def _resolve_auto_stop_interval(self) -> int:
        """Return the auto-stop interval (minutes) to attach at creation."""
        if self.auto_stop_interval is not None:
            return self.auto_stop_interval
        return max(1, (self.timeout + 59) // 60)

    def _create_sandbox(self) -> Any:
        params = CreateSandboxFromSnapshotParams(auto_stop_interval=self._resolve_auto_stop_interval())
        return self.connection.get_client().create(params, timeout=self.timeout)

    def _get_sandbox_id(self, sandbox: Any) -> str:
        return sandbox.id

    def _get_sandbox_host(self, sandbox: Any) -> str | None:
        return None

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

        try:
            logger.debug(f"Executing Python code: {code}")
            result = sandbox.code_interpreter.run_code(code, timeout=timeout)
            output_parts = []

            if result.stdout:
                output_parts.append(result.stdout)

            if result.error:
                if "NameError" in str(result.error.name) and self.persistent_sandbox:
                    logger.debug(f"Tool {self.name}: Recoverable NameError in persistent session: {result.error}")
                raise ToolExecutionException(
                    f"Error during Python code execution: {result.error.name}: {result.error.value}",
                    recoverable=True,
                )

            if result.stderr:
                output_parts.append(f"[stderr] {result.stderr}")

            return "\n".join(output_parts) if output_parts else ""

        except ToolExecutionException:
            raise
        except Exception as e:
            raise ToolExecutionException(f"Error during Python code execution: {e}", recoverable=True)

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
            result = sandbox.process.exec(command, cwd=cwd, env=env or {}, timeout=timeout)
        except Exception as e:
            raise ToolExecutionException(f"Error during shell command execution: {e}", recoverable=True)

        if result.exit_code != 0:
            raise ToolExecutionException(
                f"Error during shell command execution: {tail_of_output(result.result or '')}", recoverable=True
            )
        return result.result

    def _install_packages(self, sandbox: Any, packages: str) -> None:
        if packages:
            logger.debug(f"Tool {self.name} - {self.id}: Installing packages: {packages}")
            # Quoted: the command runs in a shell, where an unquoted pin like "pandas>=2.0" is a redirection.
            requirements = [shell_double_quote(requirement) for requirement in split_pip_requirements(packages)]
            if not requirements:
                return
            try:
                result = sandbox.process.exec(f"pip install -qq {' '.join(requirements)}")
            except Exception as e:
                raise ToolExecutionException(f"Error during package installation: {e}", recoverable=True)

            if result.exit_code != 0:
                raise ToolExecutionException(f"Error during package installation: {result.result}", recoverable=True)

    def _upload_file_to_sandbox(self, file: io.BytesIO, target_path: str, sandbox: Any) -> str:
        sandbox.fs.upload_file(file.read(), target_path)
        logger.debug(f"Tool {self.name} - {self.id}: Uploaded file to: {target_path}")
        return target_path

    def _download_file_bytes(self, file_path: str, sandbox: Any) -> bytes:
        return sandbox.fs.download_file(file_path)

    def _run_shell_command(self, command: str, sandbox: Any) -> tuple[int, str]:
        result = sandbox.process.exec(command)
        return result.exit_code, result.result or ""

    def _reconnect_sandbox(self, sandbox_id: str) -> Any:
        sandbox = self.connection.get_client().get(sandbox_id)
        if sandbox.state != SandboxState.STARTED:
            sandbox.start(timeout=self.timeout)
        return sandbox

    def _destroy_sandbox(self, sandbox: Any) -> None:
        self.connection.get_client().delete(sandbox)

    def set_timeout(self, timeout: int) -> None:
        """Update the tool-level timeout and propagate to the live sandbox."""
        super().set_timeout(timeout)
        if self._sandbox and self.persistent_sandbox:
            interval_minutes = max(1, (timeout + 59) // 60)
            try:
                self._sandbox.set_autostop_interval(interval_minutes)
                logger.debug(f"Tool {self.name} - {self.id}: Updated sandbox auto-stop to {interval_minutes}min")
            except Exception as e:
                logger.warning(f"Tool {self.name} - {self.id}: Failed to update sandbox timeout: {e}")

    def _keep_sandbox_alive(self, sandbox: Any, seconds: int) -> None:
        # Daytona stops a sandbox after the auto-stop interval passes without an SDK call, which a long command can
        # take. An interval of 0 disables auto-stop, so it is left alone.
        interval_minutes = self._resolve_auto_stop_interval()
        if interval_minutes:
            sandbox.set_autostop_interval(max(interval_minutes, (seconds + 59) // 60))

    def close(self) -> None:
        """Close the persistent sandbox if it exists."""
        if self._sandbox and self.persistent_sandbox:
            logger.debug(f"Tool {self.name} - {self.id}: Closing Sandbox")
            try:
                self.connection.get_client().delete(self._sandbox)
            except Exception as e:
                logger.warning(f"Tool {self.name} - {self.id}: Failed to delete sandbox: {e}")
            self._sandbox = None
