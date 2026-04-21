import io
from typing import Any, ClassVar

from e2b.exceptions import RateLimitException as E2BRateLimitException
from e2b_code_interpreter import Sandbox

from dynamiq.connections import E2B as E2BConnection
from dynamiq.nodes.agents.exceptions import ToolExecutionException
from dynamiq.nodes.tools.code_interpreter import DESCRIPTION_SANDBOX_INTERPRETER as DESCRIPTION_E2B
from dynamiq.nodes.tools.code_interpreter import BaseCodeInterpreterTool
from dynamiq.nodes.tools.code_interpreter import CodeInterpreterInputSchema as E2BInterpreterInputSchema  # noqa: F401
from dynamiq.utils.logger import logger


class E2BInterpreterTool(BaseCodeInterpreterTool):
    """
    E2B-specific sandbox interpreter tool.

    Thin adapter over BaseCodeInterpreterTool that delegates to the E2B SDK.
    """

    name: str = "e2b-code-interpreter-tool"
    description: str = DESCRIPTION_E2B
    connection: E2BConnection
    _sandbox: Sandbox | None = None
    _rate_limit_exception: ClassVar[type[Exception]] = E2BRateLimitException

    def _create_sandbox(self) -> Sandbox:
        return Sandbox.create(
            api_key=self.connection.api_key,
            timeout=self.timeout,
            domain=self.connection.domain,
        )

    def _get_sandbox_id(self, sandbox: Sandbox) -> str:
        return sandbox.sandbox_id

    def _get_sandbox_host(self, sandbox: Sandbox) -> str | None:
        try:
            return sandbox.get_host(port=sandbox.connection_config.envd_port)
        except Exception:
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
            logger.info(f"Executing Python code: {code}")
            execution = sandbox.run_code(code, timeout=timeout)
            output_parts = []

            if execution.text:
                output_parts.append(execution.text)

            if execution.error:
                if "NameError" in str(execution.error) and self.persistent_sandbox:
                    logger.debug(f"Tool {self.name}: Recoverable NameError in persistent session: {execution.error}")
                raise ToolExecutionException(f"Error during Python code execution: {execution.error}", recoverable=True)

            if hasattr(execution, "logs") and execution.logs:
                if hasattr(execution.logs, "stdout") and execution.logs.stdout:
                    for log in execution.logs.stdout:
                        output_parts.append(log)
                if hasattr(execution.logs, "stderr") and execution.logs.stderr:
                    for log in execution.logs.stderr:
                        output_parts.append(f"[stderr] {log}")

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

        run_kwargs: dict[str, Any] = {"background": True, "envs": env or {}, "cwd": cwd}
        if timeout:
            run_kwargs["timeout"] = timeout

        try:
            process = sandbox.commands.run(command, **run_kwargs)
        except Exception as e:
            raise ToolExecutionException(f"Error during shell command execution: {e}", recoverable=True)

        output = process.wait()
        if output.exit_code != 0:
            raise ToolExecutionException(f"Error during shell command execution: {output.stderr}", recoverable=True)
        return output.stdout

    def _install_packages(self, sandbox: Any, packages: str) -> None:
        if packages:
            logger.debug(f"Tool {self.name} - {self.id}: Installing packages: {packages}")
            try:
                process = sandbox.commands.run(f"pip install -qq {' '.join(packages.split(','))}")
            except Exception as e:
                raise ToolExecutionException(f"Error during package installation: {e}", recoverable=True)

            if process.exit_code != 0:
                raise ToolExecutionException(f"Error during package installation: {process.stderr}", recoverable=True)

    def _upload_file_to_sandbox(self, file: io.BytesIO, target_path: str, sandbox: Any) -> str:
        uploaded_info = sandbox.files.write(target_path, file)
        logger.debug(f"Tool {self.name} - {self.id}: Uploaded file info: {uploaded_info}")
        return uploaded_info.path

    def _download_file_bytes(self, file_path: str, sandbox: Any) -> bytes:
        return bytes(sandbox.files.read(file_path, "bytes"))

    def _run_shell_command(self, command: str, sandbox: Any) -> tuple[int, str]:
        res = sandbox.commands.run(command)
        if hasattr(res, "wait"):
            out = res.wait()
        else:
            out = res
        return out.exit_code, out.stdout

    def _destroy_sandbox(self, sandbox: Any) -> None:
        sandbox.kill()

    def set_timeout(self, timeout: int) -> None:
        super().set_timeout(timeout)
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
