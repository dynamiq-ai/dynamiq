import os
import shutil
import subprocess  # nosec B404 - required for sandboxed skill script execution
import tempfile
from dataclasses import dataclass
from pathlib import Path

from dynamiq.storages.file.base import FileStore
from dynamiq.utils.logger import logger


@dataclass
class SkillExecutionResult:
    """Result of running a skill script in the sandbox."""

    exit_code: int
    stdout: str
    stderr: str
    timed_out: bool
    work_dir: str | None = None
    output_files: dict[str, bytes] | None = None

    def __post_init__(self):
        if self.output_files is None:
            self.output_files = {}

    @property
    def success(self) -> bool:
        return self.exit_code == 0 and not self.timed_out

    def summary(self) -> str:
        parts = [f"exit_code={self.exit_code}", f"timed_out={self.timed_out}"]
        if self.stdout:
            parts.append(f"stdout_lines={len(self.stdout.splitlines())}")
        if self.stderr:
            parts.append(f"stderr_lines={len(self.stderr.splitlines())}")
        if self.output_files:
            parts.append(f"output_files={len(self.output_files)}")
        return "; ".join(parts)


class SkillExecutor:
    """Runs skill scripts in a subprocess sandbox.

    Extracts the skill directory from FileStore to a temporary directory,
    runs the requested script with the given arguments in a subprocess with
    timeout, and returns stdout/stderr/exit_code. Optionally cleans up the
    temp directory after execution.

    This keeps skill execution isolated from the main process and allows
    different skills to run in different sandboxes (separate temp dirs per run).

    Attributes:
        file_store: FileStore containing skill files under skills_prefix
        skills_prefix: Prefix path for skills (e.g. ".skills/")
        default_timeout_seconds: Default subprocess timeout
        cleanup_work_dir: Whether to remove temp dir after execution
        restrict_network: If True, set env to block typical network (optional)
    """

    def __init__(
        self,
        file_store: FileStore,
        skills_prefix: str = ".skills/",
        default_timeout_seconds: int = 120,
        cleanup_work_dir: bool = True,
        restrict_network: bool = False,
    ):
        self.file_store = file_store
        self.skills_prefix = skills_prefix.rstrip("/") + "/"
        self.default_timeout_seconds = default_timeout_seconds
        self.cleanup_work_dir = cleanup_work_dir
        self.restrict_network = restrict_network

    def execute_script(
        self,
        skill_name: str,
        script_relative_path: str,
        argv: list[str] | None = None,
        timeout_seconds: int | None = None,
        env_override: dict[str, str] | None = None,
        input_files: dict[str, str] | None = None,
        output_paths: list[str] | None = None,
        output_prefix: str = "",
    ) -> SkillExecutionResult:
        """Run a script from a skill in a sandbox (subprocess).

        The skill directory is extracted from FileStore to a temp directory;
        the script is run with cwd set to that directory. Optionally copy
        input files from FileStore into the sandbox and collect output files
        from the sandbox (returned as output_files for the agent to store).

        Args:
            skill_name: Name of the skill (directory under skills_prefix)
            script_relative_path: Path relative to skill root (e.g. scripts/run.py)
            argv: Optional list of arguments to pass to the script
            timeout_seconds: Override default timeout; None uses default
            env_override: Optional env vars to set (merged with sandbox env)
            input_files: Optional mapping FileStore path ->
            sandbox relative path (e.g. {"data/in.html": "input/in.html"})
            output_paths: Optional list of sandbox relative paths to collect after run (e.g. ["output/out.pptx"])
            output_prefix: FileStore path prefix for collected files (e.g. "generated/")

        Returns:
            SkillExecutionResult with exit_code, stdout, stderr, timed_out, output_files
        """
        argv = argv or []
        input_files = input_files or {}
        output_paths = output_paths or []
        timeout = timeout_seconds if timeout_seconds is not None else self.default_timeout_seconds
        work_dir: str | None = None
        output_files: dict[str, bytes] = {}

        try:
            work_dir = self._extract_skill_to_temp(skill_name)

            for store_path, sandbox_rel in input_files.items():
                store_path = store_path.replace("\\", "/")
                sandbox_rel = sandbox_rel.replace("\\", "/")
                if not self.file_store.exists(store_path):
                    return SkillExecutionResult(
                        exit_code=-1,
                        stdout="",
                        stderr=f"Input file not found in FileStore: {store_path}",
                        timed_out=False,
                        work_dir=work_dir if not self.cleanup_work_dir else None,
                        output_files=None,
                    )
                content = self.file_store.retrieve(store_path)
                target = Path(work_dir) / sandbox_rel
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(content)
            if input_files:
                logger.info(f"SkillExecutor: copied {len(input_files)} input file(s) into sandbox")

            script_path = Path(work_dir) / script_relative_path.replace("\\", "/")

            if not script_path.exists():
                return SkillExecutionResult(
                    exit_code=-1,
                    stdout="",
                    stderr=f"Script not found: {script_relative_path} (resolved: {script_path})",
                    timed_out=False,
                    work_dir=work_dir if not self.cleanup_work_dir else None,
                    output_files=None,
                )

            if script_path.suffix == ".py":
                cmd = [os.environ.get("PYTHON_EXECUTABLE", "python"), str(script_path)] + argv
            else:
                cmd = [str(script_path)] + argv

            env = self._sandbox_env(env_override)
            logger.info(
                f"SkillExecutor: running in sandbox skill={skill_name} script={script_relative_path} timeout={timeout}s"
            )

            try:
                result = subprocess.run(  # nosec B603 - cmd from skill path only, no user shell input
                    cmd,
                    cwd=work_dir,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    env=env,
                )
                timed_out = False
                exit_code = result.returncode
                stdout = result.stdout or ""
                stderr = result.stderr or ""
            except subprocess.TimeoutExpired as e:
                timed_out = True
                exit_code = -9
                stdout = (e.stdout.decode() if isinstance(e.stdout, bytes) else e.stdout) or ""
                stderr = (e.stderr.decode() if isinstance(e.stderr, bytes) else e.stderr) or ""
                stderr = f"Execution timed out after {timeout}s.\n" + stderr

            for rel in output_paths:
                rel = rel.replace("\\", "/")
                p = Path(work_dir) / rel
                if p.exists() and p.is_file():
                    output_files[(output_prefix + p.name).strip("/")] = p.read_bytes()
                elif p.exists():
                    logger.warning(f"SkillExecutor: output path is not a file, skipped: {rel}")
            if output_paths and output_files:
                logger.info(f"SkillExecutor: collected {len(output_files)} output file(s) from sandbox")

            return SkillExecutionResult(
                exit_code=exit_code,
                stdout=stdout,
                stderr=stderr,
                timed_out=timed_out,
                work_dir=work_dir if not self.cleanup_work_dir else None,
                output_files=output_files or None,
            )

        except Exception as e:
            logger.exception(f"SkillExecutor: failed to run script: {e}")
            return SkillExecutionResult(
                exit_code=-1,
                stdout="",
                stderr=str(e),
                timed_out=False,
                work_dir=work_dir if work_dir and not self.cleanup_work_dir else None,
                output_files=None,
            )

        finally:
            if work_dir and self.cleanup_work_dir:
                try:
                    shutil.rmtree(work_dir, ignore_errors=True)
                except OSError as e:
                    logger.warning(f"SkillExecutor: failed to cleanup work dir {work_dir}: {e}")

    def _extract_skill_to_temp(self, skill_name: str) -> str:
        """Extract skill directory from FileStore to a temp directory."""
        prefix = f"{self.skills_prefix}{skill_name}/"
        all_files = self.file_store.list_files(directory="", recursive=True)
        skill_files = [f for f in all_files if getattr(f, "path", f) and str(getattr(f, "path", "")).startswith(prefix)]

        if not skill_files:
            raise FileNotFoundError(f"No files found for skill: {skill_name} under {prefix}")

        work_dir = tempfile.mkdtemp(prefix=f"dynamiq_skill_{skill_name}_")
        prefix_len = len(prefix)

        for file_info in skill_files:
            path = getattr(file_info, "path", file_info)
            path = str(path)
            rel = path[prefix_len:]
            if not rel or rel.startswith("/"):
                continue
            target = Path(work_dir) / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            content = self.file_store.retrieve(path)
            target.write_bytes(content)

        return work_dir

    def _sandbox_env(self, override: dict[str, str] | None) -> dict[str, str]:
        """Build environment for sandbox: base env, optionally no network, then override."""
        env = os.environ.copy()
        if self.restrict_network:
            env["HTTP_PROXY"] = ""
            env["HTTPS_PROXY"] = ""
            env["ALL_PROXY"] = ""
            env["NO_PROXY"] = "*"
        if override:
            env.update(override)
        return env
