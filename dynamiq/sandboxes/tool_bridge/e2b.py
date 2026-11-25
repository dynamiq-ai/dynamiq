import base64
import contextvars
import hashlib
import json
import secrets
import shlex
import threading
from concurrent.futures import Future, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any, ClassVar

from dynamiq.sandboxes.tool_bridge.types import ToolBridgeConfig, ToolBridgeHandler, refusal
from dynamiq.utils.logger import logger

SANDBOX_FILES = ("broker.py", "dynamiq_tools.py")
REPLY_CHUNK_CHARS = 1024 * 1024  # one send_stdin call each
BROKER_LIFETIME_MARGIN_SECONDS = 60


def _sandbox_files() -> dict[str, str]:
    directory = Path(__file__).parent / "sandbox_files"
    return {name: (directory / name).read_text() for name in SANDBOX_FILES}


class E2BToolBridgeSession:
    """The tool bridge of one shell command over E2B: a broker in the sandbox, relayed through its stdout and stdin.

    Use it as a context manager around the command, and run the command as ``wrap(command)`` with ``envs``. Each
    session has its own socket, nonce and token. Requests run on at most ``max_concurrent_calls`` threads, each in a
    copy of the context the session was entered in, so a bridged call sees the run's context variables as a direct
    call does. A command makes at most ``max_calls_per_command`` requests; those past it are refused unrun.
    """

    BRIDGE_DIR: ClassVar[str] = "/tmp/dynamiq-bridge"  # nosec B108 - a path inside the sandbox
    _installed: ClassVar[set[tuple[str, str]]] = set()
    _install_locks: ClassVar[dict[str, threading.Lock]] = {}
    _install_locks_guard: ClassVar[threading.Lock] = threading.Lock()

    def __init__(
        self,
        sdk_sandbox: Any,
        sandbox_id: str,
        handler: ToolBridgeHandler,
        settings: ToolBridgeConfig,
        command_timeout: float,
    ):
        self._sdk = sdk_sandbox
        self._sandbox_id = sandbox_id
        self._handler = handler
        self._settings = settings
        self._lifetime = command_timeout + BROKER_LIFETIME_MARGIN_SECONDS
        self._nonce = secrets.token_hex(8)
        self._token = secrets.token_urlsafe(24)
        self._socket = f"{self.BRIDGE_DIR}/s-{self._nonce}.sock"
        self._marker = f"\x1eDQB:{self._nonce}:"
        self._max_frame_chars = settings.max_request_bytes * 4 // 3 + 256
        self._handle = None
        self._ready = threading.Event()
        self._stopping = threading.Event()
        self._send_lock = threading.Lock()
        self._requests = 0
        self._futures: list[Future] = []
        self._executor = ThreadPoolExecutor(settings.max_concurrent_calls, thread_name_prefix="tool-bridge")
        self._reader: threading.Thread | None = None
        self._http_port: int | None = None

    def __enter__(self) -> "E2BToolBridgeSession":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    @property
    def envs(self) -> dict[str, str]:
        # The template's CLIs (tools-cli, sql-cli) call the platform at DYNAMIQ_API_BASE_URL with DYNAMIQ_API_TOKEN:
        # here they reach the broker, which answers from the agent's own tools.
        return {
            "DYNAMIQ_TOOLS_SOCKET": self._socket,
            "DYNAMIQ_TOOLS_TOKEN": self._token,
            "DYNAMIQ_API_BASE_URL": f"http://127.0.0.1:{self._http_port}",
            "DYNAMIQ_API_TOKEN": self._token,
        }

    def wrap(self, command: str) -> str:
        """The command with the client on PATH and PYTHONPATH, set in the shell since E2B ``envs`` replace them."""
        lib = shlex.quote(f"{self.BRIDGE_DIR}/lib")
        bin_dir = shlex.quote(f"{self.BRIDGE_DIR}/bin")
        return f'export PATH={bin_dir}:"$PATH" PYTHONPATH={lib}"${{PYTHONPATH:+:$PYTHONPATH}}"\n{command}'

    def _install(self) -> None:
        """Upload the broker and the client once per sandbox and version of the files."""
        files = _sandbox_files()
        version = hashlib.sha256("".join(files.values()).encode()).hexdigest()[:12]
        with self._install_locks_guard:
            lock = self._install_locks.setdefault(self._sandbox_id, threading.Lock())
        with lock:
            if (self._sandbox_id, version) in self._installed:
                return
            root = self.BRIDGE_DIR
            self._sdk.files.write(f"{root}/broker.py", files["broker.py"])
            self._sdk.files.write(f"{root}/lib/dynamiq_tools.py", files["dynamiq_tools.py"])
            launcher = f'#!/bin/sh\nexec python3 {root}/lib/dynamiq_tools.py "$@"\n'
            self._sdk.files.write(f"{root}/bin/dynamiq-tools", launcher)
            self._sdk.commands.run(f"chmod 700 {root} && chmod 755 {root}/bin/dynamiq-tools", timeout=30)
            self._installed.add((self._sandbox_id, version))

    def start(self) -> None:
        """Start the broker and wait until it serves.

        Raises:
            TimeoutError: If the broker did not report itself ready within ``ready_timeout_seconds``.
        """
        self._install()
        limit = self._settings.max_request_bytes
        command = f"python3 -u {self.BRIDGE_DIR}/broker.py {self._socket} {self._nonce} {limit}"
        started: list[Any] = []

        def run_broker() -> None:
            try:
                started.append(
                    self._sdk.commands.run(
                        command,
                        background=True,
                        stdin=True,
                        envs={"DYNAMIQ_TOOLS_TOKEN": self._token},
                        timeout=self._lifetime,
                    )
                )
            except Exception as e:
                started.append(e)

        # The SDK puts no limit on opening a background command's stream, so the start is waited on in a daemon
        # thread for a bounded time.
        starter = threading.Thread(target=run_broker, name="tool-bridge-start", daemon=True)
        starter.start()
        starter.join(self._settings.ready_timeout_seconds)
        if not started:
            self._stopping.set()
            raise TimeoutError("The tool bridge did not start.")
        if isinstance(started[0], Exception):
            self._stopping.set()
            raise started[0]
        self._handle = started[0]
        self._reader = threading.Thread(
            target=contextvars.copy_context().run, args=(self._read,), name="tool-bridge-reader", daemon=True
        )
        self._reader.start()
        if not self._ready.wait(self._settings.ready_timeout_seconds):
            self.close(grace_seconds=0)
            raise TimeoutError("The tool bridge did not report itself ready.")

    def _read(self) -> None:
        buffer = ""
        try:
            for stdout, stderr, _ in self._handle:
                if stderr:
                    logger.debug(f"Tool bridge broker stderr: {stderr.strip()[-500:]!r}")
                if not stdout:
                    continue
                buffer += stdout
                while "\n" in buffer:
                    line, buffer = buffer.split("\n", 1)
                    if line.startswith(self._marker):
                        self._on_frame(line[len(self._marker) :])
                if len(buffer) > self._max_frame_chars:
                    logger.warning("Tool bridge: the broker sent an oversized frame; closing the bridge.")
                    return
        except Exception as e:
            if not self._stopping.is_set():
                logger.warning(f"Tool bridge: lost the broker's output stream ({type(e).__name__}).")
        finally:
            if not self._stopping.is_set():
                # Nothing answers without a reader: end the broker so callers fail now rather than at their timeout.
                self._kill_broker()

    def _on_frame(self, frame: str) -> None:
        if self._stopping.is_set() or len(frame) > self._max_frame_chars:
            return
        try:
            message = json.loads(base64.b64decode(frame))
        except ValueError:
            logger.warning("Tool bridge: dropped a malformed frame.")
            return
        if not isinstance(message, dict):
            return
        if message.get("kind") == "ready" and not self._ready.is_set():
            port = message.get("http_port")
            if type(port) is not int or not 0 < port < 65536:
                logger.warning("Tool bridge: the broker reported no usable port.")
                return
            self._http_port = port
            self._ready.set()
        elif message.get("kind") == "req" and isinstance(message.get("id"), str) and self._ready.is_set():
            self._requests += 1
            if self._requests > self._settings.max_calls_per_command:
                limit = self._settings.max_calls_per_command
                self._reply(message["id"], refusal("BudgetExceeded", f"at most {limit} requests per command"))
                return
            # Copied in the reader, whose context is a copy of the shell call's: each request gets its own.
            context = contextvars.copy_context()
            self._futures.append(self._executor.submit(context.run, self._serve, message))

    def _serve(self, request: dict[str, Any]) -> None:
        request_id = request.pop("id")
        request.pop("kind", None)
        if self._stopping.is_set():
            return
        try:
            reply = self._handler.handle(request)
        except BaseException as e:  # noqa: B036 - a tool's SystemExit still needs an answer, or its caller hangs
            logger.warning(f"Tool bridge: a request failed with {type(e).__name__}.")
            reply = refusal(type(e).__name__)
        self._reply(request_id, reply)

    def _reply(self, request_id: str, reply: dict[str, Any]) -> None:
        try:
            data = json.dumps(reply, ensure_ascii=False, default=str)
        except (TypeError, ValueError):
            data = json.dumps(refusal("UnserializableResult"))
        if len(data.encode()) > self._settings.max_reply_bytes:
            data = json.dumps(refusal("ReplyTooLarge", "the result is larger than max_reply_bytes"))
        encoded = base64.b64encode(data.encode()).decode()
        chunks = [encoded[i : i + REPLY_CHUNK_CHARS] for i in range(0, len(encoded), REPLY_CHUNK_CHARS)] or [""]
        try:
            with self._send_lock:
                for index, chunk in enumerate(chunks):
                    line = json.dumps({"id": request_id, "chunk": chunk, "last": index == len(chunks) - 1})
                    self._sdk.commands.send_stdin(self._handle.pid, line + "\n", request_timeout=60)
        except Exception as e:
            if not self._stopping.is_set():
                logger.warning(f"Tool bridge: failed to deliver a reply ({type(e).__name__}).")

    def _kill_broker(self) -> None:
        if self._handle is None:
            return
        try:
            self._sdk.commands.kill(self._handle.pid, request_timeout=15)
        except Exception as e:
            logger.debug(f"Tool bridge: stopping the broker failed ({type(e).__name__}).")

    def close(self, grace_seconds: float = 5) -> None:
        """Stop serving and end the broker.

        Requests not yet started are dropped. Those running get ``grace_seconds`` in all to finish; any still
        running after that finish on their own, with nowhere to send their results.
        """
        self._stopping.set()
        for future in self._futures:
            future.cancel()
        if self._futures:
            wait(self._futures, timeout=grace_seconds)
        self._executor.shutdown(wait=False, cancel_futures=True)
        self._kill_broker()
        if self._reader is not None:
            self._reader.join(5)
