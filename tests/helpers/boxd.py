"""An in-memory stand-in for the boxd SDK client, for tests.

boxd is an opt-in extra, so the test environment may not have it. ``boxd_sdk_module()`` returns the installed SDK,
or a stand-in module with the names the dynamiq integration imports. ``FakeBoxd`` keeps machines, files, proxies and
exec calls in memory and answers commands through a handler the test sets.
"""

import builtins
import importlib.util
import sys
import types
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


def _stub_module() -> types.ModuleType:
    module = types.ModuleType("boxd")

    class BoxdError(Exception):
        pass

    class NotFoundError(BoxdError):
        pass

    class RateLimitError(BoxdError):
        pass

    class Boxd:
        def __init__(self, **kwargs):
            raise RuntimeError("the boxd stand-in has no client: patch the connection's get_client()")

    module.BoxdError = BoxdError
    module.NotFoundError = NotFoundError
    module.RateLimitError = RateLimitError
    module.Boxd = Boxd
    module.Machine = Any
    return module


_STUB = _stub_module()


def boxd_sdk_module() -> types.ModuleType:
    """The installed boxd SDK, or one stand-in shared by every test so that error classes stay the same."""
    if "boxd" in sys.modules:
        return sys.modules["boxd"]
    if importlib.util.find_spec("boxd") is not None:
        import boxd

        return boxd
    return _STUB


@dataclass
class ExecCall:
    machine_id: str
    command: Any
    env: dict | None
    cwd: str | None
    timeout: float | None

    @property
    def script(self) -> str:
        """The shell script of a ``bash -c`` argv, or the command as given."""
        if isinstance(self.command, list) and self.command[:2] == ["bash", "-c"]:
            return self.command[2]
        return self.command if isinstance(self.command, str) else " ".join(self.command)


@dataclass
class ExecResult:
    stdout: str = ""
    stderr: str = ""
    exit_code: int = 0


@dataclass
class Access:
    url: str


@dataclass
class FakeMachine:
    id: str
    name: str
    status: str = "running"
    access: Access = None
    create_kwargs: dict = field(default_factory=dict)


@dataclass
class Proxy:
    name: str
    port: int


class FakeFiles:
    def __init__(self, boxd: "FakeBoxd"):
        self._boxd = boxd

    def upload(self, machine_id: str, path: str, data: bytes | str, **kwargs) -> int:
        self._boxd.require(machine_id)
        content = data.encode() if isinstance(data, str) else bytes(data)
        self._boxd.files[(machine_id, path)] = content
        return len(content)

    def download(self, machine_id: str, path: str, **kwargs) -> bytes:
        self._boxd.require(machine_id)
        try:
            return self._boxd.files[(machine_id, path)]
        except KeyError:
            raise self._boxd.sdk.NotFoundError(f"{path} not found") from None


class FakeProxies:
    def __init__(self, boxd: "FakeBoxd"):
        self._boxd = boxd
        self.by_machine: dict[str, list[Proxy]] = {}

    def list(self, machine_name: str, **kwargs) -> builtins.list[Proxy]:
        return builtins.list(self.by_machine.get(machine_name, []))

    def create(self, machine_name: str, name: str, port: int, **kwargs) -> Proxy:
        proxy = Proxy(name=name, port=port)
        self.by_machine.setdefault(machine_name, []).append(proxy)
        return proxy


class FakeMachines:
    def __init__(self, boxd: "FakeBoxd"):
        self._boxd = boxd
        self.files = FakeFiles(boxd)
        self.proxies = FakeProxies(boxd)

    def create(self, name: str | None = None, **kwargs) -> FakeMachine:
        if self._boxd.create_failures:
            raise self._boxd.create_failures.pop(0)
        machine_id = f"machine-{len(self._boxd.created) + 1}"
        machine = FakeMachine(
            id=machine_id,
            name=name,
            status="starting",
            access=Access(url=f"https://{name}.boxd.sh"),
            create_kwargs=kwargs,
        )
        self._boxd.machines[machine_id] = machine
        self._boxd.created.append(machine)
        return machine

    def get(self, machine_id: str, **kwargs) -> FakeMachine:
        return self._boxd.require(machine_id)

    def wait_until_ready(self, machine_id: str, timeout: float = 90.0, **kwargs) -> FakeMachine:
        machine = self._boxd.require(machine_id)
        self._boxd.calls.append(("wait_until_ready", machine_id))
        if self._boxd.ready_error is not None:
            raise self._boxd.ready_error
        if machine.status not in ("starting", "running"):
            raise self._boxd.sdk.BoxdError(f"machine is {machine.status}")
        machine.status = "running"
        return machine

    def _transition(self, action: str, machine_id: str, status: str) -> None:
        self._boxd.require(machine_id).status = status
        self._boxd.calls.append((action, machine_id))

    def start(self, machine_id: str, **kwargs) -> None:
        self._transition("start", machine_id, "starting")

    def wake(self, machine_id: str, **kwargs) -> None:
        self._transition("wake", machine_id, "starting")

    def resume(self, machine_id: str, **kwargs) -> None:
        self._transition("resume", machine_id, "running")

    def delete(self, machine_id: str, **kwargs) -> None:
        self._boxd.require(machine_id)
        del self._boxd.machines[machine_id]
        self._boxd.deleted.append(machine_id)

    def exec(self, machine_id: str, command: Any, *, env=None, cwd=None, timeout=None, **kwargs) -> ExecResult:
        machine = self._boxd.require(machine_id)
        if machine.status != "running":
            raise self._boxd.sdk.BoxdError(f"machine is {machine.status}")
        call = ExecCall(machine_id=machine_id, command=command, env=env, cwd=cwd, timeout=timeout)
        self._boxd.execs.append(call)
        return self._boxd.exec_handler(call)


class FakeBoxd:
    """Machines, files and exec calls of one test, in memory."""

    def __init__(self):
        self.sdk = boxd_sdk_module()
        self.machines: dict[str, FakeMachine] = {}
        self.created: list[FakeMachine] = []
        self.deleted: list[str] = []
        self.calls: list[tuple[str, str]] = []
        self.execs: list[ExecCall] = []
        self.files: dict[tuple[str, str], bytes] = {}
        self.create_failures: list[Exception] = []
        self.ready_error: Exception | None = None
        self.exec_handler: Callable[[ExecCall], ExecResult] = lambda call: ExecResult()
        self.machines_api = FakeMachines(self)

    def require(self, machine_id: str) -> FakeMachine:
        try:
            return self.machines[machine_id]
        except KeyError:
            raise self.sdk.NotFoundError(f"machine {machine_id} not found") from None

    def add_machine(self, machine_id: str, status: str, name: str = "existing") -> FakeMachine:
        machine = FakeMachine(id=machine_id, name=name, status=status, access=Access(url=f"https://{name}.boxd.sh"))
        self.machines[machine_id] = machine
        return machine

    def scripts(self) -> list[str]:
        return [call.script for call in self.execs]


class FakeBoxdClient:
    """What ``Boxd.get_client()`` returns: the SDK client's ``machines`` resource, backed by a ``FakeBoxd``."""

    def __init__(self, fake: FakeBoxd):
        self.machines = fake.machines_api
