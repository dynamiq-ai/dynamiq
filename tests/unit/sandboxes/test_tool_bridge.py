"""Tools in code: the real broker and client, run as local processes in place of an E2B sandbox."""

import os
import subprocess
import sys
import threading
import time
from datetime import datetime
from types import SimpleNamespace
from typing import Any, ClassVar, Literal
from unittest.mock import patch

import pytest
from pydantic import BaseModel, Field

from dynamiq import Workflow
from dynamiq.callbacks import TracingCallbackHandler
from dynamiq.connections import E2B, BigQuery, MySQL, PostgreSQL
from dynamiq.flows import Flow
from dynamiq.nodes import NodeGroup
from dynamiq.nodes.agents.hooks.builtin import CallLimitHook, ToolPolicyHook
from dynamiq.nodes.node import Node
from dynamiq.nodes.tools.sql_executor import SQLExecutor
from dynamiq.nodes.types import InferenceMode
from dynamiq.runnables import RunnableConfig, RunnableStatus
from dynamiq.sandboxes.base import Sandbox, ShellCommandResult
from dynamiq.sandboxes.e2b import E2BSandbox
from dynamiq.sandboxes.tool_bridge.e2b import E2BToolBridgeSession
from dynamiq.sandboxes.tool_bridge.sql_cli import sql_cli_connection
from dynamiq.sandboxes.tools.shell import SandboxShellTool
from tests.helpers.agent_hooks import ALL_MODES, build_agent, final_step, scripted, tool_step

MODE = InferenceMode.FUNCTION_CALLING


class LookupSchema(BaseModel):
    key: str = Field(default="", description="What to look up.")
    secret_header: str | None = Field(default=None, json_schema_extra={"is_accessible_to_agent": False})


class LookupTool(Node):
    """Records what it is run with; its hidden field comes from tool_params only."""

    group: Literal[NodeGroup.TOOLS] = NodeGroup.TOOLS
    name: str = "lookup"
    description: str = "Looks a key up."
    input_schema: ClassVar[type[LookupSchema]] = LookupSchema
    calls: ClassVar[list[dict]] = []
    size: int = 0
    delay: float = 0

    def execute(self, input_data: LookupSchema, config: RunnableConfig | None = None, **kwargs) -> dict[str, Any]:
        LookupTool.calls.append(input_data.model_dump())
        time.sleep(self.delay)
        if input_data.key == "raise":
            raise ValueError("internal detail sk-do-not-leak")
        if input_data.key == "exit":
            sys.exit(3)
        return {"content": "x" * self.size if self.size else f"value of {input_data.key}"}


class LocalHandle:
    def __init__(self, process: subprocess.Popen):
        self.process = process
        self.pid = process.pid

    def __iter__(self):
        for line in self.process.stdout:
            yield line, None, None


class LocalSDK:
    """The parts of the E2B SDK the bridge uses, on local processes."""

    def __init__(self):
        self.processes: dict[int, subprocess.Popen] = {}
        self.files = SimpleNamespace(write=self._write)
        self.commands = SimpleNamespace(run=self._run, send_stdin=self._send_stdin, kill=self._kill)

    @staticmethod
    def _write(path: str, data: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            f.write(data)

    def _run(self, command, background=False, stdin=False, envs=None, timeout=None):
        env = {**os.environ, **(envs or {})}
        if background:
            process = subprocess.Popen(
                f"exec {command}", shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, env=env
            )
            self.processes[process.pid] = process
            return LocalHandle(process)
        return subprocess.run(command, shell=True, capture_output=True, text=True, env=env, timeout=timeout)

    def _send_stdin(self, pid, data, request_timeout=None):
        process = self.processes[pid]
        process.stdin.write(data)
        process.stdin.flush()

    def _kill(self, pid, request_timeout=None):
        self.processes[pid].kill()


class LocalSandbox(Sandbox):
    """Runs commands as local processes, bridged by the E2B session over ``LocalSDK``."""

    bridge_dir: str

    def run_command_shell(self, command, timeout=60, run_in_background_enabled=False, envs=None):
        env = {**os.environ, **(envs or {})}
        done = subprocess.run(["bash", "-c", command], capture_output=True, text=True, env=env, timeout=timeout)
        return ShellCommandResult(stdout=done.stdout, stderr=done.stderr, exit_code=done.returncode)

    def open_tool_bridge(self, handler, command_timeout):
        session_class = type("LocalSession", (E2BToolBridgeSession,), {"BRIDGE_DIR": self.bridge_dir})
        return session_class(LocalSDK(), self.bridge_dir, handler, self.tool_bridge, command_timeout)

    def get_tools(self, llm=None):
        return [SandboxShellTool(sandbox=self)]

    def store(self, *args, **kwargs):
        raise NotImplementedError

    def list_files(self, target_dir=None):
        return []

    def retrieve(self, file_path):
        raise FileNotFoundError(file_path)

    def exists(self, file_path):
        return False

    def upload_file(self, file_name, content, destination_path=None):
        raise NotImplementedError

    def close(self, kill: bool = False):
        pass


@pytest.fixture(autouse=True)
def _reset():
    LookupTool.calls.clear()


def local_sandbox(tmp_path, **bridge) -> LocalSandbox:
    return LocalSandbox(bridge_dir=str(tmp_path / "bridge"), tool_bridge={"enabled": True, **bridge})


def make_agent(tmp_path, hooks=None, tool=None, **bridge):
    sandbox = local_sandbox(tmp_path, **bridge)
    return build_agent(hooks=hooks, tools=[tool or LookupTool()], sandbox={"enabled": True, "backend": sandbox})


def run_shell(agent, command, tool_params=None, callbacks=None, mode=MODE):
    steps = [tool_step("sandbox-shell", command=command), final_step("done")]
    config = RunnableConfig(callbacks=callbacks or [])
    with scripted(agent, mode, steps) as recorder:
        result = agent.run(input_data={"input": "go", "tool_params": tool_params or {}}, config=config)
    assert result.status == RunnableStatus.SUCCESS, result.error
    return recorder.sent_text(1)


def py(code: str) -> str:
    return f"python3 - <<'EOF'\n{code}\nEOF"


CALL_EACH = (
    "from dynamiq_tools import ToolError, call\n"
    "for name, args in {calls}:\n"
    "    try:\n        print(name, 'ok', call(name, args))\n"
    "    except ToolError as e:\n        print(name, e.type)"
)


@pytest.mark.parametrize("mode", ALL_MODES)
def test_code_calls_a_tool_with_the_runs_hidden_params(tmp_path, mode):
    agent = make_agent(tmp_path)
    code = "from dynamiq_tools import call\nfor k in 'abc':\n    print(call('lookup', {'key': k}))"
    tool_params = {"by_name": {"lookup": {"secret_header": "per-run-value"}}}
    tracing = TracingCallbackHandler()
    observation = run_shell(agent, py(code), tool_params, callbacks=[tracing], mode=mode)

    assert "value of a\nvalue of b\nvalue of c" in observation
    assert [c["secret_header"] for c in LookupTool.calls] == ["per-run-value"] * 3
    runs = list(tracing.runs.values())
    shell = next(r for r in runs if r.name == "sandbox-shell")
    bridged = [r for r in runs if r.name == "lookup"]
    assert len(bridged) == 3 and all(r.parent_run_id == shell.id for r in bridged)


def test_code_cannot_set_hidden_fields(tmp_path):
    agent = make_agent(tmp_path)
    run_shell(agent, py("from dynamiq_tools import call\ncall('lookup', {'key': 'k', 'secret_header': 'forged'})"))
    assert LookupTool.calls == [{"key": "k", "secret_header": None}]


def test_cli_lists_gets_and_calls(tmp_path):
    agent = make_agent(tmp_path)
    observation = run_shell(
        agent, 'dynamiq-tools list; dynamiq-tools get lookup; dynamiq-tools call lookup --input \'{"key": "q"}\''
    )
    assert "lookup: Looks a key up." in observation
    assert '"name": "lookup"' in observation and '"key"' in observation
    assert "\nsandbox-shell: " not in observation
    assert "value of q" in observation


@pytest.mark.parametrize("key, error", [("raise", "ValueError"), ("exit", "SystemExit")])
def test_failures_cross_as_type_only(tmp_path, key, error):
    agent = make_agent(tmp_path)
    observation = run_shell(agent, py(CALL_EACH.format(calls=f"[('lookup', {{'key': '{key}'}})]")))
    assert f"lookup {error}" in observation
    assert "sk-do-not-leak" not in observation


def test_refusals_and_budget(tmp_path):
    agent = make_agent(tmp_path, max_calls_per_command=3)
    calls = "[('sandbox-shell', {'command': 'id'}), ('nope', {}), ('lookup', {'key': '1'}), ('lookup', {'key': '2'})]"
    observation = run_shell(agent, py(CALL_EACH.format(calls=calls)))
    assert "sandbox-shell ToolRefused" in observation and "nope UnknownTool" in observation
    assert "lookup ok value of 1" in observation and "lookup BudgetExceeded" in observation
    assert LookupTool.calls == [{"key": "1", "secret_header": None}]


def test_listing_counts_against_the_budget(tmp_path):
    agent = make_agent(tmp_path, max_calls_per_command=2)
    code = (
        "from dynamiq_tools import ToolError, call, list_tools\n"
        "list_tools(); list_tools()\ntry:\n    call('lookup', {'key': 'k'})\n"
        "except ToolError as e:\n    print(e.type)"
    )
    assert "BudgetExceeded" in run_shell(agent, py(code)) and LookupTool.calls == []


def test_calls_a_hook_sends_for_approval_are_refused(tmp_path):
    approval = ToolPolicyHook(tools=["lookup"], approval=True)
    agent = make_agent(tmp_path, hooks=[approval])
    observation = run_shell(agent, py(CALL_EACH.format(calls="[('lookup', {'key': '1'})]")))
    assert "lookup ApprovalRequired" in observation and LookupTool.calls == []


def test_call_limit_hook_counts_bridged_calls(tmp_path):
    agent = make_agent(tmp_path, hooks=[CallLimitHook(tools=["lookup"], max_per_run=1)])
    tracing = TracingCallbackHandler()
    calls = "[('lookup', {'key': '1'}), ('lookup', {'key': '2'})]"
    observation = run_shell(agent, py(CALL_EACH.format(calls=calls)), callbacks=[tracing])
    assert "lookup ok value of 1" in observation and "lookup ToolBlocked" in observation
    agent_run = next(r for r in tracing.runs.values() if r.name == "assistant")
    assert [e["decision"] for e in agent_run.metadata.get("hooks", [])] == ["block"]


def test_large_result_reaches_code_whole(tmp_path):
    agent = make_agent(tmp_path, tool=LookupTool(size=200_000))
    observation = run_shell(agent, py("from dynamiq_tools import call\nprint(len(call('lookup', {'key': 'big'})))"))
    assert "200000" in observation


def test_oversized_request_is_refused(tmp_path):
    agent = make_agent(tmp_path, max_request_bytes=1024)
    observation = run_shell(agent, py(CALL_EACH.format(calls="[('lookup', {'key': 'k' * 4096})]")))
    assert "lookup RequestTooLarge" in observation and LookupTool.calls == []


def test_concurrent_calls(tmp_path):
    agent = make_agent(tmp_path)
    code = (
        "from concurrent.futures import ThreadPoolExecutor\nfrom dynamiq_tools import call\n"
        "with ThreadPoolExecutor(8) as pool:\n"
        "    print(sorted(pool.map(lambda k: call('lookup', {'key': str(k)}), range(8))))"
    )
    observation = run_shell(agent, py(code))
    assert "'value of 7'" in observation and len(LookupTool.calls) == 8


def test_requests_left_when_the_command_ends_are_not_run(tmp_path):
    agent = make_agent(tmp_path, tool=LookupTool(delay=0.5), max_concurrent_calls=2)
    code = (
        "import threading, time\nfrom dynamiq_tools import call\n"
        "for k in range(10):\n"
        "    threading.Thread(target=call, args=('lookup', {'key': str(k)}), daemon=True).start()\n"
        "time.sleep(0.3)"
    )
    run_shell(agent, py(code))
    time.sleep(1.5)
    assert len(LookupTool.calls) <= 2


def test_without_the_bridge_the_client_says_so(tmp_path):
    sandbox = LocalSandbox(bridge_dir=str(tmp_path / "bridge"))
    agent = build_agent(tools=[LookupTool()], sandbox={"enabled": True, "backend": sandbox})
    observation = run_shell(agent, "dynamiq-tools list || echo missing")
    assert "missing" in observation
    assert threading.active_count() < 50


class ForeignShellSchema(BaseModel):
    command: str = Field(default="", description="Command.")


class ForeignShellTool(Node):
    """A tool that runs a command in a sandbox of its own, which the calling agent's bridge must not reach."""

    group: Literal[NodeGroup.TOOLS] = NodeGroup.TOOLS
    name: str = "foreign-shell"
    description: str = "Runs a command elsewhere."
    input_schema: ClassVar[type[ForeignShellSchema]] = ForeignShellSchema
    sandbox: LocalSandbox

    def execute(self, input_data: ForeignShellSchema, config: RunnableConfig | None = None, **kwargs) -> dict:
        result = SandboxShellTool(sandbox=self.sandbox).run(input_data={"command": input_data.command})
        return {"content": result.output["content"]}


def test_only_the_agents_own_shell_serves_its_tools(tmp_path):
    foreign = ForeignShellTool(sandbox=local_sandbox(tmp_path / "foreign"))
    agent = build_agent(tools=[LookupTool(), foreign])
    steps = [tool_step("foreign-shell", command="dynamiq-tools list || echo missing"), final_step("done")]
    with scripted(agent, MODE, steps) as recorder:
        agent.run(input_data={"input": "go"})
    assert "missing" in recorder.sent_text(1) and LookupTool.calls == []


def test_a_sandbox_shared_with_subagents_cannot_bridge(tmp_path):
    with pytest.raises(ValueError, match="share_sandbox_with_subagents"):
        build_agent(
            tools=[LookupTool()],
            sandbox={"enabled": True, "backend": local_sandbox(tmp_path)},
            share_sandbox_with_subagents=True,
        )


def test_tool_bridge_survives_a_yaml_round_trip(tmp_path):
    sandbox = E2BSandbox(connection=E2B(api_key="test-key"), tool_bridge={"enabled": True, "max_calls_per_command": 7})
    agent = build_agent(tools=[LookupTool()], sandbox={"enabled": True, "backend": sandbox})
    workflow = Workflow(flow=Flow(nodes=[agent]))
    path = tmp_path / "flow.yaml"
    workflow.to_yaml_file(path)

    for _ in range(2):
        loaded = Workflow.from_yaml_file(str(path), init_components=True)
        settings = loaded.flow.nodes[0].sandbox.backend.tool_bridge
        assert settings.enabled is True and settings.max_calls_per_command == 7
        loaded.to_yaml_file(path)


PLATFORM = """
import json, os, urllib.error, urllib.request
def api(path, method="GET", body=None, token=None, host=None):
    headers = {"Authorization": "Bearer " + (token or os.environ["DYNAMIQ_API_TOKEN"])}
    if host:
        headers["Host"] = host
    data = json.dumps(body).encode() if body is not None else None
    request = urllib.request.Request(os.environ["DYNAMIQ_API_BASE_URL"] + path, data, headers, method=method)
    try:
        return json.loads(urllib.request.urlopen(request).read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read()).get("error", {}).get("message")
"""


def warehouse(password="pw-1") -> SQLExecutor:
    connection = PostgreSQL(host="db.example", port=5432, database="dw", user="ro", password=password)
    with patch.object(SQLExecutor, "init_components"):
        return SQLExecutor(name="warehouse", connection=connection, is_postponed_component_init=True)


def sql_agent(tmp_path, hooks=None, **bridge):
    sandbox = local_sandbox(tmp_path, **bridge)
    with patch.object(SQLExecutor, "init_components"):
        return build_agent(
            hooks=hooks, tools=[LookupTool(), warehouse()], sandbox={"enabled": True, "backend": sandbox}
        )


def test_tools_cli_routes(tmp_path):
    agent = make_agent(tmp_path)
    code = PLATFORM + (
        "print('apps', api('/v1/internal/connectors'))\n"
        "print('tools', api('/v1/internal/connectors/agent/tools'))\n"
        "print('get', api('/v1/internal/connectors/agent/tools/lookup')['data']['name'])\n"
        "print('call', api('/v1/internal/connectors/agent/tools/lookup/call', 'POST', {'input': {'key': 'h'}}))\n"
        "print('missing', api('/v1/internal/connectors/agent/tools/nope'))\n"
        "print('bad token', api('/v1/internal/connectors', token='forged'))\n"
        "print('rebound host', api('/v1/internal/connectors', host='attacker.example'))"
    )
    observation = run_shell(agent, py(code))
    assert "apps {'data': [{'slug': 'agent', 'name': 'agent', 'description': None}]}" in observation
    assert "tools {'data': [{'name': 'lookup', 'description': 'Looks a key up.'}]}" in observation
    assert "get lookup" in observation and "call {'data': 'value of h'}" in observation
    assert "missing (404," in observation and "bad token (401," in observation and "rebound host (403," in observation


def test_sql_cli_lists_databases_but_gets_no_credentials_by_default(tmp_path):
    code = PLATFORM + (
        "print('list', api('/v1/internal/sql-cli/databases'))\n"
        "print('creds', api('/v1/internal/sql-cli/databases/warehouse/credentials'))"
    )
    observation = run_shell(sql_agent(tmp_path), py(code))
    assert "'slug': 'warehouse'" in observation and "'engine': 'postgres'" in observation
    assert "creds (403, \"This agent's sandbox gets no database credentials" in observation
    assert "pw-1" not in observation


def test_sql_cli_gets_credentials_where_allowed_and_each_grant_is_traced(tmp_path):
    code = PLATFORM + "print('creds', api('/v1/internal/sql-cli/databases/warehouse/credentials'))"
    tracing = TracingCallbackHandler()
    observation = run_shell(sql_agent(tmp_path, cli_credentials_allowed=True), py(code), callbacks=[tracing])

    assert "'engine': 'postgres', 'connection': {'host': 'db.example'" in observation
    assert "'password': 'pw-1'" in observation
    runs = list(tracing.runs.values())
    shell = next(r for r in runs if r.name == "sandbox-shell")
    grant = next(r for r in runs if r.name == "warehouse")
    assert grant.parent_run_id == shell.id and grant.input == {"credentials_for": "sql-cli"}


@pytest.mark.parametrize(
    "hook",
    [
        ToolPolicyHook(tools=["warehouse"], allow_if={"user_id": "admin"}),
        ToolPolicyHook(tools=["warehouse"], approval=True),
        CallLimitHook(max_per_run=5),
    ],
)
def test_no_credentials_past_a_hook_that_governs_the_tool(tmp_path, hook):
    code = PLATFORM + "print('creds', api('/v1/internal/sql-cli/databases/warehouse/credentials'))"
    observation = run_shell(sql_agent(tmp_path, hooks=[hook], cli_credentials_allowed=True), py(code))
    assert "creds (403, \"Hooks check every call of 'warehouse'" in observation and "pw-1" not in observation


def test_sql_cli_connection_payloads():
    mysql = MySQL(host="m.example", port=3306, database="d", user="u", password="p")
    assert sql_cli_connection(mysql)["ssl"] == {"verify_identity": False}

    key = {"private_key": "k", "client_email": "sa@p.iam.gserviceaccount.com", "token_uri": "https://t"}
    bigquery = BigQuery(project="billing", location="EU", service_account=key)
    credentials = SimpleNamespace(token="ya29.short", expiry=datetime(2026, 1, 1, 12), refresh=lambda request: None)
    with patch.object(BigQuery, "get_credentials", return_value=credentials) as get_credentials:
        payload = sql_cli_connection(bigquery)
    assert get_credentials.call_args.kwargs == {"scopes": ["https://www.googleapis.com/auth/bigquery"]}
    assert payload == {
        "project": "billing",
        "location": "EU",
        "maximum_bytes_billed": None,
        "access_token": "ya29.short",
        "expires_at": "2026-01-01T12:00:00Z",
    }
