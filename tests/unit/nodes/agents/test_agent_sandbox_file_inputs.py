"""Sandbox files that the model names in a tool's file arguments."""

import io
from typing import Any, ClassVar, Literal

import pytest
from pydantic import BaseModel, ConfigDict, Field

from dynamiq.nodes import Node, NodeGroup
from dynamiq.nodes.types import InferenceMode
from dynamiq.runnables import RunnableConfig, RunnableStatus
from dynamiq.sandboxes import SandboxConfig
from dynamiq.sandboxes.base import Sandbox
from dynamiq.sandboxes.exceptions import SandboxConnectionError
from dynamiq.storages.file import FileStoreConfig
from dynamiq.storages.file.in_memory import InMemoryFileStore
from tests.helpers.agent_hooks import ALL_MODES, build_agent, final_step, run_scripted, tool_step


class FileSandbox(Sandbox):
    """A sandbox that only holds files."""

    files: dict[str, bytes] = Field(default_factory=dict)

    def get_tools(self, llm: Any = None) -> list[Node]:
        return []

    def retrieve(self, file_path: str) -> bytes:
        if file_path not in self.files:
            raise FileNotFoundError(file_path)
        return self.files[file_path]

    def exists(self, file_path: str) -> bool:
        return file_path in self.files


class UploadInput(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    files: list[io.BytesIO | str] = Field(
        default_factory=list, description="Paths of the files to upload.", json_schema_extra={"map_from_storage": True}
    )


class Uploader(Node):
    group: Literal[NodeGroup.TOOLS] = NodeGroup.TOOLS
    name: str = "uploader"
    description: str = "Uploads files."
    is_files_allowed: bool = True
    input_schema: ClassVar[type[UploadInput]] = UploadInput
    uploaded: list[str] = Field(default_factory=list)

    def execute(self, input_data: UploadInput, config: RunnableConfig | None = None, **kwargs) -> dict[str, Any]:
        self.uploaded.extend(getattr(f, "name", str(f)) for f in input_data.files)
        return {"content": f"uploaded {len(input_data.files)} file(s)"}


@pytest.mark.parametrize("mode", ALL_MODES)
def test_a_missing_file_is_reported_to_the_model_which_can_fix_the_path(mode):
    sandbox = FileSandbox(files={"/home/user/report.csv": b"a,b\n"})
    uploader = Uploader()
    agent = build_agent(tools=[uploader], sandbox=SandboxConfig(enabled=True, backend=sandbox))
    steps = [
        tool_step("uploader", files=["/home/user/reprot.csv"]),
        tool_step("uploader", files=["/home/user/report.csv"]),
        final_step("uploaded"),
    ]

    result, recorder = run_scripted(agent, mode, steps)

    assert result.status == RunnableStatus.SUCCESS
    assert "/home/user/reprot.csv" in recorder.sent_text(1)
    assert uploader.uploaded == ["report.csv"]


class FileNotFoundException(Exception):
    """Named as the error E2B raises for a path that does not exist."""


class DaytonaError(Exception):
    """Named as the one error Daytona raises for a download that failed, whatever the reason."""


class SdkSandbox(FileSandbox):
    """A sandbox whose reads fail with an SDK's error."""

    error: Any = None

    def retrieve(self, file_path: str) -> bytes:
        if file_path not in self.files:
            raise self.error
        return self.files[file_path]


@pytest.mark.parametrize(
    "error",
    [
        FileNotFoundException("path '/home/user/reprot.csv' does not exist"),
        DaytonaError('Failed to download file: {"statusCode":404,"message":"file not found","code":"FILE_NOT_FOUND"}'),
    ],
    ids=["e2b", "daytona"],
)
def test_a_missing_file_is_recognised_as_the_sandbox_sdks_report_it(error):
    sandbox = SdkSandbox(files={"/home/user/report.csv": b"a,b\n"}, error=error)
    uploader = Uploader()
    agent = build_agent(tools=[uploader], sandbox=SandboxConfig(enabled=True, backend=sandbox))
    steps = [
        tool_step("uploader", files=["/home/user/reprot.csv"]),
        tool_step("uploader", files=["/home/user/report.csv"]),
        final_step("uploaded"),
    ]

    result, _ = run_scripted(agent, InferenceMode.FUNCTION_CALLING, steps)

    assert result.status == RunnableStatus.SUCCESS
    assert uploader.uploaded == ["report.csv"]


class ExpiredSandbox(FileSandbox):
    def retrieve(self, file_path: str) -> bytes:
        raise SandboxConnectionError("sbx-1")


@pytest.mark.parametrize(
    "sandbox",
    [ExpiredSandbox(), SdkSandbox(error=DaytonaError("Failed to download file: sandbox is not started"))],
    ids=["expired", "daytona_other_error"],
)
def test_a_sandbox_that_cannot_be_reached_ends_the_run_instead_of_blaming_the_path(sandbox):
    # The model cannot fix an expired sandbox by passing another path; telling it to would only loop.
    uploader = Uploader()
    agent = build_agent(tools=[uploader], sandbox=SandboxConfig(enabled=True, backend=sandbox))
    steps = [tool_step("uploader", files=["/home/user/report.csv"]), final_step("uploaded")]

    result, recorder = run_scripted(agent, InferenceMode.FUNCTION_CALLING, steps)

    assert result.status == RunnableStatus.FAILURE
    assert recorder.calls == 1
    assert uploader.uploaded == []


class UploadingSandbox(FileSandbox):
    def upload_file(self, file_name: str, content: bytes, destination_path=None, ensure_parent_dirs=True) -> str:
        path = destination_path or f"{self.base_path}/{file_name}"
        self.files[path] = content
        return path


class NoInput(BaseModel):
    pass


class ReportTool(Node):
    """Returns a generated file under whatever name it was given, as third-party tools may."""

    group: Literal[NodeGroup.TOOLS] = NodeGroup.TOOLS
    name: str = "report"
    description: str = "Generates a report file."
    input_schema: ClassVar[type[NoInput]] = NoInput
    file_name: str = "report.txt"

    def execute(self, input_data: NoInput, config: RunnableConfig | None = None, **kwargs) -> dict[str, Any]:
        file = io.BytesIO(b"generated")
        file.name = self.file_name
        return {"content": "report generated", "files": [file]}


@pytest.mark.parametrize(
    "file_name, saved_as",
    [
        ("../../etc/escape.txt", "etc/escape.txt"),
        ("/etc/escape.txt", "etc/escape.txt"),
        ("C:\\Users\\me\\chart.png", "Users/me/chart.png"),
        ("._agent/todos.json", "todos.json"),
    ],
)
def test_a_generated_file_is_saved_inside_the_sandbox_workspace(file_name, saved_as):
    sandbox = UploadingSandbox()
    agent = build_agent(tools=[ReportTool(file_name=file_name)], sandbox=SandboxConfig(enabled=True, backend=sandbox))

    result, _ = run_scripted(agent, InferenceMode.FUNCTION_CALLING, [tool_step("report"), final_step("done")])

    assert result.status == RunnableStatus.SUCCESS
    saved = [path for path in sandbox.files if sandbox.files[path] == b"generated"]
    assert saved == [f"{sandbox.base_path}/{saved_as}"]


def test_generated_files_that_share_a_file_name_do_not_overwrite_each_other_or_an_upload():
    file_store = InMemoryFileStore()
    file_store.store("result.csv", b"uploaded")
    tools = [
        ReportTool(name="emea", file_name="/tmp/export/emea/result.csv"),
        ReportTool(name="apac", file_name="/tmp/export/apac/result.csv"),
    ]
    agent = build_agent(tools=tools, file_store=FileStoreConfig(enabled=True, backend=file_store))

    run_scripted(agent, InferenceMode.FUNCTION_CALLING, [tool_step("emea"), tool_step("apac"), final_step("done")])

    assert file_store.retrieve("result.csv") == b"uploaded"
    assert file_store.exists("tmp/export/emea/result.csv")
    assert file_store.exists("tmp/export/apac/result.csv")


def test_a_generated_file_keeps_its_folders_inside_the_workspace():
    sandbox = UploadingSandbox()
    tool = ReportTool(file_name="charts/q3.png")
    agent = build_agent(tools=[tool], sandbox=SandboxConfig(enabled=True, backend=sandbox))

    run_scripted(agent, InferenceMode.FUNCTION_CALLING, [tool_step("report"), final_step("done")])

    saved = [path for path in sandbox.files if sandbox.files[path] == b"generated"]
    assert saved == [f"{sandbox.base_path}/charts/q3.png"]
