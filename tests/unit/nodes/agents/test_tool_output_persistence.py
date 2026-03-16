import re

from dynamiq.nodes.agents.utils import ToolOutputSandboxPersistenceConfig, process_tool_output_with_sandbox_persistence


class DummySandbox:
    def __init__(self):
        self.saved = []

    def store(self, file_path, content, content_type=None, metadata=None, overwrite=False):
        self.saved.append(
            {
                "file_path": file_path,
                "content": content,
                "content_type": content_type,
                "metadata": metadata or {},
                "overwrite": overwrite,
            }
        )


def test_tool_output_not_persisted_when_under_threshold():
    sandbox = DummySandbox()
    content = "small output"
    persistence_config = ToolOutputSandboxPersistenceConfig(dump_threshold_chars=8000, summary_chars=4000)

    result = process_tool_output_with_sandbox_persistence(
        content=content,
        tool_name="SandboxShellTool",
        tool_input={"command": "echo hi"},
        sandbox=sandbox,
        save_tool_output_to_sandbox=True,
        sandbox_persistence_config=persistence_config,
    )

    assert result == content
    assert sandbox.saved == []


def test_under_threshold_still_respects_max_tokens_truncation():
    sandbox = DummySandbox()
    content = "X" * 8000
    persistence_config = ToolOutputSandboxPersistenceConfig(dump_threshold_chars=10000, summary_chars=4000)

    result = process_tool_output_with_sandbox_persistence(
        content=content,
        tool_name="SandboxShellTool",
        tool_input={"command": "echo large"},
        sandbox=sandbox,
        save_tool_output_to_sandbox=True,
        sandbox_persistence_config=persistence_config,
        max_tokens=100,
        truncate=True,
    )

    # max_tokens=100 => 400 chars effective limit; function truncates with marker in the middle.
    assert "[Content truncated]" in result
    assert len(result) <= 400
    assert sandbox.saved == []


def test_large_tool_output_persisted_to_sandbox_with_summary():
    sandbox = DummySandbox()
    content = "A" * 9000
    persistence_config = ToolOutputSandboxPersistenceConfig(dump_threshold_chars=8000, summary_chars=4000)

    result = process_tool_output_with_sandbox_persistence(
        content=content,
        tool_name="SandboxShellTool",
        tool_input={"command": "tools-cli tool list google-calendar"},
        sandbox=sandbox,
        save_tool_output_to_sandbox=True,
        sandbox_persistence_config=persistence_config,
    )

    assert len(sandbox.saved) == 1
    saved = sandbox.saved[0]

    assert saved["content"] == content
    assert saved["content_type"] == "text/plain"
    assert saved["overwrite"] is True
    assert saved["metadata"]["source"] == "agent_tool_output"
    assert saved["metadata"]["tool_name"] == "SandboxShellTool"

    assert re.match(
        r"^/home/user/\.tools/sandbox-shell-tool/tools-cli-tool-list/"
        r"\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}_\d{6}_tools-cli-tools\.txt$",
        saved["file_path"],
    )

    assert result.startswith(f"Tool output saved to: {saved['file_path']}\n\nTool output summary:\n")
    assert result.endswith(content[:4000])


def test_large_output_without_sandbox_falls_back_to_existing_truncation():
    content = "B" * 9000
    persistence_config = ToolOutputSandboxPersistenceConfig(dump_threshold_chars=8000, summary_chars=4000)

    result = process_tool_output_with_sandbox_persistence(
        content=content,
        tool_name="AnyTool",
        tool_input={"command": "echo large"},
        sandbox=None,
        save_tool_output_to_sandbox=True,
        sandbox_persistence_config=persistence_config,
        max_tokens=1000,
        truncate=True,
    )

    assert "[Content truncated]" in result


def test_large_output_not_persisted_when_tool_opt_out():
    sandbox = DummySandbox()
    content = "C" * 9000
    persistence_config = ToolOutputSandboxPersistenceConfig(dump_threshold_chars=8000, summary_chars=4000)

    result = process_tool_output_with_sandbox_persistence(
        content=content,
        tool_name="SandboxShellTool",
        tool_input={"command": "echo large"},
        sandbox=sandbox,
        save_tool_output_to_sandbox=False,
        sandbox_persistence_config=persistence_config,
        max_tokens=1000,
        truncate=True,
    )

    assert sandbox.saved == []
    assert "[Content truncated]" in result


def test_truncate_false_returns_full_content_with_sandbox_enabled():
    sandbox = DummySandbox()
    content = "D" * 9000
    persistence_config = ToolOutputSandboxPersistenceConfig(dump_threshold_chars=8000, summary_chars=4000)

    result = process_tool_output_with_sandbox_persistence(
        content=content,
        tool_name="SandboxShellTool",
        tool_input={"command": "echo large"},
        sandbox=sandbox,
        save_tool_output_to_sandbox=True,
        sandbox_persistence_config=persistence_config,
        max_tokens=1000,
        truncate=False,
    )

    assert result == content
    assert sandbox.saved == []
