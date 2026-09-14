import io
import re

from dynamiq.nodes.agents.utils import (
    ToolOutputSandboxPersistenceConfig,
    process_tool_output_for_agent,
    process_tool_output_with_sandbox_persistence,
    summarize_binary_tool_output,
)


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
    persistence_config = ToolOutputSandboxPersistenceConfig(dump_threshold_chars=8000, preview_chars=4000)

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
    persistence_config = ToolOutputSandboxPersistenceConfig(dump_threshold_chars=10000, preview_chars=4000)

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


def test_large_tool_output_persisted_to_sandbox_with_preview():
    sandbox = DummySandbox()
    content = "A" * 9000
    persistence_config = ToolOutputSandboxPersistenceConfig(dump_threshold_chars=8000, preview_chars=4000)

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

    assert result.startswith(f"Tool output saved to: {saved['file_path']}\n\nTool output preview:\n")
    assert result.endswith(content[:4000])


def test_large_tool_output_with_zero_preview_chars_returns_saved_path_only():
    sandbox = DummySandbox()
    content = "E" * 9000
    persistence_config = ToolOutputSandboxPersistenceConfig(dump_threshold_chars=8000, preview_chars=0)

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
    assert result == f"Tool output saved to: {saved['file_path']}"


def test_large_output_without_sandbox_falls_back_to_existing_truncation():
    content = "B" * 9000
    persistence_config = ToolOutputSandboxPersistenceConfig(dump_threshold_chars=8000, preview_chars=4000)

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
    persistence_config = ToolOutputSandboxPersistenceConfig(dump_threshold_chars=8000, preview_chars=4000)

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
    persistence_config = ToolOutputSandboxPersistenceConfig(dump_threshold_chars=8000, preview_chars=4000)

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


def test_binary_tool_output_is_summarized_instead_of_escaped():
    """A text-to-speech tool returns audio bytes as its content. Rendering those with str() fills
    the model's context with thousands of escape sequences and tells it nothing."""
    audio = b"\xff\xfb\x90\x64" * 2000

    result = process_tool_output_for_agent(audio)

    assert result == "<8000 bytes of binary data>"


def test_binary_values_inside_a_tool_output_do_not_break_serialization():
    """Any dict carrying bytes used to raise TypeError: Object of type bytes is not JSON serializable."""
    result = process_tool_output_for_agent({"content": b"\x00\x01", "mime_type": "audio/mpeg"})

    assert result == "<2 bytes of binary data>"

    result = process_tool_output_for_agent({"waveform": b"\x00\x01", "mime_type": "audio/mpeg"})

    assert "<2 bytes of binary data>" in result
    assert "audio/mpeg" in result


def test_binary_tool_output_is_summarized_with_the_file_it_produced():
    """The bytes are useless to the model, but knowing what was made and where it went is not."""
    audio = io.BytesIO(b"\xff\xfb\x90\x64")
    audio.name = "greeting.mp3"

    summary = summarize_binary_tool_output(
        {"content": b"\xff\xfb\x90\x64" * 2000, "files": [audio], "mime_type": "audio/mpeg"}
    )

    assert summary == "Produced audio/mpeg (8000 bytes), returned as file 'greeting.mp3'."


def test_binary_tool_output_summary_copes_with_no_file_and_no_type():
    assert summarize_binary_tool_output({"content": b"\x00\x01"}) == "Produced 2 bytes of binary data."


def test_a_text_body_that_happens_to_be_bytes_is_still_readable():
    """HttpApiCall leaves `content` as bytes for anything but an exact application/json header,
    so most JSON, HTML and XML bodies arrive here as bytes and the agent has to be able to read
    them."""
    body = b'{"answer": 42, "detail": "what the agent needed"}'

    assert process_tool_output_for_agent(body) == '{"answer": 42, "detail": "what the agent needed"}'
    assert process_tool_output_for_agent({"content": body}) == '{"answer": 42, "detail": "what the agent needed"}'


def test_genuine_binary_is_still_summarized():
    assert process_tool_output_for_agent(b"\xff\xfb\x90\x64" * 10) == "<40 bytes of binary data>"
    assert process_tool_output_for_agent(b"RIFF\x24\x00\x00\x00WAVE") == "<12 bytes of binary data>"
