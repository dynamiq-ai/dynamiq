import uuid
from datetime import datetime
from io import BytesIO

import pytest

from dynamiq.connections import OpenAI as OpenAIConnection
from dynamiq.nodes.llms import OpenAI
from dynamiq.nodes.node import Node
from dynamiq.nodes.tools.file_tools import (
    EXTRACTED_TEXT_SUFFIX,
    EditOperation,
    FileListTool,
    FileReadInputSchema,
    FileReadTool,
    FileType,
    FileWriteInputSchema,
    FileWriteTool,
    validate_file_path,
)
from dynamiq.runnables import RunnableResult, RunnableStatus
from dynamiq.sandboxes.base import Sandbox
from dynamiq.storages.file.in_memory import InMemoryFileStore


@pytest.mark.parametrize(
    "path,expected",
    [
        ("file.txt", "file.txt"),
        ("subdir/file.txt", "subdir/file.txt"),
        ("a/b/c/file.txt", "a/b/c/file.txt"),
        ("", ""),
        ("subdir/./file.txt", "subdir/file.txt"),
    ],
)
def test_validate_file_path_valid_paths(path, expected):
    """Valid paths should be allowed and normalized."""
    assert validate_file_path(path) == expected


@pytest.mark.parametrize(
    "path,error_match",
    [
        ("../etc/passwd", "Path traversal"),
        ("subdir/../../../etc/passwd", "Path traversal"),
        ("a/b/c/../../../../../../../etc/passwd", "Path traversal"),
        ("/etc/passwd", "Absolute paths"),
        ("C:\\Windows\\System32\\config", "Absolute paths"),
    ],
)
def test_validate_file_path_rejects_dangerous_paths(path, error_match):
    """Dangerous paths (traversal, absolute) should be rejected."""
    with pytest.raises(ValueError, match=error_match):
        validate_file_path(path)


@pytest.mark.parametrize(
    "path",
    [
        "../../../etc/passwd",
        "/etc/passwd",
        "subdir/../../../secret.txt",
    ],
)
def test_file_read_input_schema_rejects_dangerous_paths(path):
    """FileReadInputSchema should reject path traversal and absolute paths."""
    with pytest.raises(ValueError):
        FileReadInputSchema(file_path=path, brief="Read file")


@pytest.mark.parametrize(
    "path",
    [
        "../../../tmp/malicious.sh",
        "/tmp/file.txt",
        "subdir/../../../secret.txt",
    ],
)
def test_file_write_input_schema_rejects_dangerous_paths(path):
    """FileWriteInputSchema should reject path traversal and absolute paths."""
    with pytest.raises(ValueError):
        FileWriteInputSchema(file_path=path, content="content", action="write", brief="Write content")


def test_file_read_input_schema_valid_path_is_accepted():
    """Valid relative paths should be accepted."""
    schema = FileReadInputSchema(file_path="documents/report.pdf", brief="Read report")
    assert schema.file_path == "documents/report.pdf"


def test_file_write_input_schema_valid_path_is_accepted():
    """Valid relative paths should be accepted."""
    schema = FileWriteInputSchema(file_path="output/result.json", content="test", action="write", brief="Write test")
    assert schema.file_path == "output/result.json"


@pytest.fixture
def file_store():
    """Create an in-memory file storage instance for testing."""
    return InMemoryFileStore()


@pytest.fixture
def sample_file_path():
    """Sample file path for testing."""
    return "test/file.txt"


@pytest.fixture
def llm_model():
    connection = OpenAIConnection(id=str(uuid.uuid4()), api_key="api-key")
    return OpenAI(name="OpenAI", model="gpt-4o-mini", connection=connection)


def test_file_read_tool(file_store, sample_file_path, llm_model):
    """Test FileReadTool functionality including initialization, successful read, and error handling."""
    # Test initialization
    tool = FileReadTool(file_store=file_store, llm=llm_model)
    assert tool.name == "FileReadTool"
    assert tool.group == "tools"
    assert tool.file_store == file_store

    # Create a test file first
    test_content = "Hello, this is test content!"
    file_store.store(sample_file_path, test_content)

    # Test successful read
    input_data = {"file_path": sample_file_path, "brief": "Read test file"}
    result = tool.run(input_data)

    assert isinstance(result, RunnableResult)
    assert result.status == RunnableStatus.SUCCESS
    # FileReadTool now returns text content (string) after processing
    assert result.output["content"] == test_content


def test_file_write_tool(file_store):
    """Test FileWriteTool functionality including initialization, successful writes, and error handling."""
    # Test initialization
    tool = FileWriteTool(file_store=file_store)
    assert tool.name == "FileWriteTool"
    assert tool.group == "tools"
    assert tool.file_store == file_store

    # Test successful text write
    input_data = {
        "file_path": "test/output.txt",
        "content": "Hello, World!",
        "action": "write",
        "brief": "Write a test file",
    }
    result = tool.run(input_data)

    assert result.status == RunnableStatus.SUCCESS
    assert "written successfully" in result.output["content"]

    # Verify file was actually written
    assert file_store.exists("test/output.txt")
    stored_content = file_store.retrieve("test/output.txt")
    assert stored_content == b"Hello, World!"

    # Test custom content type
    input_data = {
        "file_path": "test/data.csv",
        "content": "name,age\nJohn,30",
        "content_type": "text/csv",
        "action": "write",
        "brief": "Write CSV data",
    }

    result = tool.run(input_data)
    assert result.status == RunnableStatus.SUCCESS

    # Verify file with custom content type
    assert file_store.exists("test/data.csv")
    stored_content = file_store.retrieve("test/data.csv")
    assert stored_content == b"name,age\nJohn,30"


def test_file_list_tool(file_store):
    """Test FileListTool functionality including initialization, listing files, and empty directory."""
    # Test initialization
    tool = FileListTool(file_store=file_store)
    assert tool.name == "FileListTool"
    assert tool.group == "tools"
    assert tool.file_store == file_store

    # Create some test files
    file_store.store("test/file1.txt", "content1")
    file_store.store("test/file2.txt", "content2")
    file_store.store("docs/readme.md", "readme content")

    # Test listing files from root
    input_data = {"file_path": "", "recursive": True}
    result = tool.run(input_data)

    assert result.status == RunnableStatus.SUCCESS
    assert "Files currently available" in result.output["content"]
    assert "file1.txt" in result.output["content"]
    assert "file2.txt" in result.output["content"]
    assert "readme.md" in result.output["content"]

    # Test listing from specific directory
    input_data = {"file_path": "test/", "recursive": False}
    result = tool.run(input_data)

    assert result.status == RunnableStatus.SUCCESS
    assert "file1.txt" in result.output["content"]
    assert "file2.txt" in result.output["content"]
    assert "readme.md" not in result.output["content"]  # Should not be in test/ directory

    # Test empty directory
    input_data = {"file_path": "empty/", "recursive": False}
    result = tool.run(input_data)

    assert result.status == RunnableStatus.SUCCESS
    assert "Files currently available" in result.output["content"]
    assert "File: " not in result.output["content"]


def test_file_tools_integration(file_store, llm_model):
    """Test file tools working together."""
    # Test tools working together
    write_tool = FileWriteTool(file_store=file_store)
    read_tool = FileReadTool(file_store=file_store, llm=llm_model)

    # Write file
    write_input = {
        "file_path": "test/integration.txt",
        "content": "Integration test content",
        "action": "write",
        "brief": "Write integration test file",
    }
    write_result = write_tool.run(write_input)
    assert write_result.status == RunnableStatus.SUCCESS

    # Read file
    read_input = {"file_path": "test/integration.txt", "brief": "Read integration file"}
    read_result = read_tool.run(read_input)
    assert read_result.status == RunnableStatus.SUCCESS
    assert read_result.output["content"] == "Integration test content"

    # Test different storage instances
    storage2 = InMemoryFileStore()
    storage2.store("test.txt", "Content from storage 2")

    read_tool2 = FileReadTool(file_store=storage2, llm=llm_model)
    input_data = {"file_path": "test.txt", "brief": "Read from storage 2"}

    result2 = read_tool2.run(input_data)
    assert result2.output["content"] == "Content from storage 2"


def test_file_read_tool_appends_hint_for_non_text(monkeypatch, file_store, llm_model):
    """Ensure cache hint is shown for non-plain-text conversions."""
    tool = FileReadTool(file_store=file_store, llm=llm_model)
    file_path = "docs/sample.pdf"
    cache_path = f"{file_path}{EXTRACTED_TEXT_SUFFIX}"
    file_store.store(file_path, b"%PDF-FAKE%")

    monkeypatch.setattr(tool, "_detect_file_type", lambda *args, **kwargs: FileType.PDF)
    monkeypatch.setattr(tool, "_process_file_with_converter", lambda *args, **kwargs: ("Converted PDF text", None))
    monkeypatch.setattr(tool, "_persist_extracted_text", lambda *args, **kwargs: cache_path)

    result = tool.run({"file_path": file_path, "brief": "Read PDF"})

    assert result.status == RunnableStatus.SUCCESS
    assert result.output["cached_text_path"] == cache_path
    expected_hint = (
        f"\n\n[Extracted text cached at '{cache_path}'. "
        "Use FileSearchTool to search this processed content without re-reading the original file.]"
    )
    assert result.output["content"].endswith(expected_hint)


def test_file_read_tool_limits_spreadsheet_preview(file_store, llm_model):
    """Large spreadsheets should be summarized instead of dumping raw XML."""
    tool = FileReadTool(file_store=file_store, llm=llm_model)

    import pandas as pd

    df = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=60, freq="D"),
            "amount": range(60),
            "type": ["debit" if i % 2 == 0 else "credit" for i in range(60)],
        }
    )

    buffer = BytesIO()
    df.to_excel(buffer, index=False)
    buffer.seek(0)
    file_store.store("data/transactions.xlsx", buffer.getvalue())

    result = tool.run({"file_path": "data/transactions.xlsx", "brief": "Read spreadsheet"})

    assert result.status == RunnableStatus.SUCCESS
    content = result.output["content"]
    assert "Spreadsheet preview" in content
    assert "Rows: 60" in content
    assert "showing up to 5 row(s)" in content


def test_file_read_tool_with_sandbox_like_backend(llm_model):
    """FileReadTool should work with sandbox-style storage backend."""

    class FakeSandbox(Sandbox):
        _data: dict[str, bytes] = {}

        def exists(self, file_path: str) -> bool:
            return file_path in self._data

        def retrieve(self, file_path: str) -> bytes:
            return self._data[file_path]

        def get_tools(self, llm=None) -> list[Node]:
            return []

    sandbox = FakeSandbox()
    sandbox._data["notes/readme.txt"] = b"sandbox text content"
    sandbox._data["/home/user/scores.csv"] = b"name,score\nAlice,95"

    tool = FileReadTool(file_store=sandbox, llm=llm_model, absolute_file_paths_allowed=True)

    # Relative path
    result = tool.run({"file_path": "notes/readme.txt", "brief": "Read readme"})
    assert result.status == RunnableStatus.SUCCESS
    assert result.output["content"] == "sandbox text content"

    # Absolute path — allowed via absolute_file_paths_allowed=True
    result = tool.run({"file_path": "/home/user/scores.csv", "brief": "Read scores"})
    assert result.status == RunnableStatus.SUCCESS
    assert "Alice" in result.output["content"]


# ---------------------------------------------------------------------------
# FileWriteTool – edit mode tests
# ---------------------------------------------------------------------------


def test_edit_sequential_and_replace_all(file_store):
    """Multiple edits applied in order; all=True replaces every occurrence."""
    file_store.store("cfg.ini", b"host=localhost\nport=foo\nlog=foo\n")
    tool = FileWriteTool(file_store=file_store)

    result = tool.run(
        {
            "action": "edit",
            "file_path": "cfg.ini",
            "edits": [
                {"find": "localhost", "replace": "0.0.0.0"},
                {"find": "foo", "replace": "bar", "replace_all": True},
            ],
            "brief": "Update host and replace foo",
        }
    )

    assert result.status == RunnableStatus.SUCCESS
    assert file_store.retrieve("cfg.ini") == b"host=0.0.0.0\nport=bar\nlog=bar\n"
    assert "2 of 2 edit(s)" in result.output["content"]


def test_edit_find_not_found_aborts(file_store):
    """If a find string doesn't exist in the original file, abort with no changes."""
    file_store.store("readme.md", b"# Title\n")
    tool = FileWriteTool(file_store=file_store)

    result = tool.run(
        {
            "action": "edit",
            "file_path": "readme.md",
            "edits": [{"find": "NONEXISTENT", "replace": "X"}],
            "brief": "Try to edit missing text",
        }
    )

    assert result.status == RunnableStatus.FAILURE
    assert file_store.retrieve("readme.md") == b"# Title\n"


def test_edit_prior_edit_removes_later_find_skips_with_warning(file_store):
    """When an earlier edit consumes text a later edit targets, it is skipped with a warning."""
    file_store.store("code.py", b"old_func()\n")
    tool = FileWriteTool(file_store=file_store)

    result = tool.run(
        {
            "action": "edit",
            "file_path": "code.py",
            "edits": [
                {"find": "old_func()", "replace": "new_func()"},
                {"find": "old_func()", "replace": "another_func()"},
            ],
            "brief": "Rename with conflict",
        }
    )

    assert result.status == RunnableStatus.SUCCESS
    assert file_store.retrieve("code.py") == b"new_func()\n"
    assert "1 of 2 edit(s)" in result.output["content"]
    assert "Warning" in result.output["content"]


def test_edit_validation_rejects_invalid_inputs():
    """Empty find string, edit without edits list, and write without content are rejected."""
    with pytest.raises(ValueError):
        EditOperation(find="", replace="x")

    with pytest.raises(ValueError):
        FileWriteInputSchema(action="edit", file_path="t.txt", brief="no edits")

    with pytest.raises(ValueError):
        FileWriteInputSchema(action="write", file_path="t.txt", brief="no content")


def test_file_info_model_dump_json_serializes_bytes_as_base64():
    """FileInfo.model_dump(mode='json') must encode content as base64 and datetime as ISO string."""
    import base64
    import json

    from dynamiq.storages.file.base import FileInfo

    raw = b"\x89PNG\r\n\x1a\n"
    info = FileInfo(
        name="image.png",
        path="/tmp/image.png",
        size=len(raw),
        content_type="image/png",
        content=raw,
    )

    dumped = info.model_dump(mode="json")

    assert isinstance(dumped["content"], str)
    assert base64.b64decode(dumped["content"]) == raw

    assert isinstance(dumped["created_at"], str)

    json.dumps(dumped)

    # model_dump() without mode="json" returns raw Python types
    raw_dumped = info.model_dump()
    assert isinstance(raw_dumped["content"], bytes)
    assert raw_dumped["content"] == raw
    assert isinstance(raw_dumped["created_at"], datetime)

    none_info = FileInfo(name="x", path="x", size=0, content=None)
    none_dumped = none_info.model_dump(mode="json")
    assert none_dumped["content"] is None
    json.dumps(none_dumped)


@pytest.mark.parametrize(
    "kwargs,error_match",
    [
        ({"start_line": 0}, "start_line must be >= 1"),
        ({"end_line": 0}, "end_line must be >= 1"),
        ({"start_line": 10, "end_line": 5}, "end_line must be >= start_line"),
        ({"start_page": 0}, "start_page must be >= 1"),
        ({"end_page": 0}, "end_page must be >= 1"),
        ({"start_page": 5, "end_page": 2}, "end_page must be >= start_page"),
        ({"start_page": 1, "start_line": 1}, "mutually exclusive"),
    ],
)
def test_input_schema_rejects_invalid_line_page_params(kwargs, error_match):
    with pytest.raises(ValueError, match=error_match):
        FileReadInputSchema(file_path="f.txt", brief="read", **kwargs)


def test_input_schema_valid_line_and_page_ranges():
    """Valid ranges are accepted; start_page forces document_mode='page'."""
    line_schema = FileReadInputSchema(file_path="f.txt", start_line=5, end_line=10, brief="read")
    assert line_schema.start_line == 5
    assert line_schema.end_line == 10

    page_schema = FileReadInputSchema(file_path="f.pdf", start_page=3, brief="read")
    assert page_schema.start_page == 3
    assert page_schema.end_page is None
    assert page_schema.document_mode == "page"


def test_slice_lines_partial_range_and_header():
    text = "line1\nline2\nline3\nline4\nline5\n"
    sliced, total, start, end = FileReadTool._slice_lines(text, 2, 4, "test.txt")
    assert total == 5
    assert (start, end) == (2, 4)
    assert "--- Lines 2-4 of 5" in sliced
    assert "line2" in sliced
    assert "line4" in sliced
    assert "line1" not in sliced
    assert "line5" not in sliced


def test_slice_lines_defaults_and_clamping():
    """start_line=None defaults to 1, end_line beyond total is clamped."""
    text = "a\nb\nc\n"
    sliced, total, start, end = FileReadTool._slice_lines(text, None, 999, "t.txt")
    assert (start, end) == (1, 3)
    assert "a" in sliced and "c" in sliced


def test_slice_lines_out_of_bounds():
    text = "line1\nline2\n"
    sliced, total, _, _ = FileReadTool._slice_lines(text, 100, None, "small.txt")
    assert total == 2
    assert "only has 2 line(s)" in sliced
    assert "line1" not in sliced


def test_filter_page_range_selects_and_clamps():
    """Selects requested range, clamps end, includes header with totals."""
    entries = [{"page": i, "content": f"Page {i}", "metadata": {}} for i in range(1, 5)]

    text, filtered, total, start, end = FileReadTool._filter_page_range(entries, 2, 3, "d.pdf")
    assert (total, start, end) == (4, 2, 3)
    assert len(filtered) == 2
    assert "Pages 2-3 of 4" in text
    assert "Page 2" in text and "Page 3" in text
    assert "Page 1" not in text and "Page 4" not in text

    text2, _, _, _, end2 = FileReadTool._filter_page_range(entries, 1, 999, "d.pdf")
    assert end2 == 4 and len(text2) > 0


def test_filter_page_range_out_of_bounds():
    entries = [{"page": 1, "content": "Only", "metadata": {}}]
    text, filtered, total, _, _ = FileReadTool._filter_page_range(entries, 10, 12, "d.pdf")
    assert total == 1
    assert filtered == []
    assert "only has 1 page(s)" in text


def test_file_read_tool_line_range(file_store, llm_model):
    """Line range returns correct slice with metadata."""
    content = "".join(f"line {i}\n" for i in range(1, 21))
    file_store.store("data.txt", content.encode())

    tool = FileReadTool(file_store=file_store, llm=llm_model)
    result = tool.run({"file_path": "data.txt", "start_line": 5, "end_line": 8, "brief": "Read lines"})

    assert result.status == RunnableStatus.SUCCESS
    assert "line 5" in result.output["content"]
    assert "line 8" in result.output["content"]
    assert "line 4" not in result.output["content"]
    assert result.output["total_lines"] == 20
    assert result.output["line_range"] == [5, 8]


def test_file_read_tool_line_range_skips_chunking(file_store, llm_model):
    """Line range on a large file returns full slice, never chunked."""
    content = "".join(f"{'x' * 200} line {i}\n" for i in range(1, 101))
    file_store.store("big.txt", content.encode())

    tool = FileReadTool(file_store=file_store, llm=llm_model, max_size=500)
    result = tool.run({"file_path": "big.txt", "start_line": 10, "end_line": 20, "brief": "Read lines"})

    assert result.status == RunnableStatus.SUCCESS
    assert "CHUNKED" not in result.output["content"]
    assert "line 10" in result.output["content"]


def test_file_read_tool_line_range_out_of_bounds(file_store, llm_model):
    """Lines beyond file length produce a clear message."""
    file_store.store("tiny.txt", b"one\ntwo\n")
    tool = FileReadTool(file_store=file_store, llm=llm_model)
    result = tool.run({"file_path": "tiny.txt", "start_line": 50, "brief": "Read past end"})

    assert result.status == RunnableStatus.SUCCESS
    assert "only has 2 line(s)" in result.output["content"]


def test_file_read_tool_page_range(monkeypatch, file_store, llm_model):
    """Page range filters converter output and includes header."""
    tool = FileReadTool(file_store=file_store, llm=llm_model)
    file_store.store("report.pdf", b"%PDF-FAKE%")

    page_entries = [
        {"page": 1, "content": "First page", "metadata": {}},
        {"page": 2, "content": "Second page", "metadata": {}},
        {"page": 3, "content": "Third page", "metadata": {}},
    ]
    full_text = "\n\n".join(f"=== PAGE {e['page']} ===\n{e['content']}" for e in page_entries)

    monkeypatch.setattr(tool, "_detect_file_type", lambda *a, **kw: FileType.PDF)
    monkeypatch.setattr(tool, "_process_file_with_converter", lambda *a, **kw: (full_text, page_entries))

    result = tool.run({"file_path": "report.pdf", "start_page": 2, "end_page": 2, "brief": "Read p2"})

    assert result.status == RunnableStatus.SUCCESS
    assert "Second page" in result.output["content"]
    assert "First page" not in result.output["content"]
    assert "Pages 2-2 of 3" in result.output["content"]
    assert result.output["total_pages"] == 3


def test_file_read_tool_page_on_non_pdf_warns(monkeypatch, file_store, llm_model):
    """start_page on non-PDF warns and returns full content."""
    tool = FileReadTool(file_store=file_store, llm=llm_model)
    file_store.store("doc.docx", b"FAKE-DOCX")

    monkeypatch.setattr(tool, "_detect_file_type", lambda *a, **kw: FileType.DOCX_DOCUMENT)
    monkeypatch.setattr(tool, "_process_file_with_converter", lambda *a, **kw: ("Full docx text", None))

    result = tool.run({"file_path": "doc.docx", "start_page": 1, "brief": "Read page of docx"})

    assert result.status == RunnableStatus.SUCCESS
    assert "only supported for PDF" in result.output["content"]
    assert "Full docx text" in result.output["content"]
