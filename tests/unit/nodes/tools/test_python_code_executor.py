import io
import os
import tempfile

import pytest

from dynamiq.nodes.tools.python_code_executor import PythonCodeExecutor, PythonCodeExecutorFileWorkspace
from dynamiq.runnables import RunnableStatus
from dynamiq.storages.file.in_memory import InMemoryFileStore


def _create_executor():
    """Create a PythonCodeExecutor instance for testing."""
    file_store = InMemoryFileStore()
    return PythonCodeExecutor(file_store=file_store)


@pytest.mark.parametrize(
    "path,should_contain",
    [
        ("data.txt", "data.txt"),
        ("subdir/file.txt", "subdir"),
        ("./normal/../file.txt", "file.txt"),
    ],
)
def test_sanitize_workspace_path_valid_paths_allowed(path, should_contain):
    """Valid paths should be allowed and returned as absolute paths within workspace."""
    executor = _create_executor()
    with tempfile.TemporaryDirectory() as workspace_dir:
        result = executor._sanitize_workspace_path(path, workspace_dir, "fallback")
        assert result is not None
        assert should_contain in result
        assert result.startswith(workspace_dir)


@pytest.mark.parametrize(
    "path",
    [
        "../etc/passwd",
        "subdir/../../../etc/passwd",
        "a/b/c/../../../../../../../etc/passwd",
    ],
)
def test_sanitize_workspace_path_traversal_is_blocked(path):
    """Paths with path traversal should be blocked."""
    executor = _create_executor()
    with tempfile.TemporaryDirectory() as workspace_dir:
        result = executor._sanitize_workspace_path(path, workspace_dir, "fallback")
        assert result is None


def test_sanitize_workspace_path_absolute_path_is_handled():
    """Absolute paths should be converted to relative and validated."""
    executor = _create_executor()
    with tempfile.TemporaryDirectory() as workspace_dir:
        result = executor._sanitize_workspace_path("/etc/passwd", workspace_dir, "fallback")
        # After lstrip(os.sep), it becomes "etc/passwd" which should be within workspace
        assert result is not None
        assert result.startswith(workspace_dir)


def test_sanitize_workspace_path_empty_path_uses_fallback():
    """Empty paths after normalization should use the fallback name."""
    executor = _create_executor()
    with tempfile.TemporaryDirectory() as workspace_dir:
        result = executor._sanitize_workspace_path("", workspace_dir, "fallback_file")
        assert result is not None
        assert result == os.path.join(os.path.abspath(workspace_dir), "fallback_file")


def test_sanitize_workspace_path_dot_path_uses_fallback():
    """Dot path (current directory) should use the fallback name to avoid IsADirectoryError."""
    executor = _create_executor()
    with tempfile.TemporaryDirectory() as workspace_dir:
        result = executor._sanitize_workspace_path(".", workspace_dir, "fallback_file")
        assert result is not None
        assert result == os.path.join(os.path.abspath(workspace_dir), "fallback_file")


def test_materialize_inline_files_malicious_filename_is_sanitized():
    """Files with path traversal in name should not escape workspace."""
    executor = _create_executor()
    with tempfile.TemporaryDirectory() as workspace_dir:
        malicious_file = io.BytesIO(b"malicious content")
        malicious_file.name = "../../../etc/cron.d/malicious"

        executor._materialize_inline_files([malicious_file], workspace_dir)

        assert not os.path.exists("/etc/cron.d/malicious")

        for root, dirs, files in os.walk(workspace_dir):
            for f in files:
                full_path = os.path.join(root, f)
                assert full_path.startswith(workspace_dir)


def test_code_executor_captures_stdout_without_print_errors():
    """Ensure the RestrictedPython print hooks no longer crash and stdout is captured."""
    file_store = InMemoryFileStore()
    executor = PythonCodeExecutor(file_store=file_store)
    code = """
def run():
    print("hello", 123)
    return {"message": "ok"}
"""

    result = executor.run({"code": code})

    assert result.status == RunnableStatus.SUCCESS
    output = result.output["content"]
    assert output["message"] == "ok"
    assert output["stdout"].strip() == "hello 123"


def test_code_executor_allows_dunder_name_access():
    """type(exc).__name__ should be accessible to user code for better error reporting."""
    file_store = InMemoryFileStore()
    executor = PythonCodeExecutor(file_store=file_store)
    code = """
def run():
    try:
        raise ValueError("boom")
    except Exception as exc:
        return {"error_type": type(exc).__name__}
"""

    result = executor.run({"code": code})

    assert result.status == RunnableStatus.SUCCESS
    assert result.output["content"]["error_type"] == "ValueError"


def test_workspace_read_file_auto_handles_binary_and_text():
    """read_file helper should keep binary payloads intact while decoding text."""
    file_store = InMemoryFileStore()
    file_store.store("notes/data.txt", "plain text payload")
    file_store.store("reports/data.xlsx", b"\x00\x01binary-bytes")

    workspace = PythonCodeExecutorFileWorkspace(file_store=file_store)

    assert workspace.read("notes/data.txt") == "plain text payload"
    assert workspace.read("reports/data.xlsx") == b"\x00\x01binary-bytes"


def test_code_executor_reads_files_via_helper_function():
    """Code snippets should rely on the injected read_file helper to access uploaded artifacts."""
    file_store = InMemoryFileStore()
    file_store.store("transactions.csv", "amount\n10\n25\n")

    executor = PythonCodeExecutor(file_store=file_store)
    code = """
import pandas as pd
from io import StringIO

def run():
    csv_text = read_file('transactions.csv')
    df = pd.read_csv(StringIO(csv_text))
    return {'rows': len(df), 'total': float(df['amount'].sum())}
"""

    result = executor.run({"code": code})

    assert result.status == RunnableStatus.SUCCESS
    payload = result.output["content"]
    assert payload["rows"] == 2
    assert payload["total"] == 35.0
