import io
import os
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from dynamiq.nodes.node import ErrorHandling
from dynamiq.nodes.tools import python_code_executor
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


def test_default_description_lists_importable_modules_not_whole_stdlib():
    """The default refuses open() and every non-allow-listed import, so the description points at the
    read_file() helpers and names the importable modules rather than claiming the whole standard library
    or promising pd.read_csv."""
    executor = PythonCodeExecutor(file_store=InMemoryFileStore())
    description = executor.description

    assert "read_file(" in description
    assert "`open('notes.txt')` is refused" in description
    assert "Only these modules can be imported" in description
    assert "json" in description
    assert "requests" in description
    # Modules the node refuses must not be advertised to the agent.
    assert "hashlib" not in description
    assert "pd.read_csv" not in description


def test_description_names_only_enabled_libraries():
    """When a trusted instance enables pandas, the description names it and may show pd.read_csv; a
    default instance never does."""
    enabled = PythonCodeExecutor(file_store=InMemoryFileStore(), extra_allowed_modules=["pandas"])

    assert "pandas" in enabled.description
    assert "pd.read_csv" in enabled.description


def test_description_refreshes_when_extra_modules_change():
    """The description is derived, not stored: to_dict() dumps it, so a saved workflow whose
    extra_allowed_modules is later edited (a UI/YAML change) must recompute the text rather than keep the
    stale version that named the wrong importable modules."""
    stale = PythonCodeExecutor(file_store=InMemoryFileStore()).description
    assert "pd.read_csv" not in stale

    reloaded = PythonCodeExecutor(
        file_store=InMemoryFileStore(), extra_allowed_modules=["pandas"], description=stale
    )

    assert reloaded.description != stale
    assert "pd.read_csv" in reloaded.description


def test_custom_description_is_preserved():
    """A caller's own description is kept verbatim; only a generated one is refreshed, so a saved workflow
    with a tailored tool description does not lose it on load."""
    custom = "Analyse sales CSVs only."
    executor = PythonCodeExecutor(file_store=InMemoryFileStore(), extra_allowed_modules=["pandas"], description=custom)

    assert executor.description == custom


def test_description_names_real_write_helpers():
    """The description must name helpers that are actually injected (write_text/write_bytes), not
    write_file_text/write_file_bytes, which do not exist."""
    description = PythonCodeExecutor(file_store=InMemoryFileStore()).description

    assert "write_text()" in description
    assert "write_bytes()" in description
    assert "write_file_text" not in description
    assert "write_file_bytes" not in description


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

    executor = PythonCodeExecutor(file_store=file_store, extra_allowed_modules=["pandas"])
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


def test_concurrent_runs_read_their_own_files():
    code = (
        "import time\nimport pandas as pd\n"
        "def run(params=None):\n    time.sleep(0.2)\n    return str(pd.read_csv('data.csv', header=None)[0][0])\n"
    )
    cwd = os.getcwd()

    def run(tenant):
        file_store = InMemoryFileStore()
        file_store.store("data.csv", tenant.encode())
        return (
            PythonCodeExecutor(file_store=file_store, extra_allowed_modules=["pandas"])
            .run(input_data={"code": code})
            .output["content"]
        )

    tenants = [f"tenant-{i}" for i in range(6)]
    with ThreadPoolExecutor(len(tenants)) as pool:
        assert list(pool.map(run, tenants)) == tenants
    assert os.getcwd() == cwd


def test_a_run_past_its_time_limit_keeps_its_directory_while_no_other_run_waits(tmp_path, monkeypatch):
    monkeypatch.setattr(python_code_executor, "_WORKSPACE_TIME_LIMIT_SECONDS", 0.2)
    monkeypatch.chdir(tmp_path)
    code = (
        "import time\nimport pandas as pd\n"
        "def run(params=None):\n    time.sleep(0.5)\n    pd.DataFrame({'a': [7]}).to_csv('result.csv', index=False)\n"
        "    return str(pd.read_csv('result.csv')['a'][0])\n"
    )

    result = PythonCodeExecutor(file_store=InMemoryFileStore(), extra_allowed_modules=["pandas"]).run(
        input_data={"code": code}
    )

    assert result.output["content"] == "7"
    assert not (tmp_path / "result.csv").exists()


def test_a_run_past_its_time_limit_does_not_block_the_next_one():
    hung = PythonCodeExecutor(file_store=InMemoryFileStore(), error_handling=ErrorHandling(timeout_seconds=0.5))
    file_store = InMemoryFileStore()
    file_store.store("data.csv", b"next")
    following = PythonCodeExecutor(file_store=file_store, extra_allowed_modules=["pandas"])

    with ThreadPoolExecutor(1) as pool:
        pool.submit(hung.run, input_data={"code": "import time\ndef run(params=None):\n    time.sleep(4)\n"})
        deadline = time.monotonic() + 30
        while not python_code_executor._WORKSPACE_LOCK.locked() and time.monotonic() < deadline:
            time.sleep(0.01)
        started = time.monotonic()
        result = following.run(
            input_data={
                "code": "import pandas as pd\ndef run(params=None):\n"
                "    return str(pd.read_csv('data.csv', header=None)[0][0])\n"
            }
        )

    assert result.output["content"] == "next"
    assert time.monotonic() - started < 2


def test_queued_runs_wait_for_runs_that_finish_within_their_time_limits():
    def run(_):
        with python_code_executor._workspace_directory(1):
            time.sleep(0.4)

    # The four runs take 1.6 s in all, more than each one's 1 s limit, but none of them is past its own limit.
    with ThreadPoolExecutor(4) as pool:
        list(pool.map(run, range(4)))
