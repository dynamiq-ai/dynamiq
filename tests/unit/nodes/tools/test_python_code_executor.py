from dynamiq.nodes.tools.python_code_executor import PythonCodeExecutor, PythonCodeExecutorFileWorkspace
from dynamiq.runnables import RunnableStatus
from dynamiq.storages.file import InMemorySandbox


def test_code_executor_captures_stdout_without_print_errors():
    """Ensure the RestrictedPython print hooks no longer crash and stdout is captured."""
    file_store = InMemorySandbox()
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
    file_store = InMemorySandbox()
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
    file_store = InMemorySandbox()
    file_store.store("notes/data.txt", "plain text payload")
    file_store.store("reports/data.xlsx", b"\x00\x01binary-bytes")

    workspace = PythonCodeExecutorFileWorkspace(file_store=file_store)

    assert workspace.read("notes/data.txt") == "plain text payload"
    assert workspace.read("reports/data.xlsx") == b"\x00\x01binary-bytes"


def test_code_executor_reads_files_via_helper_function():
    """Code snippets should rely on the injected read_file helper to access uploaded artifacts."""
    file_store = InMemorySandbox()
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
