"""Tests for the parallelized MultiFileTypeConverter.

These exercise three guarantees of the concurrency change:
    1. Correctness and output ordering with multiple files of different types.
    2. That per-file conversions actually run concurrently (proved with a barrier and
       a sleep-based timing assertion).
    3. That error semantics match the prior sequential behavior (first failure raises).
"""

import threading
import time
from io import BytesIO

import pytest

from dynamiq.nodes.converters.multi_file_type_converter import MultiFileTypeConverter
from dynamiq.nodes.converters.text import TextFileConverter
from dynamiq.runnables import RunnableConfig, RunnableStatus


def _txt(content: str, name: str) -> BytesIO:
    buf = BytesIO(content.encode("utf-8"))
    buf.name = name
    return buf


@pytest.fixture
def converter():
    node = MultiFileTypeConverter()
    node.init_components()
    return node


def _contents(documents):
    return [doc.content for doc in documents]


# ---------------------------------------------------------------------------
# (a) Correctness + output ordering with multiple files of different types
# ---------------------------------------------------------------------------


def test_preserves_output_order_with_files(converter):
    """Documents map back to the input files in their original order."""
    files = [_txt(f"content number {i}", f"file_{i}.txt") for i in range(8)]

    result = converter.run(input_data={"files": files})

    assert result.status == RunnableStatus.SUCCESS
    documents = result.output["documents"]
    assert len(documents) == 8
    # Each TextFileConverter yields exactly one document; order must match the inputs.
    for i, content in enumerate(_contents(documents)):
        assert f"content number {i}" in content


def test_mixed_file_types_convert_correctly(converter):
    """Different file types route to the right converters and keep their order."""
    files = [
        _txt("plain text body", "a.txt"),
        _txt("# Markdown heading\nbody", "b.md"),
        _txt("another text body", "c.txt"),
    ]

    result = converter.run(input_data={"files": files})

    assert result.status == RunnableStatus.SUCCESS
    contents = _contents(result.output["documents"])
    assert len(contents) == 3
    assert "plain text body" in contents[0]
    assert "Markdown heading" in contents[1]
    assert "another text body" in contents[2]


def test_file_paths_preserve_per_file_metadata(tmp_path, converter):
    """Explicit file paths keep their per-file metadata aligned after parallel processing."""
    paths = []
    for i in range(4):
        p = tmp_path / f"f{i}.txt"
        p.write_text(f"body {i}")
        paths.append(str(p))

    metadata = [{"idx": i} for i in range(4)]
    result = converter.run(input_data={"file_paths": paths, "metadata": metadata})

    assert result.status == RunnableStatus.SUCCESS
    documents = result.output["documents"]
    assert len(documents) == 4
    # Each document's content and its idx metadata must stay consistent with each other.
    for doc in documents:
        idx = doc.metadata["idx"]
        assert f"body {idx}" in doc.content


# ---------------------------------------------------------------------------
# (b) Conversions actually run concurrently
# ---------------------------------------------------------------------------


def test_conversions_run_concurrently_via_barrier(converter):
    """All N converters must be inside run() simultaneously, proving real parallelism.

    A barrier of size N only releases once N threads reach it. If conversion were
    sequential, the second thread would never arrive and the barrier would time out.
    """
    n = 6
    files = [_txt(f"body {i}", f"f{i}.txt") for i in range(n)]

    barrier = threading.Barrier(n, timeout=5)
    original_run = TextFileConverter.run

    def barriered_run(self, *args, **kwargs):
        # Block until all N file conversions have entered concurrently.
        barrier.wait()
        return original_run(self, *args, **kwargs)

    TextFileConverter.run = barriered_run
    try:
        result = converter.run(input_data={"files": files})
    finally:
        TextFileConverter.run = original_run

    assert result.status == RunnableStatus.SUCCESS
    assert len(result.output["documents"]) == n


def test_parallel_is_faster_than_sequential(converter):
    """Wall-clock for N sleep-mocked files is ~one sleep, not N sleeps."""
    n = 8
    sleep_s = 0.05
    files = [_txt(f"body {i}", f"f{i}.txt") for i in range(n)]

    original_run = TextFileConverter.run

    def slow_run(self, *args, **kwargs):
        time.sleep(sleep_s)
        return original_run(self, *args, **kwargs)

    TextFileConverter.run = slow_run
    try:
        start = time.perf_counter()
        result = converter.run(input_data={"files": files})
        elapsed = time.perf_counter() - start
    finally:
        TextFileConverter.run = original_run

    assert result.status == RunnableStatus.SUCCESS
    sequential_estimate = n * sleep_s
    # Parallel execution should be well under half the sequential time.
    assert (
        elapsed < sequential_estimate / 2
    ), f"Expected parallel run << {sequential_estimate:.3f}s sequential, got {elapsed:.3f}s"


def test_max_workers_caps_concurrency(converter):
    """max_workers limits how many conversions run at once."""
    n = 6
    converter.max_workers = 2
    files = [_txt(f"body {i}", f"f{i}.txt") for i in range(n)]

    lock = threading.Lock()
    state = {"active": 0, "peak": 0}
    original_run = TextFileConverter.run

    def tracking_run(self, *args, **kwargs):
        with lock:
            state["active"] += 1
            state["peak"] = max(state["peak"], state["active"])
        try:
            time.sleep(0.02)
            return original_run(self, *args, **kwargs)
        finally:
            with lock:
                state["active"] -= 1

    TextFileConverter.run = tracking_run
    try:
        result = converter.run(input_data={"files": files})
    finally:
        TextFileConverter.run = original_run

    assert result.status == RunnableStatus.SUCCESS
    assert state["peak"] <= 2


def test_config_max_node_workers_caps_concurrency(converter):
    """RunnableConfig.max_node_workers is honored as an upper bound on concurrency."""
    n = 6
    files = [_txt(f"body {i}", f"f{i}.txt") for i in range(n)]

    lock = threading.Lock()
    state = {"active": 0, "peak": 0}
    original_run = TextFileConverter.run

    def tracking_run(self, *args, **kwargs):
        with lock:
            state["active"] += 1
            state["peak"] = max(state["peak"], state["active"])
        try:
            time.sleep(0.02)
            return original_run(self, *args, **kwargs)
        finally:
            with lock:
                state["active"] -= 1

    TextFileConverter.run = tracking_run
    try:
        result = converter.run(input_data={"files": files}, config=RunnableConfig(max_node_workers=3))
    finally:
        TextFileConverter.run = original_run

    assert result.status == RunnableStatus.SUCCESS
    assert state["peak"] <= 3


def test_max_workers_zero_is_honored_not_widened(converter):
    """An explicit max_workers=0 floors to a single worker, not the full file count."""
    n = 5
    converter.max_workers = 0
    files = [_txt(f"body {i}", f"f{i}.txt") for i in range(n)]

    lock = threading.Lock()
    state = {"active": 0, "peak": 0}
    original_run = TextFileConverter.run

    def tracking_run(self, *args, **kwargs):
        with lock:
            state["active"] += 1
            state["peak"] = max(state["peak"], state["active"])
        try:
            time.sleep(0.02)
            return original_run(self, *args, **kwargs)
        finally:
            with lock:
                state["active"] -= 1

    TextFileConverter.run = tracking_run
    try:
        result = converter.run(input_data={"files": files})
    finally:
        TextFileConverter.run = original_run

    assert result.status == RunnableStatus.SUCCESS
    assert state["peak"] == 1


def test_reused_bytesio_instance_is_isolated_per_file(converter):
    """The same BytesIO passed multiple times is copied per work item, not shared."""
    buf = _txt("shared body", "dup.txt")
    files = [buf, buf, buf]

    result = converter.run(input_data={"files": files})

    assert result.status == RunnableStatus.SUCCESS
    documents = result.output["documents"]
    assert len(documents) == 3
    assert all("shared body" in doc.content for doc in documents)
    assert buf.name == "dup.txt"
    assert buf.getvalue() == b"shared body"


# ---------------------------------------------------------------------------
# (c) Error propagation matches prior behavior
# ---------------------------------------------------------------------------


def test_single_file_failure_fails_the_node(converter):
    """A failure converting one file fails the whole node (no silent swallowing)."""
    files = [_txt(f"body {i}", f"f{i}.txt") for i in range(4)]

    original_run = TextFileConverter.run

    def failing_run(self, *args, **kwargs):
        input_data = kwargs.get("input_data") or (args[0] if args else None)
        files_arg = input_data.get("files") if isinstance(input_data, dict) else None
        if files_arg and getattr(files_arg[0], "name", "") == "f2.txt":
            raise RuntimeError("boom on f2")
        return original_run(self, *args, **kwargs)

    TextFileConverter.run = failing_run
    try:
        result = converter.run(input_data={"files": files})
    finally:
        TextFileConverter.run = original_run

    # Same as sequential: the node surfaces a failure rather than returning partial output.
    assert result.status == RunnableStatus.FAILURE


def test_no_documents_raises(converter):
    """An empty resolved file set fails just as before."""
    result = converter.run(input_data={"file_paths": ["/path/that/does/not/exist.txt"]})
    assert result.status == RunnableStatus.FAILURE
