"""Shared helpers for native-async node tests.

Reused by every PR in the native-async node effort (embedders, HTTP tools,
retrievers, writers). First introduced in the embedders PR.
"""
import asyncio
import gc
import time
from typing import Any


async def assert_concurrent_execution(
    node: Any,
    payloads: list[dict],
    expected_single_call_s: float,
    tolerance_factor: float = 3.0,
) -> list[Any]:
    """Fire N ``run_async`` calls via ``asyncio.gather`` and verify they overlap.

    Fails if total wall time exceeds ``expected_single_call_s * tolerance_factor``,
    which would indicate the async path is accidentally serialized (e.g. running
    on a single-worker thread pool).

    Args:
        node: any object exposing ``async run_async(input_data, **kwargs)``.
        payloads: one payload dict per concurrent call.
        expected_single_call_s: wall time a single call should take (driven by
            the mocked ``asyncio.sleep`` in the underlying async boundary).
        tolerance_factor: multiplier on ``expected_single_call_s`` for the
            allowed total. Default 3x leaves generous slack for scheduler noise.

    Returns:
        List of results from the concurrent calls, in payload order.
    """
    # A full garbage collection of the test process's heap takes over 100 ms, about the whole allowance at the default
    # tolerance, so one landing inside the window failed the check with nothing serialized. Keep it out of the window.
    gc_was_enabled = gc.isenabled()
    gc.disable()
    try:
        start = time.perf_counter()
        results = await asyncio.gather(*(node.run_async(input_data=payload) for payload in payloads))
        elapsed = time.perf_counter() - start
    finally:
        if gc_was_enabled:
            gc.enable()

    max_allowed = expected_single_call_s * tolerance_factor
    assert elapsed < max_allowed, (
        f"Concurrent calls did not overlap: elapsed={elapsed:.3f}s, "
        f"max_allowed={max_allowed:.3f}s, "
        f"expected_single_call={expected_single_call_s:.3f}s, "
        f"call_count={len(payloads)}"
    )
    return results
