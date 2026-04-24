import asyncio

import pytest

from tests.helpers.async_node import assert_concurrent_execution


class FakeAsyncNode:
    """Minimal stand-in: `run_async` awaits `sleep_s` per call."""

    def __init__(self, sleep_s: float):
        self._sleep_s = sleep_s

    async def run_async(self, input_data, **kwargs):
        await asyncio.sleep(self._sleep_s)
        return {"ok": True, "input": input_data}


@pytest.mark.asyncio
async def test_assert_concurrent_execution_passes_when_calls_overlap():
    node = FakeAsyncNode(sleep_s=0.1)
    payloads = [{"i": i} for i in range(10)]

    results = await assert_concurrent_execution(
        node,
        payloads,
        expected_single_call_s=0.1,
        tolerance_factor=3.0,
    )

    assert len(results) == 10
    assert all(r["ok"] for r in results)


@pytest.mark.asyncio
async def test_assert_concurrent_execution_fails_when_calls_are_serialized():
    class SerialNode:
        def __init__(self):
            self._lock = asyncio.Lock()

        async def run_async(self, input_data, **kwargs):
            async with self._lock:
                await asyncio.sleep(0.1)
            return {"ok": True}

    node = SerialNode()
    payloads = [{"i": i} for i in range(10)]

    with pytest.raises(AssertionError, match="did not overlap"):
        await assert_concurrent_execution(
            node,
            payloads,
            expected_single_call_s=0.1,
            tolerance_factor=3.0,
        )
