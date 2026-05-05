import asyncio

import pytest_asyncio


@pytest_asyncio.fixture(autouse=True)
async def cancel_pending_tasks_after_test():
    yield
    current = asyncio.current_task()
    pending = [t for t in asyncio.all_tasks() if t is not current and not t.done()]
    for task in pending:
        task.cancel()
    if pending:
        await asyncio.gather(*pending, return_exceptions=True)
