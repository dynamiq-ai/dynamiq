import asyncio
import subprocess  # nosec B404
import sys
import textwrap
import threading
from pathlib import Path

from dynamiq.callbacks.streaming import AsyncStreamingIteratorCallbackHandler

REPO_ROOT = Path(__file__).parents[3]


def run_in_fresh_interpreter(code: str) -> subprocess.CompletedProcess:
    """Run `code` in a new interpreter, whose main thread has no event loop set, without touching this process's."""
    return subprocess.run(  # nosec B603
        [sys.executable, "-W", "ignore::DeprecationWarning", "-c", textwrap.dedent(code)],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        timeout=60,
    )


def test_a_handler_built_inside_a_running_loop_uses_that_loop():
    async def build():
        return AsyncStreamingIteratorCallbackHandler(), asyncio.get_running_loop()

    handler, running = asyncio.run(build())

    assert handler.loop is running


def test_a_handler_given_a_loop_uses_it():
    loop = asyncio.new_event_loop()
    try:
        assert AsyncStreamingIteratorCallbackHandler(loop=loop).loop is loop
    finally:
        loop.close()


def test_a_handler_built_before_asyncio_run_delivers_to_the_loop_that_consumes_it():
    done = run_in_fresh_interpreter(
        """
        import asyncio, threading
        from dynamiq.callbacks.streaming import AsyncStreamingIteratorCallbackHandler

        handler = AsyncStreamingIteratorCallbackHandler()
        assert handler.loop is None
        handler.send_to_queue("sent before the loop started")

        async def consume():
            first = await asyncio.wait_for(handler.__anext__(), timeout=5)
            assert handler.loop is asyncio.get_running_loop()
            threading.Thread(target=handler.send_to_queue, args=("sent from a worker",)).start()
            second = await asyncio.wait_for(handler.__anext__(), timeout=5)
            return [first, second]

        assert asyncio.run(consume()) == ["sent before the loop started", "sent from a worker"]
        """
    )

    assert done.returncode == 0, done.stderr


def test_a_handler_built_before_its_loop_delivers_to_a_consumer_reading_the_queue():
    handler = AsyncStreamingIteratorCallbackHandler()

    async def consume():
        threading.Thread(target=handler.send_to_queue, args=("event",)).start()
        return await asyncio.wait_for(handler.queue.get(), timeout=5)

    assert asyncio.run(consume()) == "event"


def test_a_handler_built_in_a_worker_thread_binds_to_the_loop_that_consumes_it():
    built = []
    worker = threading.Thread(target=lambda: built.append(AsyncStreamingIteratorCallbackHandler()))
    worker.start()
    worker.join()
    handler = built[0]
    handler.send_to_queue("event")

    async def consume():
        return await asyncio.wait_for(handler.__anext__(), timeout=5), handler.loop is asyncio.get_running_loop()

    assert asyncio.run(consume()) == ("event", True)


def test_events_sent_from_a_worker_thread_reach_the_async_consumer():
    async def consume():
        handler = AsyncStreamingIteratorCallbackHandler()
        events = ["first", "second"]

        def produce():
            for event in events:
                handler.send_to_queue(event)

        worker = threading.Thread(target=produce)
        worker.start()
        # Bounded, so an event that never arrives fails the test instead of hanging the run.
        received = [await asyncio.wait_for(handler.queue.get(), timeout=5) for _ in events]
        worker.join()
        return received

    assert asyncio.run(consume()) == ["first", "second"]
