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


def test_a_handler_built_in_the_main_thread_before_any_loop_gets_one_and_sets_it():
    done = run_in_fresh_interpreter(
        """
        import asyncio
        from dynamiq.callbacks.streaming import AsyncStreamingIteratorCallbackHandler

        handler = AsyncStreamingIteratorCallbackHandler()
        assert not handler.loop.is_closed()
        assert asyncio.get_event_loop() is handler.loop
        """
    )

    assert done.returncode == 0, done.stderr


def test_a_handler_built_in_another_thread_without_a_loop_still_fails():
    errors = []

    def build():
        try:
            AsyncStreamingIteratorCallbackHandler()
        except RuntimeError as error:
            errors.append(error)

    worker = threading.Thread(target=build)
    worker.start()
    worker.join()

    assert len(errors) == 1


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
