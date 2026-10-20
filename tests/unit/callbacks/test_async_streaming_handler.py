import asyncio
import subprocess  # nosec B404
import sys
import textwrap
import threading
from pathlib import Path

import pytest

from dynamiq.callbacks.streaming import AsyncStreamingIteratorCallbackHandler

REPO_ROOT = Path(__file__).parents[3]


def run_in_fresh_interpreter(code: str, timeout: float = 60) -> subprocess.CompletedProcess:
    """Run `code` in a new interpreter, whose main thread has no event loop set, without touching this process's."""
    return subprocess.run(  # nosec B603
        [sys.executable, "-W", "ignore::DeprecationWarning", "-c", textwrap.dedent(code)],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        timeout=timeout,
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


def test_a_handler_built_between_two_asyncio_runs_delivers_to_the_second():
    # asyncio.run() clears the current loop when it returns, so the handler is built with no loop to take.
    asyncio.run(asyncio.sleep(0))
    handler = AsyncStreamingIteratorCallbackHandler()

    async def consume():
        handler.send_to_queue("event")
        return await asyncio.wait_for(handler.queue.get(), timeout=5)

    assert asyncio.run(consume()) == "event"


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


def test_held_events_wait_for_room_in_a_bounded_queue_and_keep_their_order():
    asyncio.run(asyncio.sleep(0))  # Leaves no current loop, so the handler holds what it is sent.
    handler = AsyncStreamingIteratorCallbackHandler(queue=asyncio.Queue(maxsize=1))
    for event in ("first", "second", "third"):
        handler.send_to_queue(event)

    async def consume():
        return [await asyncio.wait_for(handler.__anext__(), timeout=5) for _ in range(3)]

    assert asyncio.run(consume()) == ["first", "second", "third"]


def test_events_sent_while_held_ones_are_replayed_come_after_them():
    asyncio.run(asyncio.sleep(0))
    handler = AsyncStreamingIteratorCallbackHandler(queue=asyncio.Queue(maxsize=1))
    handler.send_to_queue("held first")
    handler.send_to_queue("held second")

    async def consume():
        received = [await asyncio.wait_for(handler.__anext__(), timeout=5)]
        worker = threading.Thread(target=handler.send_to_queue, args=("sent later",))
        worker.start()
        worker.join()
        received += [await asyncio.wait_for(handler.__anext__(), timeout=5) for _ in range(2)]
        return received

    assert asyncio.run(consume()) == ["held first", "held second", "sent later"]


def test_a_consumer_starting_after_the_run_ended_gets_every_held_event():
    asyncio.run(asyncio.sleep(0))
    handler = AsyncStreamingIteratorCallbackHandler(queue=asyncio.Queue(maxsize=1))
    handler.send_to_queue("first")
    handler.send_to_queue("second")
    handler.done_event.set()

    async def consume():
        return [event async for event in handler]

    assert asyncio.run(asyncio.wait_for(consume(), timeout=5)) == ["first", "second"]


def test_a_bounded_queue_given_with_a_set_loop_receives_every_event():
    done = run_in_fresh_interpreter(
        """
        import asyncio
        from dynamiq.callbacks.streaming import AsyncStreamingIteratorCallbackHandler

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        handler = AsyncStreamingIteratorCallbackHandler(queue=asyncio.Queue(maxsize=1))
        handler.send_to_queue("first")
        handler.send_to_queue("second")

        async def consume():
            return [await asyncio.wait_for(handler.__anext__(), timeout=5) for _ in range(2)]

        assert loop.run_until_complete(consume()) == ["first", "second"]
        """
    )

    assert done.returncode == 0, done.stderr


def test_a_queue_given_with_a_set_loop_reaches_a_direct_reader():
    done = run_in_fresh_interpreter(
        """
        import asyncio
        from dynamiq.callbacks.streaming import AsyncStreamingIteratorCallbackHandler

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        queue = asyncio.Queue()
        handler = AsyncStreamingIteratorCallbackHandler(queue=queue)
        assert handler.loop is loop
        handler.send_to_queue("event")

        assert loop.run_until_complete(asyncio.wait_for(queue.get(), timeout=5)) == "event"
        """
    )

    assert done.returncode == 0, done.stderr


def test_a_closed_set_loop_is_not_used():
    done = run_in_fresh_interpreter(
        """
        import asyncio
        from dynamiq.callbacks.streaming import AsyncStreamingIteratorCallbackHandler

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.close()
        handler = AsyncStreamingIteratorCallbackHandler()
        assert handler.loop is None
        handler.send_to_queue("event")

        async def consume():
            return await asyncio.wait_for(handler.__anext__(), timeout=5)

        assert asyncio.run(consume()) == "event"
        """
    )

    assert done.returncode == 0, done.stderr


@pytest.mark.skipif(sys.version_info < (3, 12), reason="asyncio.eager_task_factory arrived in Python 3.12")
def test_held_events_replay_in_order_under_the_eager_task_factory():
    # A deadlock blocks the loop itself, so only a timeout outside the interpreter can stop it.
    done = run_in_fresh_interpreter(
        """
        import asyncio
        from dynamiq.callbacks.streaming import AsyncStreamingIteratorCallbackHandler

        asyncio.set_event_loop(None)
        handler = AsyncStreamingIteratorCallbackHandler(queue=asyncio.Queue(maxsize=1))
        for event in ("first", "second", "third"):
            handler.send_to_queue(event)

        async def consume():
            asyncio.get_running_loop().set_task_factory(asyncio.eager_task_factory)
            received = [await handler.__anext__()]
            handler.send_to_queue("sent during the replay")
            received += [await handler.__anext__() for _ in range(3)]
            return received

        assert asyncio.run(consume()) == ["first", "second", "third", "sent during the replay"]
        """,
        timeout=20,
    )

    assert done.returncode == 0, done.stderr
