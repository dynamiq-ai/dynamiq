import asyncio
import threading
import warnings

import pytest

from dynamiq.callbacks.streaming import AsyncStreamingIteratorCallbackHandler


@pytest.fixture
def no_current_loop():
    asyncio.set_event_loop(None)
    yield
    asyncio.set_event_loop(None)


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


def test_a_handler_built_in_the_main_thread_before_any_loop_gets_one_and_sets_it(no_current_loop):
    with warnings.catch_warnings():
        # Before 3.14, asyncio.get_event_loop() warns while creating the loop.
        warnings.simplefilter("ignore", DeprecationWarning)
        handler = AsyncStreamingIteratorCallbackHandler()
    try:
        assert not handler.loop.is_closed()
        assert asyncio.get_event_loop() is handler.loop
    finally:
        handler.loop.close()


def test_a_handler_built_in_another_thread_without_a_loop_still_fails(no_current_loop):
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
        received = [await handler.queue.get() for _ in events]
        worker.join()
        return received

    assert asyncio.run(consume()) == ["first", "second"]
