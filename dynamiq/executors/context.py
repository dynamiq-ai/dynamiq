import contextvars
from concurrent.futures import ThreadPoolExecutor


class ContextAwareThreadPoolExecutor(ThreadPoolExecutor):
    """ThreadPoolExecutor that automatically propagates contextvars to worker threads.

    Standard ThreadPoolExecutor does not copy the calling thread's contextvars
    into worker threads, causing context (e.g. request_id, tracing metadata)
    to be lost. This subclass captures a snapshot via copy_context() before
    each submit and runs the callable inside that snapshot.

    Since ThreadPoolExecutor.map() delegates to submit(), both submit() and
    map() are covered automatically.
    """

    def submit(self, fn, /, *args, **kwargs):
        ctx = contextvars.copy_context()
        return super().submit(ctx.run, fn, *args, **kwargs)
