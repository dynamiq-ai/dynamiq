"""Benchmark LLM streaming overhead: the OpenAI SDK and LiteLLM alone, and dynamiq's OpenAI node on top of each.

A local OpenAI-compatible server streams a fixed reply, so only client-side work is measured. It runs in separate
processes, keeping its CPU out of the client's numbers.

    uv run python scripts/benchmark_llm_streaming.py                  # both modes
    uv run python scripts/benchmark_llm_streaming.py --mode cpu       # CPU time per stream, one stream at a time
    uv run python scripts/benchmark_llm_streaming.py --mode load --concurrency 25 100

`cpu` reports the client's CPU time per stream, best of `--repeat` rounds. `load` runs `--concurrency` streams at once,
each paced at `--delay-ms` between chunks the way a model generates them, and reports the wall time, the chunks per
second delivered across all streams, and the longest stall of the event loop: how long any other request on the same
process would have waited.

The dynamiq variants build a node per run, as a server building its workflow per request does, and stream to an async
streaming callback with tracing enabled. The SDK variant runs only on a dynamiq version that has `use_native_sdk`.
dynamiq logs every node run at INFO on stderr, a cost it keeps paying here; redirect stderr for a clean table.
"""

import argparse
import asyncio
import json
import multiprocessing
import os
import socket
import time
import warnings
from collections.abc import Awaitable, Callable

MESSAGES = [{"role": "user", "content": "hi"}]


def serve(port: int, n_chunks: int, delay: float) -> None:
    """Serve /v1/chat/completions, streaming `n_chunks` content chunks `delay` seconds apart, then usage."""
    from aiohttp import web

    def chunk(delta: dict | None, finish: str | None = None, usage: dict | None = None) -> bytes:
        body = {
            "id": "chatcmpl-bench",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "gpt-4o-mini",
            "choices": [] if delta is None else [{"index": 0, "delta": delta, "finish_reason": finish}],
        }
        if usage:
            body["usage"] = usage
        return b"data: " + json.dumps(body).encode() + b"\n\n"

    # Distinct tokens: LiteLLM aborts a stream that repeats one chunk too often.
    chunks = [chunk({"role": "assistant", "content": ""})] + [chunk({"content": f"t{i} "}) for i in range(1, n_chunks)]
    finish = chunk({}, "stop")
    usage = chunk(None, usage={"prompt_tokens": 8, "completion_tokens": n_chunks, "total_tokens": 8 + n_chunks})

    async def completions(request: web.Request) -> web.StreamResponse:
        body = await request.json()
        response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await response.prepare(request)
        for c in chunks:
            await response.write(c)
            if delay:
                await asyncio.sleep(delay)
        await response.write(finish)
        if (body.get("stream_options") or {}).get("include_usage"):
            await response.write(usage)
        await response.write(b"data: [DONE]\n\n")
        await response.write_eof()
        return response

    app = web.Application()
    app.router.add_post("/v1/chat/completions", completions)
    web.run_app(app, host="127.0.0.1", port=port, reuse_port=True, access_log=None, print=None)


def start_servers(n_servers: int, n_chunks: int, delay: float) -> tuple[str, list[multiprocessing.Process]]:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    servers = [
        multiprocessing.Process(target=serve, args=(port, n_chunks, delay), daemon=True) for _ in range(n_servers)
    ]
    for server in servers:
        server.start()
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        try:
            socket.create_connection(("127.0.0.1", port), timeout=1).close()
            break
        except OSError:
            time.sleep(0.1)
    return f"http://127.0.0.1:{port}/v1", servers


def build_variants(base_url: str) -> dict[str, Callable[[], Awaitable[None]]]:
    import litellm
    import openai

    from dynamiq.callbacks import TracingCallbackHandler
    from dynamiq.callbacks.streaming import AsyncStreamingIteratorCallbackHandler
    from dynamiq.connections import OpenAI as OpenAIConnection
    from dynamiq.nodes.llms import OpenAI
    from dynamiq.prompts import Prompt
    from dynamiq.runnables import RunnableConfig, RunnableStatus
    from dynamiq.types.streaming import StreamingConfig

    litellm.suppress_debug_info = True
    sdk_client = openai.AsyncOpenAI(base_url=base_url, api_key="bench")
    connection = OpenAIConnection(api_key="bench", url=base_url)

    async def openai_sdk() -> None:
        stream = await sdk_client.chat.completions.create(model="gpt-4o-mini", messages=MESSAGES, stream=True)
        async for _ in stream:
            pass

    async def litellm_alone() -> None:
        stream = await litellm.acompletion(
            model="openai/gpt-4o-mini", messages=MESSAGES, api_base=base_url, api_key="bench", stream=True
        )
        async for _ in stream:
            pass

    def dynamiq_node(**node_kwargs) -> Callable[[], Awaitable[None]]:
        async def run() -> None:
            llm = OpenAI(
                model="gpt-4o-mini",
                connection=connection,
                prompt=Prompt(messages=MESSAGES),
                streaming=StreamingConfig(enabled=True),
                **node_kwargs,
            )
            handler = AsyncStreamingIteratorCallbackHandler()

            async def drain() -> None:
                async for _ in handler:
                    pass

            consumer = asyncio.create_task(drain())
            config = RunnableConfig(callbacks=[handler, TracingCallbackHandler()])
            result = await llm.run_async(input_data={}, config=config)
            consumer.cancel()
            if result.status != RunnableStatus.SUCCESS:
                raise RuntimeError(f"LLM run failed: {result.error}")

        return run

    has_sdk_transport = "use_native_sdk" in OpenAI.model_fields
    variants = {
        "OpenAI SDK alone": openai_sdk,
        "LiteLLM alone": litellm_alone,
        "dynamiq node, LiteLLM": dynamiq_node(**({"use_native_sdk": False} if has_sdk_transport else {})),
    }
    if has_sdk_transport:
        variants["dynamiq node, OpenAI SDK"] = dynamiq_node(use_native_sdk=True)
    return variants


async def cpu_per_stream(run: Callable[[], Awaitable[None]], streams: int, repeat: int) -> float:
    for _ in range(3):
        await run()
    best = float("inf")
    for _ in range(repeat):
        start = time.process_time()
        for _ in range(streams):
            await run()
        best = min(best, (time.process_time() - start) / streams)
    return best * 1000


async def under_load(run: Callable[[], Awaitable[None]], concurrency: int) -> tuple[float, float]:
    await asyncio.gather(*(run() for _ in range(min(concurrency, 10))))
    worst_lag = 0.0
    done = False

    async def heartbeat() -> None:
        nonlocal worst_lag
        while not done:
            start = time.perf_counter()
            await asyncio.sleep(0.01)
            worst_lag = max(worst_lag, time.perf_counter() - start - 0.01)

    beat = asyncio.create_task(heartbeat())
    start = time.perf_counter()
    await asyncio.gather(*(run() for _ in range(concurrency)))
    wall = time.perf_counter() - start
    done = True
    await beat
    return wall, worst_lag * 1000


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=["cpu", "load", "both"], default="both")
    parser.add_argument("--chunks", type=int, default=300, help="content chunks per streamed reply")
    parser.add_argument("--delay-ms", type=float, default=10, help="pause between chunks in load mode")
    parser.add_argument("--concurrency", type=int, nargs="+", default=[25, 100], help="concurrent streams (load)")
    parser.add_argument("--streams", type=int, default=20, help="streams per round (cpu)")
    parser.add_argument("--repeat", type=int, default=3, help="rounds per variant, best kept (cpu)")
    parser.add_argument("--servers", type=int, default=3, help="server processes")
    args = parser.parse_args()

    os.environ.setdefault("LITELLM_LOG", "ERROR")
    os.environ["NO_PROXY"] = os.environ["no_proxy"] = "127.0.0.1,localhost"
    warnings.filterwarnings("ignore")

    async def cpu_mode(base_url: str) -> None:
        print(f"\nCPU time per {args.chunks}-chunk stream (best of {args.repeat} x {args.streams} streams)")
        print(f"{'variant':<28} {'CPU ms / stream':>16}")
        for name, run in build_variants(base_url).items():
            print(f"{name:<28} {await cpu_per_stream(run, args.streams, args.repeat):>16.1f}", flush=True)

    async def load_mode(base_url: str) -> None:
        ideal = args.chunks * args.delay_ms / 1000
        print(f"\nConcurrent {args.chunks}-chunk streams, {args.delay_ms:g} ms between chunks (ideal {ideal:.1f} s)")
        print(f"{'variant':<28} {'streams':>8} {'wall s':>8} {'chunks/s':>9} {'max loop stall ms':>18}")
        variants = build_variants(base_url)
        for concurrency in args.concurrency:
            for name, run in variants.items():
                wall, lag = await under_load(run, concurrency)
                rate = concurrency * args.chunks / wall
                print(f"{name:<28} {concurrency:>8} {wall:>8.2f} {rate:>9.0f} {lag:>18.0f}", flush=True)

    modes = {"cpu": (cpu_mode, 0.0), "load": (load_mode, args.delay_ms / 1000)}
    for mode in ("cpu", "load"):
        if args.mode not in (mode, "both"):
            continue
        measure, delay = modes[mode]
        base_url, servers = start_servers(args.servers, args.chunks, delay)
        try:
            asyncio.run(measure(base_url))
        finally:
            for server in servers:
                server.terminate()


if __name__ == "__main__":
    main()
