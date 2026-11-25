"""HTTP clients that send a node's request under an `EgressPolicy`.

Each request gets its own session or client, so no cookie or connection outlives it or reaches another run. The address
check runs where the clients open a TCP connection: the host is resolved once, every address it resolves to must be
public, and the connection goes to one of those addresses, so a name that resolves differently the next time cannot
redirect it. TLS still verifies the certificate against the host name.
"""

import asyncio
import ipaddress
import socket
import ssl
import sys
import threading
import time
import zlib
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from contextvars import ContextVar
from functools import cache, partial
from typing import Any

import httpcore
import httpx
import requests
from requests.adapters import BaseAdapter, HTTPAdapter
from urllib3.connection import HTTPConnection, HTTPSConnection
from urllib3.connectionpool import HTTPConnectionPool, HTTPSConnectionPool
from urllib3.exceptions import ConnectTimeoutError, NameResolutionError, NewConnectionError
from urllib3.util import connection as urllib3_connection

from dynamiq.types.egress import EgressPolicy

# The private ranges and the rest of the IANA special-purpose IPv4 registry. ipaddress's is_global is not used: it
# counts multicast and other blocks that no public API is served from as global.
DENIED_NETWORKS = tuple(
    ipaddress.IPv4Network(network)
    for network in (
        "0.0.0.0/8",
        "10.0.0.0/8",
        "100.64.0.0/10",
        "127.0.0.0/8",
        "169.254.0.0/16",
        "172.16.0.0/12",
        "192.0.0.0/24",
        "192.0.2.0/24",
        "192.31.196.0/24",
        "192.52.193.0/24",
        "192.88.99.0/24",
        "192.168.0.0/16",
        "192.175.48.0/24",
        "198.18.0.0/15",
        "198.51.100.0/24",
        "203.0.113.0/24",
        "224.0.0.0/4",
        "240.0.0.0/4",
    )
)

# Both clients read bodies raw and decode them here, so that no decoder can produce more than the policy allows in one
# step; only zlib's formats can be decoded with such a bound, so they are the only ones asked for and accepted.
ACCEPT_ENCODING = "gzip, deflate"
SUPPORTED_ENCODINGS = frozenset({"gzip", "x-gzip", "deflate"})

READ_CHUNK_BYTES = 64 * 1024

# Each connect and read waits at most until the deadline, so its own timeout and the deadline fire together; a timeout
# this close to the deadline is reported as the deadline.
DEADLINE_SLACK_SECONDS = 0.1

# getaddrinfo cannot be cancelled, so a lookup that hangs keeps its thread until the system resolver gives up. Lookups
# run in a pool of their own, so that such lookups cannot hold up the other clients in the process, and a request waits
# for one only until its deadline.
RESOLVER_THREADS = 16
_resolver = ThreadPoolExecutor(max_workers=RESOLVER_THREADS, thread_name_prefix="dynamiq-egress-resolver")


class EgressError(Exception):
    """A request the egress policy stopped. Sending it again gives the same result."""


class EgressRefusedError(EgressError):
    """A request that would connect to an address the policy does not allow."""


class EgressLimitError(EgressError):
    """A request that redirects too often, or a response that is too large, takes too long or uses an encoding the
    policy cannot bound."""


def is_public_address(address: str) -> bool:
    """Reports whether address is a public IPv4 address."""
    try:
        ip = ipaddress.ip_address(address)
    except ValueError:
        return False
    return ip.version == 4 and not any(ip in network for network in DENIED_NETWORKS)


def _timed_out(timeout_seconds: float) -> EgressLimitError:
    return EgressLimitError(f"The request took longer than {timeout_seconds:g} seconds.")


def _check_host(host: str) -> str:
    """The host to resolve, without brackets. Raises EgressRefusedError for an IPv6 address."""
    host = host.strip("[]")
    if ":" in host:
        raise EgressRefusedError(f"{host} is an IPv6 address. Only public IPv4 addresses are allowed.")
    return host


def _lookup(host: str, port: int) -> list[str]:
    infos = socket.getaddrinfo(host, port, socket.AF_INET, socket.SOCK_STREAM)
    addresses = list(dict.fromkeys(info[4][0] for info in infos))
    if not addresses:
        raise socket.gaierror(socket.EAI_NONAME, f"{host} has no IPv4 address")
    # One answer outside the public internet refuses the host, so it cannot pass by sometimes resolving to it.
    if not all(is_public_address(address) for address in addresses):
        raise EgressRefusedError(f"{host} resolves to an address outside the public internet, which is not allowed.")
    return addresses


def resolve_public(host: str, port: int, timeout: float | None = None) -> list[str]:
    """The IPv4 addresses host resolves to, when all of them are public.

    Raises:
        EgressRefusedError: host is an IPv6 address, or one of its addresses is not public.
        TimeoutError: the lookup did not finish within timeout seconds.
        socket.gaierror: host has no IPv4 address.
    """
    host = _check_host(host)
    future = _resolver.submit(_lookup, host, port)
    try:
        return future.result(timeout=timeout)
    except FutureTimeoutError as e:
        future.cancel()
        raise TimeoutError(f"Looking up {host} took too long.") from e


async def resolve_public_async(host: str, port: int) -> list[str]:
    """`resolve_public()` for the running event loop, without a timeout of its own."""
    host = _check_host(host)
    future = _resolver.submit(_lookup, host, port)
    try:
        return await asyncio.wrap_future(future)
    except asyncio.CancelledError:
        future.cancel()
        raise


@cache
def _ssl_context() -> ssl.SSLContext:
    """The TLS context of both clients, which trusts the certifi bundle whatever the environment sets.

    Building one parses the whole CA bundle, so the process shares it. Only HTTP/1.1 clients may share it: httpcore sets
    the ALPN protocols on the context for every connection it opens.
    """
    return httpx.create_ssl_context(trust_env=False)


class _Call:
    """The deadline and byte budget of one request, its redirects included.

    For a synchronous request it also stops the request when the deadline passes. requests applies its timeout to each
    read, so a server that sends a byte now and then would hold the request, and the thread, as long as it liked. A
    duplicate of each socket the request opens is kept for this: TLS takes over the original socket object, but shutting
    down the duplicate ends every read and write on the connection, the handshake included.
    """

    def __init__(self, policy: EgressPolicy):
        self.policy = policy
        self.deadline = time.monotonic() + policy.timeout_seconds
        self.wire_bytes_left = policy.max_response_bytes
        self.body_bytes_left = policy.max_response_bytes
        self.expired = False
        self._sockets: list[socket.socket] = []
        self._lock = threading.Lock()
        self._timer: threading.Timer | None = None

    def remaining(self) -> float:
        return self.deadline - time.monotonic()

    def check(self) -> None:
        """Raises EgressLimitError once the deadline has passed."""
        if self.expired or self.remaining() <= 0:
            raise _timed_out(self.policy.timeout_seconds)

    def timeout(self, timeout: Any) -> Any:
        """The timeout of the next connect or read: timeout, or the time left if that is shorter.

        A (connect, read) pair is capped item by item. Raises EgressLimitError once the deadline has passed.
        """
        self.check()
        if isinstance(timeout, tuple):
            return tuple(self.timeout(item) for item in timeout)
        remaining = self.remaining()
        return min(timeout, remaining) if isinstance(timeout, (int, float)) else remaining

    def start_timer(self) -> None:
        self._timer = threading.Timer(self.policy.timeout_seconds, self._expire)
        self._timer.daemon = True
        self._timer.start()

    def watch(self, sock: socket.socket) -> None:
        """Keeps a duplicate of sock to shut down at the deadline. Closes sock and raises if the deadline has passed."""
        duplicate = sock.dup()
        with self._lock:
            if not self.expired:
                self._sockets.append(duplicate)
                return
        duplicate.close()
        sock.close()
        raise _timed_out(self.policy.timeout_seconds)

    def _expire(self) -> None:
        with self._lock:
            self.expired = True
            sockets = list(self._sockets)
        for sock in sockets:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass

    def close(self) -> None:
        if self._timer is not None:
            self._timer.cancel()
        with self._lock:
            sockets, self._sockets = self._sockets, []
        for sock in sockets:
            sock.close()


_current_call: ContextVar[_Call | None] = ContextVar("dynamiq_egress_call", default=None)


def _require_call() -> _Call:
    call = _current_call.get()
    if call is None:
        raise EgressRefusedError("A guarded connection was opened outside a guarded request.")
    return call


class _TrackedConnectionMixin:
    """Connects as urllib3 does, within the request's deadline."""

    def _new_conn(self) -> socket.socket:
        call = _require_call()
        self.timeout = call.timeout(self.timeout)
        sock = super()._new_conn()
        call.watch(sock)
        return sock


class _PublicConnectionMixin:
    """Connects only to a public address the host resolved to, within the request's deadline."""

    def _new_conn(self) -> socket.socket:
        call = _require_call()
        try:
            addresses = resolve_public(self._dns_host, self.port, timeout=call.timeout(None))
        except TimeoutError as e:
            raise _timed_out(call.policy.timeout_seconds) from e
        except socket.gaierror as e:
            raise NameResolutionError(self.host, self, e) from e
        error: Exception | None = None
        for address in addresses:
            try:
                sock = urllib3_connection.create_connection(
                    (address, self.port),
                    call.timeout(self.timeout),
                    source_address=self.source_address,
                    socket_options=self.socket_options,
                )
            except TimeoutError:
                error = ConnectTimeoutError(self, f"Connection to {self.host} timed out.")
            except OSError as e:
                error = NewConnectionError(self, f"Failed to establish a new connection: {e}")
            else:
                call.watch(sock)
                sys.audit("http.client.connect", self, self.host, self.port)
                return sock
        raise error


class _PublicHTTPConnection(_PublicConnectionMixin, HTTPConnection):
    pass


class _PublicHTTPSConnection(_PublicConnectionMixin, HTTPSConnection):
    pass


class _TrackedHTTPConnection(_TrackedConnectionMixin, HTTPConnection):
    pass


class _TrackedHTTPSConnection(_TrackedConnectionMixin, HTTPSConnection):
    pass


class _PublicHTTPConnectionPool(HTTPConnectionPool):
    ConnectionCls = _PublicHTTPConnection


class _PublicHTTPSConnectionPool(HTTPSConnectionPool):
    ConnectionCls = _PublicHTTPSConnection


class _TrackedHTTPConnectionPool(HTTPConnectionPool):
    ConnectionCls = _TrackedHTTPConnection


class _TrackedHTTPSConnectionPool(HTTPSConnectionPool):
    ConnectionCls = _TrackedHTTPSConnection


class _PoolAdapter(HTTPAdapter):
    """An adapter whose pools open connections with the given classes, and that never uses a proxy."""

    def __init__(self, pool_classes: dict[str, type[HTTPConnectionPool]]):
        self._pool_classes = pool_classes
        super().__init__()

    def init_poolmanager(self, *args, **kwargs) -> None:
        super().init_poolmanager(*args, **kwargs)
        self.poolmanager.pool_classes_by_scheme = self._pool_classes

    def proxy_manager_for(self, proxy, **proxy_kwargs):
        # A proxy would make the connection, so the address check would only ever see the proxy's address.
        raise EgressRefusedError("Requests may not go through a proxy.")


class _PolicyAdapter(BaseAdapter):
    """Sends each request, every redirect included, through the address check unless the policy exempts its URL."""

    def __init__(self, policy: EgressPolicy):
        super().__init__()
        self._policy = policy
        self._public = _PoolAdapter({"http": _PublicHTTPConnectionPool, "https": _PublicHTTPSConnectionPool})
        self._exempt = _PoolAdapter({"http": _TrackedHTTPConnectionPool, "https": _TrackedHTTPSConnectionPool})

    def send(self, request: requests.PreparedRequest, timeout: float | None = None, **kwargs) -> requests.Response:
        adapter = self._exempt if self._policy.is_exempt(request.url) else self._public
        return adapter.send(request, timeout=_require_call().timeout(timeout), **kwargs)

    def close(self) -> None:
        self._public.close()
        self._exempt.close()


def _too_large(max_bytes: int) -> EgressLimitError:
    return EgressLimitError(f"The response is larger than {max_bytes} bytes, the most allowed.")


class _BoundedDecoder:
    """Decodes a gzip, deflate or plain body, counting its bytes on the wire and decoded against the request's budget.

    No step produces more than the budget has left, so a small compressed body cannot expand past it.
    """

    def __init__(self, content_encoding: str | None, call: _Call):
        codings = [
            coding
            for coding in (part.strip().lower() for part in (content_encoding or "").split(","))
            if coding and coding != "identity"
        ]
        if len(codings) > 1 or (codings and codings[0] not in SUPPORTED_ENCODINGS):
            raise EgressLimitError(f"Responses with the content encoding {content_encoding!r} are not supported.")
        self._call = call
        self._coding = codings[0] if codings else None
        if self._coding in ("gzip", "x-gzip"):
            self._decompressor = zlib.decompressobj(zlib.MAX_WBITS | 16)
        elif self._coding == "deflate":
            self._decompressor = zlib.decompressobj()
        else:
            self._decompressor = None
        self._started = False

    def _take(self, decoded: bytes) -> bytes:
        if len(decoded) > self._call.body_bytes_left:
            raise _too_large(self._call.policy.max_response_bytes)
        self._call.body_bytes_left -= len(decoded)
        return decoded

    def decode(self, data: bytes) -> bytes:
        self._call.check()
        if len(data) > self._call.wire_bytes_left:
            raise _too_large(self._call.policy.max_response_bytes)
        self._call.wire_bytes_left -= len(data)
        if self._decompressor is None:
            return self._take(data)
        # Data after the end of the stream is dropped, as httpx drops it, instead of piling up in the decompressor.
        if self._decompressor.eof:
            return b""
        try:
            decoded = self._decompressor.decompress(data, self._call.body_bytes_left + 1)
        except zlib.error as e:
            # Some servers send raw deflate data without the zlib header, which httpx and urllib3 accept too.
            if self._coding != "deflate" or self._started:
                raise EgressLimitError(f"The response body could not be decoded: {e}") from e
            self._decompressor = zlib.decompressobj(-zlib.MAX_WBITS)
            decoded = self._decompressor.decompress(data, self._call.body_bytes_left + 1)
        self._started = True
        return self._take(decoded)

    def flush(self) -> bytes:
        if self._decompressor is None or self._decompressor.eof:
            return b""
        try:
            return self._take(self._decompressor.flush())
        except zlib.error as e:
            raise EgressLimitError(f"The response body could not be decoded: {e}") from e


def _read_body(call: _Call, response: requests.Response, **kwargs) -> requests.Response:
    """Reads the body into the response, every redirect included, which requests would otherwise read whole.

    The body of a redirect counts against the budget and is then dropped.
    """
    try:
        decoder = _BoundedDecoder(response.headers.get("content-encoding"), call)
        chunks = [decoder.decode(raw) for raw in response.raw.stream(READ_CHUNK_BYTES, decode_content=False)]
        chunks.append(decoder.flush())
    except BaseException:
        response.close()
        raise
    response._content = b"" if response.is_redirect else b"".join(chunks)
    response._content_consumed = True
    return response


def send(policy: EgressPolicy, *, timeout: float | None = None, **request_kwargs: Any) -> requests.Response:
    """Sends a request with requests under policy and returns the response with its body read.

    Args:
        policy: The policy to apply.
        timeout: The timeout of each connect and read, which the time left before the policy's deadline caps.
        **request_kwargs: The arguments of `requests.request()` other than timeout, stream and hooks.

    Raises:
        EgressRefusedError: The request, or a redirect, would connect to an address the policy does not allow.
        EgressLimitError: The request redirected too often, or the response was too large, used an encoding other than
            gzip or deflate, or took longer than the policy allows.
        requests.RequestException: The request failed, as requests reports it.
    """
    call = _Call(policy)
    token = _current_call.set(call)
    session = requests.Session()
    session.trust_env = False
    session.max_redirects = policy.max_redirects
    session.headers["Accept-Encoding"] = ACCEPT_ENCODING
    adapter = _PolicyAdapter(policy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    call.start_timer()
    try:
        response = session.request(
            timeout=timeout,
            stream=True,
            hooks={"response": partial(_read_body, call)},
            **request_kwargs,
        )
        call.check()
        return response
    except EgressError:
        raise
    except requests.TooManyRedirects as e:
        raise EgressLimitError(f"The request was redirected more than {policy.max_redirects} times.") from e
    except Exception as e:
        if call.expired or (isinstance(e, requests.Timeout) and call.remaining() < DEADLINE_SLACK_SECONDS):
            raise _timed_out(policy.timeout_seconds) from e
        raise
    finally:
        call.close()
        _current_call.reset(token)
        session.close()


class _PublicNetworkBackend(httpcore.AsyncNetworkBackend):
    """Connects only to a public address the host resolved to."""

    def __init__(self):
        self._backend = httpcore.AnyIOBackend()

    async def connect_tcp(
        self,
        host: str,
        port: int,
        timeout: float | None = None,
        local_address: str | None = None,
        socket_options: Any = None,
    ) -> httpcore.AsyncNetworkStream:
        try:
            addresses = await resolve_public_async(host, port)
        except socket.gaierror as e:
            raise httpcore.ConnectError(str(e)) from e
        error: Exception | None = None
        for address in addresses:
            try:
                return await self._backend.connect_tcp(
                    address, port, timeout=timeout, local_address=local_address, socket_options=socket_options
                )
            except (httpcore.ConnectError, httpcore.ConnectTimeout) as e:
                error = e
        raise error

    async def connect_unix_socket(self, path: str, timeout: float | None = None, socket_options: Any = None):
        raise EgressRefusedError("Requests may not connect to a Unix socket.")

    async def sleep(self, seconds: float) -> None:
        await self._backend.sleep(seconds)


class _PublicTransport(httpx.AsyncHTTPTransport):
    def __init__(self):
        super().__init__(verify=_ssl_context(), trust_env=False)
        # httpx takes no network backend, so the pool it built is replaced by one with the same settings that has it.
        self._pool = httpcore.AsyncConnectionPool(ssl_context=_ssl_context(), network_backend=_PublicNetworkBackend())


class _PolicyTransport(httpx.AsyncBaseTransport):
    """Sends each request, every redirect included, through the address check unless the policy exempts its URL."""

    def __init__(self, policy: EgressPolicy):
        self._policy = policy
        self._public = _PublicTransport()
        self._exempt = httpx.AsyncHTTPTransport(verify=_ssl_context(), trust_env=False)

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        transport = self._exempt if self._policy.is_exempt(str(request.url)) else self._public
        return await transport.handle_async_request(request)

    async def aclose(self) -> None:
        await self._public.aclose()
        await self._exempt.aclose()


async def _read_body_async(call: _Call, response: httpx.Response) -> None:
    """Reads the body into the response, every redirect included, which httpx would otherwise read whole.

    The body of a redirect counts against the budget and is then dropped.
    """
    try:
        decoder = _BoundedDecoder(response.headers.get("content-encoding"), call)
        chunks = [decoder.decode(raw) async for raw in response.aiter_raw(READ_CHUNK_BYTES)]
        chunks.append(decoder.flush())
    except BaseException:
        await response.aclose()
        raise
    response._content = b"" if response.has_redirect_location else b"".join(chunks)


async def send_async(policy: EgressPolicy, *, timeout: float | None = None, **request_kwargs: Any) -> httpx.Response:
    """Sends a request with httpx under policy and returns the response with its body read.

    Args:
        policy: The policy to apply.
        timeout: The timeout of each connect, read and write, capped at the policy's timeout.
        **request_kwargs: The arguments of `httpx.AsyncClient.request()` other than timeout.

    Raises:
        EgressRefusedError: The request, or a redirect, would connect to an address the policy does not allow.
        EgressLimitError: The request redirected too often, or the response was too large, used an encoding other than
            gzip or deflate, or took longer than the policy allows.
        httpx.HTTPError: The request failed, as httpx reports it.
    """
    call = _Call(policy)
    async with httpx.AsyncClient(
        transport=_PolicyTransport(policy),
        follow_redirects=True,
        max_redirects=policy.max_redirects,
        trust_env=False,
        timeout=policy.timeout_seconds,
        headers={"Accept-Encoding": ACCEPT_ENCODING},
        event_hooks={"response": [partial(_read_body_async, call)]},
    ) as client:
        deadline = asyncio.timeout(policy.timeout_seconds)
        try:
            async with deadline:
                return await client.request(
                    timeout=min(timeout or policy.timeout_seconds, policy.timeout_seconds), **request_kwargs
                )
        except TimeoutError as e:
            if deadline.expired():
                raise _timed_out(policy.timeout_seconds) from e
            raise
        except httpx.TimeoutException as e:
            if asyncio.get_running_loop().time() > deadline.when() - DEADLINE_SLACK_SECONDS:
                raise _timed_out(policy.timeout_seconds) from e
            raise
        except httpx.TooManyRedirects as e:
            raise EgressLimitError(f"The request was redirected more than {policy.max_redirects} times.") from e
