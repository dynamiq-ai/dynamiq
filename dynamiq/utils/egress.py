"""HTTP clients that send a node's request under an `EgressPolicy`.

Each request gets its own session or client, so no cookie or connection outlives it or reaches another run. The address
check runs where the clients open a TCP connection: the host is resolved once, every address it resolves to must be
public, and the connection goes to one of those addresses, so a name that resolves differently the next time cannot
redirect it. TLS still verifies the certificate against the host name.
"""

import asyncio
import ipaddress
import os
import re
import socket
import ssl
import sys
import threading
import time
import zlib
from collections.abc import Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from contextvars import ContextVar
from functools import cache, partial
from typing import Any
from urllib.parse import SplitResult, urlsplit

import httpcore
import httpx
import requests
from requests.adapters import BaseAdapter, HTTPAdapter
from urllib3 import exceptions as urllib3_exceptions
from urllib3.connection import HTTPConnection, HTTPSConnection
from urllib3.connectionpool import HTTPConnectionPool, HTTPSConnectionPool
from urllib3.exceptions import ConnectTimeoutError, NameResolutionError, NewConnectionError
from urllib3.util import connection as urllib3_connection

from dynamiq.types.egress import DEFAULT_PORTS, EgressPolicy

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

# What the response headers of a request, its redirects included, may take. The body has a budget of its own on the
# wire and once decoded, so headers cannot make room for more body, nor a body for more headers.
HEADER_ALLOWANCE_BYTES = 64 * 1024

# Each connect and read waits at most until the deadline, so its own timeout and the deadline fire together; a failure
# this close to the deadline is reported as the deadline.
DEADLINE_SLACK_SECONDS = 0.1

# getaddrinfo cannot be cancelled, so a lookup that hangs keeps its thread until the system resolver gives up. Lookups
# run in a pool of their own, so that such lookups cannot hold up the other clients in the process, and a request waits
# for one only until its deadline.
RESOLVER_THREADS = 16

# The most lookups waiting for a thread or running at once. A lookup holds its place until it ends, even after the
# request that started it gave up, and one beyond the limit fails at once instead of queueing behind lookups that may
# never end.
RESOLVER_CAPACITY = 4 * RESOLVER_THREADS

# Lookups of the host's own exempt URLs have a smaller pool of their own, so that tenants' lookups, which may hang,
# cannot crowd out the platform's own tools.
EXEMPT_RESOLVER_THREADS = 4
EXEMPT_RESOLVER_CAPACITY = 4 * EXEMPT_RESOLVER_THREADS


class _Resolver:
    """A pool of lookup threads and the places it admits lookups to."""

    def __init__(self, threads: int, capacity: int, name: str):
        self.pool = ThreadPoolExecutor(max_workers=threads, thread_name_prefix=name)
        self.slots = threading.BoundedSemaphore(capacity)


def _new_resolvers() -> dict[bool, _Resolver]:
    """The resolvers by public_only."""
    return {
        True: _Resolver(RESOLVER_THREADS, RESOLVER_CAPACITY, "dynamiq-egress-resolver"),
        False: _Resolver(EXEMPT_RESOLVER_THREADS, EXEMPT_RESOLVER_CAPACITY, "dynamiq-egress-exempt-resolver"),
    }


_resolvers = _new_resolvers()


def _reset_resolvers() -> None:
    # A forked child has none of the pools' threads, but a pool would still count one as idle and never start another.
    global _resolvers
    _resolvers = _new_resolvers()


if hasattr(os, "register_at_fork"):  # not on Windows, which does not fork
    os.register_at_fork(after_in_child=_reset_resolvers)


class EgressError(Exception):
    """A request the egress policy stopped."""


class EgressRefusedError(EgressError):
    """A request that would connect to an address the policy does not allow. Sending it again gives the same result."""


class EgressLimitError(EgressError):
    """A request that redirects too often, or a response that is too large, takes too long or uses an encoding the
    policy cannot bound. Sending it again gives the same result."""


class EgressBusyError(EgressError):
    """A request that could not start just now, because too many lookups are in progress in the process or the resolver
    failed for a while. It may succeed later."""


NOT_PUBLIC = "The host does not resolve to a public IPv4 address, which is required."


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
        raise EgressRefusedError("The host is an IPv6 address; only public IPv4 addresses are allowed.")
    return host


def _public_addresses(host: str, addresses: list[str]) -> list[str]:
    if not addresses:
        raise socket.gaierror(socket.EAI_NONAME, f"{host} has no IPv4 address")
    # One answer outside the public internet refuses the host, so it cannot pass by sometimes resolving to it.
    if not all(is_public_address(address) for address in addresses):
        raise EgressRefusedError(NOT_PUBLIC)
    return addresses


def _lookup(host: str, port: int, public_only: bool = True) -> list[str]:
    try:
        infos = socket.getaddrinfo(host, port, socket.AF_INET, socket.SOCK_STREAM)
    except socket.gaierror as e:
        if public_only and e.errno == socket.EAI_AGAIN:
            raise EgressBusyError("The host could not be looked up just now. Try again later.") from e
        # The same refusal as for a private address, so that it does not tell which internal names exist.
        if public_only:
            raise EgressRefusedError(NOT_PUBLIC) from e
        raise
    addresses = list(dict.fromkeys(info[4][0] for info in infos))
    if not public_only:
        if not addresses:
            raise socket.gaierror(socket.EAI_NONAME, f"{host} has no IPv4 address")
        return addresses
    if not addresses:
        raise EgressRefusedError(NOT_PUBLIC)
    return _public_addresses(host, addresses)


class _AbandonedLookupError(Exception):
    """A lookup that its request stopped waiting for before a thread took it up."""


def _charged_lookup(
    slots: threading.BoundedSemaphore, abandoned: threading.Event, host: str, port: int, public_only: bool
) -> list[str]:
    try:
        if abandoned.is_set():
            raise _AbandonedLookupError(host)
        return _lookup(host, port, public_only)
    finally:
        slots.release()


def _submit_lookup(host: str, port: int, public_only: bool) -> tuple[Future, threading.Event]:
    """Starts a lookup in the pool for public_only, and returns it with the event to set when nobody waits for it.

    Raises EgressBusyError when the pool is full. A lookup holds its place until a thread takes it off the queue: a
    cancelled future would leave its queue entry behind without giving the place back, so it is never cancelled, and a
    thread skips one that was given up instead. A lookup a thread has started holds its place until it ends.
    """
    resolver = _resolvers[public_only]
    if not resolver.slots.acquire(blocking=False):
        raise EgressBusyError("Too many host name lookups are in progress. Try again later.")
    abandoned = threading.Event()
    try:
        return resolver.pool.submit(_charged_lookup, resolver.slots, abandoned, host, port, public_only), abandoned
    except BaseException:
        resolver.slots.release()
        raise


def _literal(host: str) -> str | None:
    """host as a dotted-quad IPv4 address, which needs no lookup; None for a name or any other form of address."""
    try:
        return str(ipaddress.IPv4Address(host))
    except ValueError:
        return None


def resolve_public(host: str, port: int, timeout: float | None = None, public_only: bool = True) -> list[str]:
    """The IPv4 addresses host resolves to, when all of them are public, or any IPv4 addresses without public_only.

    Raises:
        EgressRefusedError: host is an IPv6 address, or one of its addresses is not public.
        EgressBusyError: too many lookups are in progress in the process.
        TimeoutError: the lookup did not finish within timeout seconds.
        socket.gaierror: host has no IPv4 address.
    """
    host = _check_host(host)
    if (address := _literal(host)) is not None:
        return _public_addresses(host, [address]) if public_only else [address]
    future, abandoned = _submit_lookup(host, port, public_only)
    try:
        return future.result(timeout=timeout)
    except FutureTimeoutError as e:
        abandoned.set()
        raise TimeoutError(f"Looking up {host} took too long.") from e


async def resolve_public_async(host: str, port: int, public_only: bool = True) -> list[str]:
    """`resolve_public()` for the running event loop, without a timeout of its own."""
    host = _check_host(host)
    if (address := _literal(host)) is not None:
        return _public_addresses(host, [address]) if public_only else [address]
    future, abandoned = _submit_lookup(host, port, public_only)
    lookup = asyncio.wrap_future(future)
    # Read the result even when nobody awaits it any more, so that its error is not reported as never retrieved.
    lookup.add_done_callback(lambda done: done.cancelled() or done.exception())
    try:
        # Shielded, so that cancelling the request does not cancel the lookup; see _submit_lookup.
        return await asyncio.shield(lookup)
    except asyncio.CancelledError:
        abandoned.set()
        raise


@cache
def _ssl_context() -> ssl.SSLContext:
    """The TLS context of the async client, which trusts the certifi bundle whatever the environment sets.

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
        self.header_bytes_left = HEADER_ALLOWANCE_BYTES
        self.wire_bytes_left = policy.max_response_bytes
        self.body_bytes_left = policy.max_response_bytes
        # Whether the hop being read went to an exempt URL, whose errors must not name the internal host.
        self.exempt_hop = False
        self.expired = False
        self._sockets: list[socket.socket] = []
        self._lock = threading.Lock()
        self._timer: threading.Timer | None = None

    def remaining(self) -> float:
        return self.deadline - time.monotonic()

    def take_header_bytes(self, size: int) -> None:
        """Counts size bytes of response headers against their allowance. Raises EgressLimitError when it runs out."""
        if size > self.header_bytes_left:
            raise EgressLimitError(
                f"The response headers are larger than {HEADER_ALLOWANCE_BYTES} bytes, the most allowed."
            )
        self.header_bytes_left -= size

    def take_wire_bytes(self, size: int) -> None:
        """Counts size bytes of body received against the budget. Raises EgressLimitError when it runs out."""
        if size > self.wire_bytes_left:
            raise _too_large(self.policy.max_response_bytes)
        self.wire_bytes_left -= size

    def check(self) -> None:
        """Raises EgressLimitError once the deadline has passed."""
        if self.expired or self.remaining() <= 0:
            raise _timed_out(self.policy.timeout_seconds)

    def timeout(self, timeout: Any) -> Any:
        """The timeout of the next connect or read: timeout, or the time left if that is shorter.

        A (connect, read) pair is capped item by item. Raises EgressLimitError once the deadline has passed.
        """
        if isinstance(timeout, tuple):
            return tuple(self.timeout(item) for item in timeout)
        remaining = self.remaining()
        if self.expired or remaining <= 0:
            raise _timed_out(self.policy.timeout_seconds)
        return min(timeout, remaining) if isinstance(timeout, (int, float)) else remaining

    def start_timer(self) -> None:
        self._timer = threading.Timer(max(self.remaining(), 0), self._expire)
        self._timer.daemon = True
        self._timer.start()

    def watch(self, sock: socket.socket) -> None:
        """Keeps a duplicate of sock to shut down at the deadline. Closes sock and raises if the deadline has passed."""
        try:
            duplicate = sock.dup()
        except OSError:
            sock.close()
            raise
        with self._lock:
            if not self.expired:
                self._sockets.append(duplicate)
                return
        duplicate.close()
        sock.close()
        raise _timed_out(self.policy.timeout_seconds)

    def _expire(self) -> None:
        # Under the lock, so that close() cannot close a duplicate and free its descriptor for another socket first.
        with self._lock:
            self.expired = True
            for sock in self._sockets:
                try:
                    sock.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass

    def close(self) -> None:
        if self._timer is not None:
            self._timer.cancel()
        with self._lock:
            for sock in self._sockets:
                sock.close()
            self._sockets = []


_current_call: ContextVar[_Call | None] = ContextVar("dynamiq_egress_call", default=None)


def _require_call() -> _Call:
    call = _current_call.get()
    if call is None:
        raise EgressRefusedError("A guarded connection was opened outside a guarded request.")
    return call


class _GuardedConnectionMixin:
    """Connects within the request's deadline to an address the host resolved to, a public one only if public_only."""

    public_only = True

    def _new_conn(self) -> socket.socket:
        call = _require_call()
        try:
            addresses = resolve_public(
                self._dns_host, self.port, timeout=call.timeout(None), public_only=self.public_only
            )
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


class _PublicConnectionMixin(_GuardedConnectionMixin):
    public_only = True


class _TrackedConnectionMixin(_GuardedConnectionMixin):
    public_only = False


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


# The headers a request to an exempt URL may carry: the ones the clients set themselves and those that describe its
# body. Front ends route by many other headers, so the rest are refused rather than listed.
EXEMPT_HEADERS = frozenset(
    {
        "accept",
        "accept-encoding",
        "accept-language",
        "authorization",
        "cache-control",
        "connection",
        "content-length",
        "content-type",
        "cookie",
        "host",
        "user-agent",
    }
)

# The values of Connection an exempt request may send: one that lists header names would have a front end drop them.
EXEMPT_CONNECTION_OPTIONS = frozenset({"keep-alive", "close"})

# What a failed request to an exempt URL says, instead of the client's text, which names the internal host and path.
EXEMPT_FAILURE = "The request to an exempt URL failed."

# A Host value: a host name, an IPv4 address or a bracketed IPv6 address, and an optional port, and nothing after them.
HOST_HEADER = re.compile(r"(?:\[[0-9A-Fa-f:.]+\]|[A-Za-z0-9._-]+)(?::[0-9]+)?")


def _same_authority(url: SplitResult, host_header: str) -> bool:
    value = host_header.strip()
    if not HOST_HEADER.fullmatch(value):
        return False
    try:
        header = urlsplit(f"//{value}")
        header_port = header.port
    except ValueError:
        return False
    if not header.hostname or header.username is not None or header.password is not None or header.path:
        return False
    if header_port == 0:
        return False
    default_port = DEFAULT_PORTS.get(url.scheme.lower())
    header_origin = (header.hostname.rstrip("."), header_port or default_port)
    return header_origin == ((url.hostname or "").rstrip("."), url.port or default_port)


def _check_exempt_headers(
    url: str, headers: Mapping[str, str], body_length: int | None, allowed: tuple[str, ...]
) -> None:
    """Raises EgressRefusedError when a request to an exempt URL carries a header that could have it served elsewhere.

    That is a header outside EXEMPT_HEADERS and allowed, a Host that names another host, a Connection other than
    "keep-alive" or "close", or a Content-Length other than body_length, when that is known. The exemption is decided
    on the URL, but a front end serves what such a header names instead, or reads a second request from the body.
    """
    parts = urlsplit(url)
    allowed_names = EXEMPT_HEADERS | {name.lower() for name in allowed}
    for name, value in headers.items():
        lowered = name.lower()
        if lowered not in allowed_names:
            raise EgressRefusedError(f"A request to an exempt URL may not set the {name!r} header.")
        if lowered == "host" and not _same_authority(parts, value):
            raise EgressRefusedError("A request to an exempt URL may not name another host in its Host header.")
        if (
            lowered == "connection"
            and not {option.strip().lower() for option in value.split(",")} <= EXEMPT_CONNECTION_OPTIONS
        ):
            raise EgressRefusedError("A request to an exempt URL may not set Connection options.")
        if lowered == "content-length" and body_length is not None and value.strip() != str(body_length):
            raise EgressRefusedError("A request to an exempt URL may not set a Content-Length its body does not have.")


def _exempt_hop(
    policy: EgressPolicy, first_hop: bool, url: str, headers: Mapping[str, str], body_length: int | None
) -> bool:
    """Reports whether a hop of a request skips the address check.

    Only the URL the node was given may be exempt: a redirect, even to another exempt URL, is checked as any other.
    Raises EgressRefusedError for an exempt URL whose headers would route it elsewhere.
    """
    if not first_hop or not policy.is_exempt(url):
        return False
    _check_exempt_headers(url, headers, body_length, policy.exempt_request_headers)
    return True


def _body_length(body: Any) -> int | None:
    """The length of a prepared request body, as requests counts it for Content-Length; None for a stream."""
    if body is None:
        return 0
    if isinstance(body, (bytes, bytearray, str)):
        return len(body)
    return None


class _PolicyAdapter(BaseAdapter):
    """Sends each request, every redirect included, through the address check unless the policy exempts its URL."""

    def __init__(self, policy: EgressPolicy):
        super().__init__()
        self._policy = policy
        self._public = _PoolAdapter({"http": _PublicHTTPConnectionPool, "https": _PublicHTTPSConnectionPool})
        self._exempt = _PoolAdapter({"http": _TrackedHTTPConnectionPool, "https": _TrackedHTTPSConnectionPool})
        self._first_hop = True

    def send(self, request: requests.PreparedRequest, timeout: float | None = None, **kwargs) -> requests.Response:
        # Set on every hop, so that a caller's own header cannot ask for an encoding the decoder cannot bound.
        request.headers["Accept-Encoding"] = ACCEPT_ENCODING
        exempt = _exempt_hop(self._policy, self._first_hop, request.url, request.headers, _body_length(request.body))
        self._first_hop = False
        call = _require_call()
        call.exempt_hop = exempt
        timeout = call.timeout(timeout)
        if not exempt:
            return self._public.send(request, timeout=timeout, **kwargs)
        try:
            return self._exempt.send(request, timeout=timeout, **kwargs)
        except requests.exceptions.Timeout as e:
            raise requests.exceptions.Timeout(EXEMPT_FAILURE) from e
        except requests.exceptions.ConnectionError as e:
            raise requests.exceptions.ConnectionError(EXEMPT_FAILURE) from e

    def close(self) -> None:
        self._public.close()
        self._exempt.close()


def _too_large(max_bytes: int) -> EgressLimitError:
    return EgressLimitError(f"The response is larger than {max_bytes} bytes, the most allowed.")


GZIP_MAGIC = b"\x1f\x8b"


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
        # Refused only when a body arrives: a response without one, such as a HEAD or a 304, cannot exceed a bound.
        self._unsupported = len(codings) > 1 or bool(codings and codings[0] not in SUPPORTED_ENCODINGS)
        self._call = call
        self._coding = codings[0] if codings and not self._unsupported else None
        self._decompressor = self._new_decompressor()
        self._started = False
        self._fed = False
        self._first_bytes = b""
        self._done = False
        self._pending = b""

    def _new_decompressor(self):
        if self._coding in ("gzip", "x-gzip"):
            return zlib.decompressobj(zlib.MAX_WBITS | 16)
        if self._coding == "deflate":
            return zlib.decompressobj()
        return None

    def _take(self, decoded: bytes) -> bytes:
        if len(decoded) > self._call.body_bytes_left:
            raise _too_large(self._call.policy.max_response_bytes)
        self._call.body_bytes_left -= len(decoded)
        return decoded

    def _decompress(self, data: bytes) -> bytes:
        # Some servers send raw deflate data without the zlib header, which httpx and urllib3 accept too. Until the zlib
        # format has produced output, the bytes so far are kept to decode again as raw deflate if it turns out wrong.
        if self._coding == "deflate" and not self._started:
            self._first_bytes += data
        self._fed = True
        try:
            decoded = self._decompressor.decompress(data, self._call.body_bytes_left + 1)
        except zlib.error as e:
            if self._coding != "deflate" or self._started:
                raise EgressLimitError(f"The response body could not be decoded: {e}") from e
            self._decompressor = zlib.decompressobj(-zlib.MAX_WBITS)
            self._started = True
            data, self._first_bytes = self._first_bytes, b""
            try:
                return self._decompressor.decompress(data, self._call.body_bytes_left + 1)
            except zlib.error as e:
                raise EgressLimitError(f"The response body could not be decoded: {e}") from e
        if decoded or self._decompressor.eof:
            self._started = True
            self._first_bytes = b""
        return decoded

    def decode(self, data: bytes) -> bytes:
        self._call.check()
        if not data:
            return b""
        if self._unsupported:
            # The server's value is not quoted: it is not checked, and an exempt server is an internal one.
            raise EgressLimitError("Responses in a content encoding other than gzip or deflate are not supported.")
        self._call.take_wire_bytes(len(data))
        if self._decompressor is None:
            return self._take(data)
        decoded = bytearray()
        data = self._pending + data
        self._pending = b""
        while data and not self._done:
            if self._decompressor.eof:
                # data follows the end of a stream. A gzip body may hold several members one after another, which
                # urllib3 decodes too. Anything else is dropped, as httpx drops it, instead of piling up in zlib.
                if self._coding == "deflate":
                    self._done = True
                elif len(data) < len(GZIP_MAGIC) and GZIP_MAGIC.startswith(data):
                    self._pending = data
                elif data.startswith(GZIP_MAGIC):
                    self._decompressor = self._new_decompressor()
                    self._fed = False
                    continue
                else:
                    self._done = True
                break
            decoded += self._take(self._decompress(data))
            data = self._decompressor.unused_data if self._decompressor.eof else b""
        return bytes(decoded)

    def flush(self) -> None:
        """Raises EgressLimitError when the body ended inside a compressed stream, which leaves its data unverified."""
        if self._pending or (self._decompressor is not None and self._fed and not self._decompressor.eof):
            raise EgressLimitError("The response body ended before its compressed data did.")


def _raw_chunks(response: requests.Response):
    """The raw body, with urllib3's errors turned into requests' own, as `requests.Response.iter_content` does."""
    try:
        yield from response.raw.stream(READ_CHUNK_BYTES, decode_content=False)
    except urllib3_exceptions.ProtocolError as e:
        raise requests.exceptions.ChunkedEncodingError(e) from e
    except urllib3_exceptions.ReadTimeoutError as e:
        raise requests.exceptions.ConnectionError(e) from e
    except urllib3_exceptions.SSLError as e:
        raise requests.exceptions.SSLError(e) from e


def _read_body(call: _Call, response: requests.Response, **kwargs) -> requests.Response:
    """Reads the body into the response, every redirect included, which requests would otherwise read whole.

    The body of a redirect counts against the budget and is then dropped.
    """
    body = bytearray()
    try:
        # http.client keeps up to 100 header lines of 64 KiB each, and requests keeps every redirect's.
        call.take_header_bytes(sum(len(name) + len(value) + 4 for name, value in response.raw.headers.items()))
        decoder = _BoundedDecoder(response.headers.get("content-encoding"), call)
        for raw in _raw_chunks(response):
            body += decoder.decode(raw)
        decoder.flush()
    except requests.RequestException as e:
        response.close()
        if call.exempt_hop:
            raise type(e)(EXEMPT_FAILURE) from e
        raise
    except BaseException:
        response.close()
        raise
    response._content = b"" if response.is_redirect else bytes(body)
    response._content_consumed = True
    return response


def invalid_url_error() -> EgressRefusedError:
    """The refusal of a URL that is not an http or https URL with a host, which names neither."""
    return EgressRefusedError("Only http and https URLs with a host are allowed.")


def send(policy: EgressPolicy, *, timeout: float | None = None, **request_kwargs: Any) -> requests.Response:
    """Sends a request with requests under policy and returns the response with its body read.

    Args:
        policy: The policy to apply.
        timeout: The timeout of each connect and read, which the time left before the policy's deadline caps.
        **request_kwargs: The arguments of `requests.request()` other than timeout, stream and hooks.

    Raises:
        EgressRefusedError: The request, or a redirect, would connect to an address the policy does not allow, or its
            URL is not an http or https URL with a host.
        EgressLimitError: The request redirected too often, too many lookups were in progress, or the response was too
            large, incomplete, used an encoding other than gzip or deflate, or took longer than the policy allows.
        requests.RequestException: The request failed, as requests reports it.
    """
    call = _Call(policy)
    session = requests.Session()
    session.trust_env = False
    session.max_redirects = policy.max_redirects
    session.headers["Accept-Encoding"] = ACCEPT_ENCODING
    adapter = _PolicyAdapter(policy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    token = _current_call.set(call)
    try:
        call.start_timer()
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
    except (requests.exceptions.InvalidSchema, requests.exceptions.MissingSchema, requests.exceptions.InvalidURL) as e:
        raise invalid_url_error() from e
    except Exception as e:
        # Every connect and read waits at most until the deadline, so a failure this close to it is the deadline,
        # whichever error the client raised first.
        if call.expired or call.remaining() < DEADLINE_SLACK_SECONDS:
            raise _timed_out(policy.timeout_seconds) from e
        raise
    finally:
        call.close()
        _current_call.reset(token)
        session.close()


class _GuardedNetworkBackend(httpcore.AsyncNetworkBackend):
    """Connects to an IPv4 address the host resolved to in the bounded pool, a public one only if public_only."""

    def __init__(self, public_only: bool):
        self._public_only = public_only
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
            addresses = await resolve_public_async(host, port, public_only=self._public_only)
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


class _GuardedTransport(httpx.AsyncHTTPTransport):
    def __init__(self, public_only: bool):
        super().__init__(verify=_ssl_context(), trust_env=False)
        # httpx takes no network backend, so the pool it built is replaced by one with the same settings that has it.
        self._pool = httpcore.AsyncConnectionPool(
            ssl_context=_ssl_context(), network_backend=_GuardedNetworkBackend(public_only)
        )


class _PolicyTransport(httpx.AsyncBaseTransport):
    """Sends each request, every redirect included, through the address check unless the policy exempts its URL."""

    def __init__(self, policy: EgressPolicy, call: _Call):
        self._policy = policy
        self._call = call
        self._public = _GuardedTransport(public_only=True)
        self._exempt = _GuardedTransport(public_only=False)
        self._first_hop = True

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        # Set on every hop, so that a caller's own header cannot ask for an encoding the decoder cannot bound.
        request.headers["Accept-Encoding"] = ACCEPT_ENCODING
        try:
            body_length = len(request.content)
        except httpx.RequestNotRead:
            body_length = None
        exempt = _exempt_hop(self._policy, self._first_hop, str(request.url), request.headers, body_length)
        self._first_hop = False
        self._call.exempt_hop = exempt
        if not exempt:
            return await self._public.handle_async_request(request)
        try:
            return await self._exempt.handle_async_request(request)
        except httpx.TimeoutException as e:
            raise type(e)(EXEMPT_FAILURE, request=request) from e
        except httpx.TransportError as e:
            raise httpx.ConnectError(EXEMPT_FAILURE, request=request) from e

    async def aclose(self) -> None:
        await self._public.aclose()
        await self._exempt.aclose()


async def _read_body_async(call: _Call, response: httpx.Response) -> None:
    """Reads the body into the response, every redirect included, which httpx would otherwise read whole.

    The body of a redirect counts against the budget and is then dropped.
    """
    body = bytearray()
    try:
        call.take_header_bytes(sum(len(name) + len(value) + 4 for name, value in response.headers.raw))
        decoder = _BoundedDecoder(response.headers.get("content-encoding"), call)
        async for raw in response.aiter_raw(READ_CHUNK_BYTES):
            body += decoder.decode(raw)
        decoder.flush()
    except httpx.TransportError as e:
        await response.aclose()
        if call.exempt_hop:
            raise type(e)(EXEMPT_FAILURE, request=response.request) from e
        raise
    except BaseException:
        await response.aclose()
        raise
    response._content = b"" if response.has_redirect_location else bytes(body)


async def send_async(policy: EgressPolicy, *, timeout: float | None = None, **request_kwargs: Any) -> httpx.Response:
    """Sends a request with httpx under policy and returns the response with its body read.

    Args:
        policy: The policy to apply.
        timeout: The timeout of each connect, read and write, capped at the policy's timeout.
        **request_kwargs: The arguments of `httpx.AsyncClient.request()` other than timeout.

    Raises:
        EgressRefusedError: The request, or a redirect, would connect to an address the policy does not allow, or its
            URL is not an http or https URL with a host.
        EgressLimitError: The request redirected too often, too many lookups were in progress, or the response was too
            large, incomplete, used an encoding other than gzip or deflate, or took longer than the policy allows.
        httpx.HTTPError: The request failed, as httpx reports it.
    """
    call = _Call(policy)
    async with httpx.AsyncClient(
        transport=_PolicyTransport(policy, call),
        follow_redirects=True,
        max_redirects=policy.max_redirects,
        trust_env=False,
        timeout=policy.timeout_seconds,
        headers={"Accept-Encoding": ACCEPT_ENCODING},
        event_hooks={"response": [partial(_read_body_async, call)]},
    ) as client:
        # Counted from when the request began, like the sync client's deadline, not from here.
        deadline = asyncio.timeout(max(call.remaining(), 0))
        try:
            async with deadline:
                response = await client.request(
                    timeout=call.timeout(timeout or policy.timeout_seconds), **request_kwargs
                )
            call.check()
            return response
        except EgressError:
            raise
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
        except (httpx.UnsupportedProtocol, httpx.InvalidURL) as e:
            raise invalid_url_error() from e
