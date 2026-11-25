"""HTTP clients that send a node's request under an `EgressPolicy`.

Each request gets its own session or client, so no cookie or connection outlives it or reaches another run. The address
check runs where the clients open a TCP connection: the host is resolved once, every address it resolves to must be
public, and the connection goes to one of those addresses, so a name that resolves differently the next time cannot
redirect it. TLS still verifies the certificate against the host name.
"""

import asyncio
import ipaddress
import socket
import sys
import threading
import zlib
from contextvars import ContextVar
from functools import partial
from typing import Any

import httpcore
import httpx
import requests
from requests.adapters import BaseAdapter, HTTPAdapter
from urllib3.connection import HTTPConnection, HTTPSConnection
from urllib3.connectionpool import HTTPConnectionPool, HTTPSConnectionPool
from urllib3.exceptions import ConnectTimeoutError, NameResolutionError, NewConnectionError
from urllib3.util import connection as urllib3_connection

from dynamiq.connections.connections import _shared_ssl_context
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

# httpx decodes a whole chunk at once, so a small compressed body could expand past any limit in one step. Responses are
# read raw and decoded here with a bound, which only zlib's formats allow; clients ask for those alone.
ACCEPT_ENCODING = "gzip, deflate"

READ_CHUNK_BYTES = 64 * 1024


class EgressError(Exception):
    """A request the egress policy stopped. Sending it again gives the same result."""


class EgressRefusedError(EgressError):
    """A request that would connect to an address the policy does not allow."""


class EgressLimitError(EgressError):
    """A response that is too large, takes too long or uses an encoding the policy cannot bound."""


def is_public_address(address: str) -> bool:
    """Reports whether address is a public IPv4 address."""
    try:
        ip = ipaddress.ip_address(address)
    except ValueError:
        return False
    return ip.version == 4 and not any(ip in network for network in DENIED_NETWORKS)


def _check_host(host: str) -> str:
    """The host to resolve, without brackets. Raises EgressRefusedError for an IPv6 address."""
    host = host.strip("[]")
    if ":" in host:
        raise EgressRefusedError(f"{host} is an IPv6 address. Only public IPv4 addresses are allowed.")
    return host


def _public_addresses(host: str, addresses: list[str]) -> list[str]:
    if not addresses:
        raise socket.gaierror(socket.EAI_NONAME, f"{host} has no IPv4 address")
    # One answer outside the public internet refuses the host, so it cannot pass by sometimes resolving to it.
    if not all(is_public_address(address) for address in addresses):
        raise EgressRefusedError(f"{host} resolves to an address outside the public internet, which is not allowed.")
    return addresses


def resolve_public(host: str, port: int) -> list[str]:
    """The IPv4 addresses host resolves to, when all of them are public.

    Raises:
        EgressRefusedError: host is an IPv6 address, or one of its addresses is not public.
        socket.gaierror: host has no IPv4 address.
    """
    host = _check_host(host)
    infos = socket.getaddrinfo(host, port, socket.AF_INET, socket.SOCK_STREAM)
    return _public_addresses(host, list(dict.fromkeys(info[4][0] for info in infos)))


async def resolve_public_async(host: str, port: int) -> list[str]:
    """`resolve_public()` for the running event loop."""
    host = _check_host(host)
    infos = await asyncio.get_running_loop().getaddrinfo(host, port, family=socket.AF_INET, type=socket.SOCK_STREAM)
    return _public_addresses(host, list(dict.fromkeys(info[4][0] for info in infos)))


class _SyncCall:
    """The connections one synchronous request opens, closed when its time is up.

    requests applies its timeout to each read, so a server that sends a byte now and then would hold the request, and
    the thread, as long as it likes.
    """

    def __init__(self, timeout_seconds: float):
        self.expired = False
        self._connections: list[HTTPConnection] = []
        self._lock = threading.Lock()
        self._timer = threading.Timer(timeout_seconds, self._expire)
        self._timer.daemon = True

    def start(self) -> None:
        self._timer.start()

    def cancel(self) -> None:
        self._timer.cancel()

    def track(self, connection: HTTPConnection) -> None:
        with self._lock:
            self._connections.append(connection)

    def _expire(self) -> None:
        with self._lock:
            self.expired = True
            connections = list(self._connections)
        for connection in connections:
            sock = connection.sock
            if sock is None:
                continue
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass


_current_call: ContextVar[_SyncCall | None] = ContextVar("dynamiq_egress_call", default=None)


def _track(connection: HTTPConnection) -> None:
    if (call := _current_call.get()) is not None:
        call.track(connection)


class _TrackedConnectionMixin:
    def _new_conn(self) -> socket.socket:
        _track(self)
        return super()._new_conn()


class _PublicConnectionMixin:
    """Connects only to a public address the host resolved to."""

    def _new_conn(self) -> socket.socket:
        _track(self)
        try:
            addresses = resolve_public(self._dns_host, self.port)
        except socket.gaierror as e:
            raise NameResolutionError(self.host, self, e) from e
        error: Exception | None = None
        for address in addresses:
            try:
                sock = urllib3_connection.create_connection(
                    (address, self.port),
                    self.timeout,
                    source_address=self.source_address,
                    socket_options=self.socket_options,
                )
            except TimeoutError:
                error = ConnectTimeoutError(
                    self, f"Connection to {self.host} timed out. (connect timeout={self.timeout})"
                )
            except OSError as e:
                error = NewConnectionError(self, f"Failed to establish a new connection: {e}")
            else:
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

    def send(self, request: requests.PreparedRequest, **kwargs) -> requests.Response:
        adapter = self._exempt if self._policy.is_exempt(request.url) else self._public
        return adapter.send(request, **kwargs)

    def close(self) -> None:
        self._public.close()
        self._exempt.close()


def _too_large(max_bytes: int) -> EgressLimitError:
    return EgressLimitError(f"The response is larger than {max_bytes} bytes, the most allowed.")


def _read_body(max_bytes: int, response: requests.Response, **kwargs) -> requests.Response:
    """Reads the body into the response, redirects included, which requests would otherwise read whole."""
    chunks = []
    size = 0
    try:
        for chunk in response.iter_content(chunk_size=READ_CHUNK_BYTES):
            size += len(chunk)
            if size > max_bytes:
                raise _too_large(max_bytes)
            chunks.append(chunk)
    except BaseException:
        response.close()
        raise
    response._content = b"".join(chunks)
    response._content_consumed = True
    return response


def send(policy: EgressPolicy, *, timeout: float | None = None, **request_kwargs: Any) -> requests.Response:
    """Sends a request with requests under policy and returns the response with its body read.

    Args:
        policy: The policy to apply.
        timeout: The timeout of each connect and read, capped at the policy's timeout.
        **request_kwargs: The arguments of `requests.request()` other than timeout, stream and hooks.

    Raises:
        EgressRefusedError: The request, or a redirect, would connect to an address the policy does not allow.
        EgressLimitError: The response was too large, or the request took longer than the policy allows.
        requests.RequestException: The request failed, as requests reports it.
    """
    call = _SyncCall(policy.timeout_seconds)
    token = _current_call.set(call)
    session = requests.Session()
    session.trust_env = False
    session.max_redirects = policy.max_redirects
    adapter = _PolicyAdapter(policy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    call.start()
    try:
        return session.request(
            timeout=min(timeout or policy.timeout_seconds, policy.timeout_seconds),
            stream=True,
            hooks={"response": partial(_read_body, policy.max_response_bytes)},
            **request_kwargs,
        )
    except EgressError:
        raise
    except Exception as e:
        if call.expired:
            raise EgressLimitError(f"The request took longer than {policy.timeout_seconds:g} seconds.") from e
        raise
    finally:
        call.cancel()
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
        ssl_context = _shared_ssl_context()
        super().__init__(verify=ssl_context, trust_env=False)
        # httpx takes no network backend, so the pool it built is replaced by one with the same settings that has it.
        self._pool = httpcore.AsyncConnectionPool(ssl_context=ssl_context, network_backend=_PublicNetworkBackend())


class _PolicyTransport(httpx.AsyncBaseTransport):
    """Sends each request, every redirect included, through the address check unless the policy exempts its URL."""

    def __init__(self, policy: EgressPolicy):
        self._policy = policy
        self._public = _PublicTransport()
        self._exempt = httpx.AsyncHTTPTransport(verify=_shared_ssl_context(), trust_env=False)

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        transport = self._exempt if self._policy.is_exempt(str(request.url)) else self._public
        return await transport.handle_async_request(request)

    async def aclose(self) -> None:
        await self._public.aclose()
        await self._exempt.aclose()


class _BoundedDecoder:
    """Decodes a gzip, deflate or plain body without producing more than max_bytes at any step."""

    def __init__(self, content_encoding: str | None, max_bytes: int):
        codings = [
            coding
            for coding in (part.strip().lower() for part in (content_encoding or "").split(","))
            if coding and coding != "identity"
        ]
        if len(codings) > 1 or (codings and codings[0] not in ("gzip", "x-gzip", "deflate")):
            raise EgressLimitError(f"Responses with the content encoding {content_encoding!r} are not supported.")
        self._coding = codings[0] if codings else None
        if self._coding in ("gzip", "x-gzip"):
            self._decompressor = zlib.decompressobj(zlib.MAX_WBITS | 16)
        elif self._coding == "deflate":
            self._decompressor = zlib.decompressobj()
        else:
            self._decompressor = None
        self._max_bytes = max_bytes
        self._remaining = max_bytes
        self._started = False

    def _take(self, decoded: bytes) -> bytes:
        if len(decoded) > self._remaining:
            raise _too_large(self._max_bytes)
        self._remaining -= len(decoded)
        return decoded

    def decode(self, data: bytes) -> bytes:
        if self._decompressor is None:
            return self._take(data)
        try:
            decoded = self._decompressor.decompress(data, self._remaining + 1)
        except zlib.error as e:
            # Some servers send raw deflate data without the zlib header, which httpx accepts too.
            if self._coding != "deflate" or self._started:
                raise httpx.DecodingError(str(e)) from e
            self._decompressor = zlib.decompressobj(-zlib.MAX_WBITS)
            decoded = self._decompressor.decompress(data, self._remaining + 1)
        self._started = True
        return self._take(decoded)

    def flush(self) -> bytes:
        if self._decompressor is None:
            return b""
        try:
            return self._take(self._decompressor.flush())
        except zlib.error as e:
            raise httpx.DecodingError(str(e)) from e


async def _read_body_async(max_bytes: int, response: httpx.Response) -> None:
    """Reads the body into the response, redirects included, which httpx would otherwise read whole."""
    try:
        decoder = _BoundedDecoder(response.headers.get("content-encoding"), max_bytes)
        chunks = [decoder.decode(raw) async for raw in response.aiter_raw(READ_CHUNK_BYTES)]
        chunks.append(decoder.flush())
    except BaseException:
        await response.aclose()
        raise
    response._content = b"".join(chunks)


async def send_async(policy: EgressPolicy, *, timeout: float | None = None, **request_kwargs: Any) -> httpx.Response:
    """Sends a request with httpx under policy and returns the response with its body read.

    Args:
        policy: The policy to apply.
        timeout: The timeout of each connect, read and write, capped at the policy's timeout.
        **request_kwargs: The arguments of `httpx.AsyncClient.request()` other than timeout.

    Raises:
        EgressRefusedError: The request, or a redirect, would connect to an address the policy does not allow.
        EgressLimitError: The response was too large or used an encoding other than gzip or deflate, or the request
            took longer than the policy allows.
        httpx.HTTPError: The request failed, as httpx reports it.
    """
    async with httpx.AsyncClient(
        transport=_PolicyTransport(policy),
        follow_redirects=True,
        max_redirects=policy.max_redirects,
        trust_env=False,
        timeout=policy.timeout_seconds,
        headers={"Accept-Encoding": ACCEPT_ENCODING},
        event_hooks={"response": [partial(_read_body_async, policy.max_response_bytes)]},
    ) as client:
        deadline = asyncio.timeout(policy.timeout_seconds)
        try:
            async with deadline:
                return await client.request(
                    timeout=min(timeout or policy.timeout_seconds, policy.timeout_seconds), **request_kwargs
                )
        except TimeoutError as e:
            if deadline.expired():
                raise EgressLimitError(f"The request took longer than {policy.timeout_seconds:g} seconds.") from e
            raise
