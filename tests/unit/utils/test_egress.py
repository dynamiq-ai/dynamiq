import asyncio
import datetime
import gzip
import ipaddress
import socket
import ssl
import threading
import time
import zlib
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock
from urllib.parse import parse_qs, urlsplit

import httpx
import pytest
import requests
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import NameOID

from dynamiq import connections
from dynamiq.nodes.agents.exceptions import ToolExecutionException
from dynamiq.nodes.node import ErrorHandling
from dynamiq.nodes.tools import HttpApiCall, ResponseType
from dynamiq.nodes.tools.http_api_call import EGRESS_ERROR_BODY_CHARS, HttpApiCallInputSchema
from dynamiq.runnables import RunnableConfig, RunnableStatus
from dynamiq.types.egress import EgressPolicy, set_default_egress_policy
from dynamiq.utils import egress

MAX_BYTES = 64 * 1024
GZIP_BOMB = gzip.compress(b"\0" * (64 * 1024 * 1024))


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    hits = 0

    def log_message(self, *args):
        pass

    def _send(self, status: int, body: bytes = b"", headers: dict[str, str] | None = None):
        self.send_response(status)
        for name, value in (headers or {}).items():
            self.send_header(name, value)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        url = urlsplit(self.path)
        query = {key: values[0] for key, values in parse_qs(url.query).items()}
        path = url.path.removeprefix("/exempt")
        if path == "/ok":
            self._send(200, b"hello", {"Content-Type": "text/plain"})
        elif path == "/redirect":
            self._send(302, b"", {"Location": query["to"]})
        elif path == "/redirect-big":
            self._send(302, b"x" * (MAX_BYTES + 1), {"Location": query["to"]})
        elif path == "/loop":
            self._send(302, b"", {"Location": "/loop"})
        elif path == "/error":
            self._send(500, b"e" * 10000)
        elif path == "/size":
            self._send(200, b"x" * int(query["n"]))
        elif path == "/gzip-bomb":
            self._send(200, GZIP_BOMB, {"Content-Encoding": "gzip"})
        elif path == "/gzip":
            self._send(200, gzip.compress(b"hello"), {"Content-Encoding": "gzip"})
        elif path == "/gzip-trailing":
            self._send(200, gzip.compress(b"hello") + b"\0" * (MAX_BYTES * 4), {"Content-Encoding": "gzip"})
        elif path == "/deflate-raw":
            compressor = zlib.compressobj(wbits=-zlib.MAX_WBITS)
            self._send(200, compressor.compress(b"hello") + compressor.flush(), {"Content-Encoding": "deflate"})
        elif path == "/brotli":
            self._send(200, b"\x0b\x02\x80hello\x03", {"Content-Encoding": "br"})
        elif path == "/zstd":
            self._send(200, b"\x28\xb5\x2f\xfd", {"Content-Encoding": "zstd"})
        elif path == "/gzip-members":
            self._send(200, gzip.compress(b"hello ") + gzip.compress(b"world"), {"Content-Encoding": "gzip"})
        elif path == "/empty-brotli":
            self._send(204, b"", {"Content-Encoding": "br"})
        elif path == "/chunked":
            self.send_response(200)
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            for _ in range(1000):
                self.wfile.write(b"2\r\nab\r\n")
            self.wfile.write(b"0\r\n\r\n")
        elif path == "/silent-body":
            Handler.hits += 1
            self.send_response(200)
            self.send_header("Content-Length", "100")
            self.end_headers()
            self.wfile.write(b"x" * 10)
            self.wfile.flush()
            time.sleep(2)
        elif path == "/slow-redirect":
            time.sleep(0.8)
            self._send(302, b"", {"Location": query["to"]})
        elif path == "/set-cookie":
            self._send(200, b"", {"Set-Cookie": "session=secret; Path=/"})
        elif path == "/echo-cookie":
            self._send(200, (self.headers.get("Cookie") or "").encode())
        elif path == "/drip":
            self.send_response(200)
            self.send_header("Content-Length", "1000")
            self.end_headers()
            self._drip(b"x" * 1000)
        elif path == "/drip-headers":
            self._drip(b"HTTP/1.1 200 OK\r\n" + b"X-Slow: " + b"y" * 1000 + b"\r\n\r\n")
        else:
            self._send(404)

    def _drip(self, data: bytes):
        try:
            for byte in data:
                self.wfile.write(bytes([byte]))
                self.wfile.flush()
                time.sleep(0.1)
        except OSError:
            pass


def _serve(ssl_context: ssl.SSLContext | None = None):
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    if ssl_context is not None:
        server.socket = ssl_context.wrap_socket(server.socket, server_side=True)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


@pytest.fixture
def server():
    server = _serve()
    yield server
    server.shutdown()
    server.server_close()


@pytest.fixture
def port(server) -> int:
    return server.server_address[1]


@pytest.fixture
def loopback_allowed(monkeypatch):
    """Counts loopback as public, so that a test server on 127.0.0.1 stands for a host on the internet."""
    monkeypatch.setattr(
        egress, "DENIED_NETWORKS", tuple(n for n in egress.DENIED_NETWORKS if n != ipaddress.IPv4Network("127.0.0.0/8"))
    )


@pytest.fixture
def resolver(monkeypatch):
    """Answers for the .test names the tests use; every other name and every address resolves as usual."""
    real_getaddrinfo = socket.getaddrinfo
    answers: dict[str, list[list[str]]] = {
        "public.test": [["127.0.0.1"]],
        "other.test": [["127.0.0.1"]],
        "internal.test": [["127.0.0.1"]],
        "private.test": [["10.0.0.1"]],
        "metadata.test": [["169.254.169.254"]],
        "mixed.test": [["127.0.0.1", "10.0.0.1"]],
        "v6.test": [[]],
        "slow.test": [["127.0.0.1"]],
    }
    delays = {"slow.test": 3.0}
    lookups: list[str] = []

    def getaddrinfo(host, port, family=0, type=0, proto=0, flags=0):
        if isinstance(host, bytes):
            host = host.decode()
        if host not in answers:
            return real_getaddrinfo(host, port, family, type, proto, flags)
        lookups.append(host)
        time.sleep(delays.get(host, 0))
        sequence = answers[host]
        addresses = sequence.pop(0) if len(sequence) > 1 else sequence[0]
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (address, port)) for address in addresses]

    monkeypatch.setattr(socket, "getaddrinfo", getaddrinfo)
    return answers, lookups


def _send_sync(policy: EgressPolicy, url: str, **kwargs) -> requests.Response:
    return egress.send(policy, method="GET", url=url, timeout=30, **kwargs)


async def _send_async(policy: EgressPolicy, url: str, **kwargs) -> httpx.Response:
    return await egress.send_async(policy, method="GET", url=url, timeout=30, **kwargs)


POLICY = EgressPolicy(max_response_bytes=MAX_BYTES, timeout_seconds=5)


@pytest.mark.parametrize(
    "address, public",
    [
        ("8.8.8.8", True),
        ("1.1.1.1", True),
        ("10.1.2.3", False),
        ("172.31.0.1", False),
        ("192.168.1.1", False),
        ("100.64.0.1", False),
        ("127.0.0.1", False),
        ("169.254.169.254", False),
        ("0.0.0.0", False),  # nosec B104
        ("192.0.2.1", False),
        ("198.18.0.1", False),
        ("224.0.0.1", False),
        ("255.255.255.255", False),
        ("::1", False),
        ("2606:4700:4700::1111", False),
        ("::ffff:8.8.8.8", False),
        ("not an address", False),
    ],
)
def test_is_public_address(address, public):
    assert egress.is_public_address(address) is public


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1:{port}/ok",
        "http://localhost:{port}/ok",
        "http://2130706433:{port}/ok",
        "http://0x7f.1:{port}/ok",
        "http://[::1]:{port}/ok",
        "http://[::ffff:127.0.0.1]:{port}/ok",
        "http://private.test:{port}/ok",
        "http://metadata.test:{port}/ok",
        "http://mixed.test:{port}/ok",
    ],
)
def test_send_refuses_private_addresses(resolver, port, url):
    with pytest.raises(egress.EgressRefusedError):
        _send_sync(POLICY, url.format(port=port))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1:{port}/ok",
        "http://localhost:{port}/ok",
        "http://2130706433:{port}/ok",
        "http://[::1]:{port}/ok",
        "http://private.test:{port}/ok",
        "http://metadata.test:{port}/ok",
        "http://mixed.test:{port}/ok",
    ],
)
async def test_send_async_refuses_private_addresses(resolver, port, url):
    with pytest.raises(egress.EgressRefusedError):
        await _send_async(POLICY, url.format(port=port))


def test_send_reaches_public_address(resolver, loopback_allowed, port):
    response = _send_sync(POLICY, f"http://public.test:{port}/ok")

    assert response.status_code == 200
    assert response.text == "hello"


@pytest.mark.asyncio
async def test_send_async_reaches_public_address(resolver, loopback_allowed, port):
    response = await _send_async(POLICY, f"http://public.test:{port}/ok")

    assert response.status_code == 200
    assert response.text == "hello"


def test_send_dials_the_address_it_checked(resolver, loopback_allowed, port):
    answers, lookups = resolver
    answers["public.test"] = [["127.0.0.1"], ["10.0.0.1"]]

    assert _send_sync(POLICY, f"http://public.test:{port}/ok").status_code == 200
    assert lookups == ["public.test"]
    with pytest.raises(egress.EgressRefusedError):
        _send_sync(POLICY, f"http://public.test:{port}/ok")


@pytest.mark.asyncio
async def test_send_async_dials_the_address_it_checked(resolver, loopback_allowed, port):
    answers, lookups = resolver
    answers["public.test"] = [["127.0.0.1"], ["10.0.0.1"]]

    assert (await _send_async(POLICY, f"http://public.test:{port}/ok")).status_code == 200
    assert lookups == ["public.test"]
    with pytest.raises(egress.EgressRefusedError):
        await _send_async(POLICY, f"http://public.test:{port}/ok")


def test_send_checks_every_redirect(resolver, loopback_allowed, port):
    with pytest.raises(egress.EgressRefusedError):
        _send_sync(POLICY, f"http://public.test:{port}/redirect?to=http://metadata.test:{port}/ok")


@pytest.mark.asyncio
async def test_send_async_checks_every_redirect(resolver, loopback_allowed, port):
    with pytest.raises(egress.EgressRefusedError):
        await _send_async(POLICY, f"http://public.test:{port}/redirect?to=http://metadata.test:{port}/ok")


def test_send_follows_redirect_to_public_address(resolver, loopback_allowed, port):
    response = _send_sync(POLICY, f"http://public.test:{port}/redirect?to=http://other.test:{port}/ok")

    assert response.text == "hello"


def test_send_exempts_only_matching_urls(resolver, port):
    policy = POLICY.model_copy(update={"exempt_url_prefixes": (f"http://internal.test:{port}/exempt/",)})

    assert _send_sync(policy, f"http://internal.test:{port}/exempt/ok").text == "hello"
    with pytest.raises(egress.EgressRefusedError):
        _send_sync(policy, f"http://internal.test:{port}/ok")
    with pytest.raises(egress.EgressRefusedError):
        _send_sync(policy, f"http://internal.test:{port}/exempt/redirect?to=http://internal.test:{port}/ok")


@pytest.mark.asyncio
async def test_send_async_exempts_only_matching_urls(resolver, port):
    policy = POLICY.model_copy(update={"exempt_url_prefixes": (f"http://internal.test:{port}/exempt/",)})

    assert (await _send_async(policy, f"http://internal.test:{port}/exempt/ok")).text == "hello"
    with pytest.raises(egress.EgressRefusedError):
        await _send_async(policy, f"http://internal.test:{port}/ok")
    with pytest.raises(egress.EgressRefusedError):
        await _send_async(policy, f"http://internal.test:{port}/exempt/redirect?to=http://internal.test:{port}/ok")


def test_send_limits_response_size(resolver, loopback_allowed, port):
    assert len(_send_sync(POLICY, f"http://public.test:{port}/size?n={MAX_BYTES}").content) == MAX_BYTES
    with pytest.raises(egress.EgressLimitError):
        _send_sync(POLICY, f"http://public.test:{port}/size?n={MAX_BYTES + 1}")


@pytest.mark.asyncio
async def test_send_async_limits_response_size(resolver, loopback_allowed, port):
    assert len((await _send_async(POLICY, f"http://public.test:{port}/size?n={MAX_BYTES}")).content) == MAX_BYTES
    with pytest.raises(egress.EgressLimitError):
        await _send_async(POLICY, f"http://public.test:{port}/size?n={MAX_BYTES + 1}")


@pytest.mark.parametrize("path", ["/gzip-bomb", "/gzip-trailing", "/redirect-big?to=/ok", "/brotli", "/zstd"])
def test_send_limits_decoded_and_redirect_bodies(resolver, loopback_allowed, port, path):
    with pytest.raises(egress.EgressLimitError):
        _send_sync(POLICY, f"http://public.test:{port}{path}")


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["/gzip-bomb", "/gzip-trailing", "/redirect-big?to=/ok", "/brotli", "/zstd"])
async def test_send_async_limits_decoded_and_redirect_bodies(resolver, loopback_allowed, port, path):
    with pytest.raises(egress.EgressLimitError):
        await _send_async(POLICY, f"http://public.test:{port}{path}")


@pytest.mark.parametrize("path", ["/gzip", "/deflate-raw"])
def test_send_decodes_gzip_and_deflate(resolver, loopback_allowed, port, path):
    assert _send_sync(POLICY, f"http://public.test:{port}{path}").content == b"hello"


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["/gzip", "/deflate-raw"])
async def test_send_async_decodes_gzip_and_deflate(resolver, loopback_allowed, port, path):
    assert (await _send_async(POLICY, f"http://public.test:{port}{path}")).content == b"hello"


def test_send_drops_redirect_bodies(resolver, loopback_allowed, port):
    response = _send_sync(POLICY, f"http://public.test:{port}/redirect?to=/ok")

    assert response.text == "hello"
    assert [r.content for r in response.history] == [b""]


@pytest.mark.parametrize("path", ["/drip", "/drip-headers"])
def test_send_stops_at_the_policy_timeout(resolver, loopback_allowed, port, path):
    policy = POLICY.model_copy(update={"timeout_seconds": 1})
    start = time.monotonic()

    with pytest.raises(egress.EgressLimitError):
        _send_sync(policy, f"http://public.test:{port}{path}")
    assert time.monotonic() - start < 5


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["/drip", "/drip-headers"])
async def test_send_async_stops_at_the_policy_timeout(resolver, loopback_allowed, port, path):
    policy = POLICY.model_copy(update={"timeout_seconds": 1})
    start = time.monotonic()

    with pytest.raises(egress.EgressLimitError):
        await _send_async(policy, f"http://public.test:{port}{path}")
    assert time.monotonic() - start < 5


def test_send_limits_redirects(resolver, loopback_allowed, port):
    with pytest.raises(egress.EgressLimitError, match="redirected"):
        _send_sync(POLICY, f"http://public.test:{port}/loop")


@pytest.mark.asyncio
async def test_send_async_limits_redirects(resolver, loopback_allowed, port):
    with pytest.raises(egress.EgressLimitError, match="redirected"):
        await _send_async(POLICY, f"http://public.test:{port}/loop")


def test_send_ignores_proxy_settings(resolver, loopback_allowed, port, monkeypatch):
    for name in ("HTTP_PROXY", "http_proxy", "ALL_PROXY", "all_proxy"):
        monkeypatch.setenv(name, "http://127.0.0.1:9")

    assert _send_sync(POLICY, f"http://public.test:{port}/ok").text == "hello"


@pytest.mark.asyncio
async def test_send_async_ignores_proxy_settings(resolver, loopback_allowed, port, monkeypatch):
    for name in ("HTTP_PROXY", "http_proxy", "ALL_PROXY", "all_proxy"):
        monkeypatch.setenv(name, "http://127.0.0.1:9")

    assert (await _send_async(POLICY, f"http://public.test:{port}/ok")).text == "hello"


def test_send_refuses_explicit_proxy(resolver, loopback_allowed, port):
    with pytest.raises(egress.EgressRefusedError):
        _send_sync(POLICY, f"http://public.test:{port}/ok", proxies={"http": "http://127.0.0.1:9"})


def test_send_keeps_no_cookies_between_requests(resolver, loopback_allowed, port):
    _send_sync(POLICY, f"http://public.test:{port}/set-cookie")

    assert _send_sync(POLICY, f"http://public.test:{port}/echo-cookie").text == ""


@pytest.mark.asyncio
async def test_send_async_keeps_no_cookies_between_requests(resolver, loopback_allowed, port):
    await _send_async(POLICY, f"http://public.test:{port}/set-cookie")

    assert (await _send_async(POLICY, f"http://public.test:{port}/echo-cookie")).text == ""


@pytest.fixture
def tls(tmp_path):
    """A server whose certificate names public.test only, and the CA file that trusts it."""
    now = datetime.datetime.now(datetime.timezone.utc)
    ca_key = ec.generate_private_key(ec.SECP256R1())
    ca_name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "test CA")])
    ca_cert = (
        x509.CertificateBuilder()
        .subject_name(ca_name)
        .issuer_name(ca_name)
        .public_key(ca_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(days=1))
        .not_valid_after(now + datetime.timedelta(days=1))
        .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
        .add_extension(
            x509.KeyUsage(
                digital_signature=False,
                content_commitment=False,
                key_encipherment=False,
                data_encipherment=False,
                key_agreement=False,
                key_cert_sign=True,
                crl_sign=True,
                encipher_only=False,
                decipher_only=False,
            ),
            critical=True,
        )
        .add_extension(x509.SubjectKeyIdentifier.from_public_key(ca_key.public_key()), critical=False)
        .sign(ca_key, hashes.SHA256())
    )
    key = ec.generate_private_key(ec.SECP256R1())
    cert = (
        x509.CertificateBuilder()
        .subject_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "public.test")]))
        .issuer_name(ca_name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(days=1))
        .not_valid_after(now + datetime.timedelta(days=1))
        .add_extension(x509.SubjectAlternativeName([x509.DNSName("public.test")]), critical=False)
        .add_extension(x509.AuthorityKeyIdentifier.from_issuer_public_key(ca_key.public_key()), critical=False)
        .sign(ca_key, hashes.SHA256())
    )
    ca_file = tmp_path / "ca.pem"
    ca_file.write_bytes(ca_cert.public_bytes(serialization.Encoding.PEM))
    cert_file = tmp_path / "cert.pem"
    cert_file.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    key_file = tmp_path / "key.pem"
    key_file.write_bytes(
        key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption())
    )
    server_context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
    server_context.load_cert_chain(cert_file, key_file)
    server = _serve(server_context)
    yield server.server_address[1], str(ca_file)
    server.shutdown()
    server.server_close()


def test_send_verifies_the_certificate_against_the_host_name(resolver, loopback_allowed, tls):
    port, ca_file = tls

    assert _send_sync(POLICY, f"https://public.test:{port}/ok", verify=ca_file).text == "hello"
    with pytest.raises(requests.exceptions.SSLError):
        _send_sync(POLICY, f"https://other.test:{port}/ok", verify=ca_file)


@pytest.mark.asyncio
async def test_send_async_verifies_the_certificate_against_the_host_name(resolver, loopback_allowed, tls, monkeypatch):
    port, ca_file = tls
    client_context = ssl.create_default_context(cafile=ca_file)
    monkeypatch.setattr(egress, "_ssl_context", lambda: client_context)

    assert (await _send_async(POLICY, f"https://public.test:{port}/ok")).text == "hello"
    with pytest.raises(httpx.ConnectError, match="CERTIFICATE_VERIFY_FAILED"):
        await _send_async(POLICY, f"https://other.test:{port}/ok")


def test_send_async_under_uvloop(loopback_allowed, port):
    uvloop = pytest.importorskip("uvloop")

    async def main():
        response = await _send_async(POLICY, f"http://127.0.0.1:{port}/ok")
        with pytest.raises(egress.EgressRefusedError):
            await _send_async(POLICY, "http://169.254.169.254/latest/meta-data/")
        with pytest.raises(egress.EgressRefusedError):
            await _send_async(POLICY, "http://10.0.0.1/ok")
        return response

    assert uvloop.run(main()).text == "hello"


@pytest.mark.parametrize(
    "url, exempt",
    [
        ("http://tools.internal/v1/conversations/abc/agent-runs", True),
        ("http://tools.internal:80/v1/conversations/abc", True),
        ("http://TOOLS.internal./v1/conversations/abc/x", True),
        ("http://tools.internal/v1/conversations/abcd", False),
        ("http://tools.internal/v1/conversations/abc/../other", False),
        ("http://tools.internal/v1/conversations/abc/%2e%2e/other", False),
        ("http://tools.internal/v1/conversations/abc/a%2Fb", False),
        ("http://tools.internal:8080/v1/conversations/abc/x", False),
        ("https://tools.internal/v1/conversations/abc/x", False),
        ("http://user@tools.internal/v1/conversations/abc/x", False),
        ("http://tools.internal/v1/other", False),
        ("http://tools.internal:bad/v1/conversations/abc/x", False),
        ("http://tools.internal:0/v1/conversations/abc/x", False),
        ("http://tools.internal/v1/conversations/abc/..;/other/secret", False),
        ("http://tools.internal/v1/conversations/abc/%2e%2e;/other/x", False),
        ("http://tools.internal/v1/conversations/abc/runs;jsessionid=1", False),
        ("http://tools.internal/v1/conversations/abc/%252e%252e/other", False),
        ("http://tools.internal/v1/conversations/abc/..%20/other", False),
        ("http://tools.internal/v1/conversations/abc/a%00b", False),
        ("http://tools.internal/v1/conversations/abc/a%5cb", False),
    ],
)
def test_egress_policy_is_exempt(url, exempt):
    policy = EgressPolicy(exempt_url_prefixes=("http://tools.internal/v1/conversations/abc/",))

    assert policy.is_exempt(url) is exempt


@pytest.mark.parametrize(
    "prefix",
    [
        "ftp://tools.internal/",
        "http:///v1/",
        "http://u:p@tools.internal/",
        "http://tools.internal/a/../b/",
        "http://tools.internal/a;b/",
    ],
)
def test_egress_policy_refuses_bad_prefixes(prefix):
    with pytest.raises(ValueError):
        EgressPolicy(exempt_url_prefixes=(prefix,))


def test_runnable_config_leaves_egress_out_of_checkpoints():
    config = RunnableConfig(egress=EgressPolicy())

    assert "egress" not in config.to_checkpoint_dict()


@pytest.fixture
def default_policy():
    set_default_egress_policy(POLICY)
    yield
    set_default_egress_policy(None)


def _node(url: str) -> HttpApiCall:
    return HttpApiCall(
        connection=connections.Http(method=connections.HTTPMethod.GET, url=url),
        response_type=ResponseType.TEXT,
        timeout=30,
    )


def test_http_api_call_refuses_private_address_under_run_policy(resolver, port):
    node = _node(f"http://metadata.test:{port}/ok")

    with pytest.raises(ToolExecutionException, match="Request refused") as error:
        node.execute(HttpApiCallInputSchema(), config=RunnableConfig(egress=POLICY))
    assert error.value.retryable is False


def test_http_api_call_refuses_private_input_url_under_default_policy(default_policy, resolver, loopback_allowed, port):
    node = _node(f"http://public.test:{port}/ok")

    with pytest.raises(ToolExecutionException, match="Request refused"):
        node.execute(HttpApiCallInputSchema(url=f"http://private.test:{port}/ok"))


@pytest.mark.asyncio
async def test_http_api_call_execute_async_refuses_private_address_under_default_policy(default_policy, resolver, port):
    node = _node(f"http://metadata.test:{port}/ok")

    with pytest.raises(ToolExecutionException, match="Request refused"):
        await node.execute_async(HttpApiCallInputSchema())


def test_http_api_call_reaches_public_address_under_default_policy(default_policy, resolver, loopback_allowed, port):
    result = _node(f"http://public.test:{port}/ok").execute(HttpApiCallInputSchema())

    assert result == {"content": "hello", "status_code": 200}


@pytest.mark.asyncio
async def test_http_api_call_execute_async_reaches_public_address_under_default_policy(
    default_policy, resolver, loopback_allowed, port
):
    result = await _node(f"http://public.test:{port}/ok").execute_async(HttpApiCallInputSchema())

    assert result == {"content": "hello", "status_code": 200}


def test_http_api_call_quotes_part_of_the_error_body_under_policy(default_policy, resolver, loopback_allowed, port):
    node = _node(f"http://public.test:{port}/error")

    with pytest.raises(ToolExecutionException) as error:
        node.execute(HttpApiCallInputSchema())
    assert "e" * EGRESS_ERROR_BODY_CHARS in str(error.value)
    assert "e" * (EGRESS_ERROR_BODY_CHARS + 1) not in str(error.value)


def test_http_api_call_without_policy_connects_anywhere(port):
    result = _node(f"http://127.0.0.1:{port}/ok").execute(HttpApiCallInputSchema())

    assert result == {"content": "hello", "status_code": 200}


def _within(start: float, seconds: float) -> bool:
    return time.monotonic() - start < seconds


@pytest.mark.parametrize(
    "path",
    [
        "/slow-redirect?to=http://slow.test:{port}/drip",
        "/redirect?to=http://slow.test:{port}/drip",
    ],
)
def test_send_deadline_covers_lookups_and_later_connections(resolver, loopback_allowed, port, path):
    policy = POLICY.model_copy(update={"timeout_seconds": 1})
    start = time.monotonic()

    with pytest.raises(egress.EgressLimitError, match="took longer"):
        _send_sync(policy, f"http://public.test:{port}{path.format(port=port)}")
    assert _within(start, 2)


def test_send_deadline_covers_a_slow_first_lookup(resolver, loopback_allowed, port):
    policy = POLICY.model_copy(update={"timeout_seconds": 1})
    start = time.monotonic()

    with pytest.raises(egress.EgressLimitError, match="took longer"):
        _send_sync(policy, f"http://slow.test:{port}/drip")
    assert _within(start, 2)


@pytest.mark.parametrize("exempt", [False, True])
def test_send_drops_a_connection_that_opens_after_the_deadline(resolver, loopback_allowed, port, monkeypatch, exempt):
    real_create_connection = egress.urllib3_connection.create_connection

    def slow_create_connection(*args, **kwargs):
        time.sleep(1.5)
        return real_create_connection(*args, **kwargs)

    monkeypatch.setattr(egress.urllib3_connection, "create_connection", slow_create_connection)
    host = "internal.test" if exempt else "public.test"
    policy = POLICY.model_copy(
        update={"timeout_seconds": 1, "exempt_url_prefixes": (f"http://internal.test:{port}/",) if exempt else ()}
    )
    start = time.monotonic()

    with pytest.raises(egress.EgressLimitError, match="took longer"):
        _send_sync(policy, f"http://{host}:{port}/drip")
    assert _within(start, 3)


@pytest.fixture
def tls_handshake_drip():
    """A server that starts a TLS handshake record and then sends one byte of it every 0.1 s for 10 s."""
    listener = socket.create_server(("127.0.0.1", 0))
    stop = threading.Event()

    def serve(conn):
        with conn:
            try:
                conn.sendall(b"\x16\x03\x03\x40\x00")
                for _ in range(100):
                    if stop.wait(0.1):
                        return
                    conn.sendall(b"\0")
            except OSError:
                pass

    def accept():
        while not stop.is_set():
            try:
                conn, _ = listener.accept()
            except OSError:
                return
            threading.Thread(target=serve, args=(conn,), daemon=True).start()

    threading.Thread(target=accept, daemon=True).start()
    yield listener.getsockname()[1]
    stop.set()
    listener.close()


def test_send_deadline_covers_the_tls_handshake(resolver, loopback_allowed, tls_handshake_drip):
    policy = POLICY.model_copy(update={"timeout_seconds": 1})
    start = time.monotonic()

    with pytest.raises(egress.EgressLimitError, match="took longer"):
        _send_sync(policy, f"https://public.test:{tls_handshake_drip}/ok")
    assert _within(start, 3)


@pytest.mark.asyncio
async def test_send_async_deadline_covers_the_tls_handshake(resolver, loopback_allowed, tls_handshake_drip):
    policy = POLICY.model_copy(update={"timeout_seconds": 1})
    start = time.monotonic()

    with pytest.raises(egress.EgressLimitError, match="took longer"):
        await _send_async(policy, f"https://public.test:{tls_handshake_drip}/ok")
    assert _within(start, 3)


@pytest.mark.asyncio
async def test_send_async_deadline_covers_a_slow_lookup(resolver, loopback_allowed, port):
    policy = POLICY.model_copy(update={"timeout_seconds": 1})
    start = time.monotonic()

    with pytest.raises(egress.EgressLimitError, match="took longer"):
        await _send_async(policy, f"http://public.test:{port}/redirect?to=http://slow.test:{port}/ok")
    assert _within(start, 2)


@pytest.mark.asyncio
@pytest.mark.parametrize("use_async", [False, True])
@pytest.mark.parametrize("from_checkpoint", [False, True])
async def test_http_api_call_sends_a_refused_request_once(monkeypatch, use_async, from_checkpoint):
    default_policy = EgressPolicy(max_redirects=1)
    run_policy = EgressPolicy(max_redirects=2)
    set_default_egress_policy(default_policy)
    try:
        config = RunnableConfig(egress=run_policy)
        if from_checkpoint:
            config = RunnableConfig.model_validate(config.to_checkpoint_dict())
        send = (AsyncMock if use_async else Mock)(side_effect=egress.EgressRefusedError("Refused."))
        monkeypatch.setattr(egress, "send_async" if use_async else "send", send)
        node = HttpApiCall(
            connection=connections.Http(method=connections.HTTPMethod.GET, url="https://example.test/resource"),
            error_handling=ErrorHandling(max_retries=2, retry_interval_seconds=0),
        )
        input_data = {"egress": {"max_redirects": 9}}
        if use_async:
            result = await node.run_async(input_data, config=config)
        else:
            result = await asyncio.to_thread(node.run, input_data, config=config)
    finally:
        set_default_egress_policy(None)

    assert result.status == RunnableStatus.FAILURE
    assert "Request refused" in result.error.message
    assert send.call_count == 1
    assert send.call_args.args[0] is (default_policy if from_checkpoint else run_policy)


@pytest.mark.parametrize("url", ["ftp://example.test/resource", "example.test/resource", "http://"])
def test_send_refuses_other_schemes(url):
    with pytest.raises(egress.EgressRefusedError, match="Only http and https"):
        egress.send(POLICY, method="GET", url=url)


@pytest.mark.asyncio
@pytest.mark.parametrize("url", ["ftp://example.test/resource", "example.test/resource"])
async def test_send_async_refuses_other_schemes(url):
    with pytest.raises(egress.EgressRefusedError, match="Only http and https"):
        await egress.send_async(POLICY, method="GET", url=url)


def test_send_reports_a_body_that_stops_at_the_deadline_as_the_deadline(resolver, loopback_allowed, port):
    policy = POLICY.model_copy(update={"timeout_seconds": 0.3})

    for _ in range(10):
        with pytest.raises(egress.EgressLimitError, match="took longer"):
            _send_sync(policy, f"http://public.test:{port}/silent-body")


def test_http_api_call_sends_a_request_that_ran_out_of_time_once(resolver, loopback_allowed, port):
    Handler.hits = 0
    node = HttpApiCall(
        connection=connections.Http(method=connections.HTTPMethod.GET, url=f"http://public.test:{port}/silent-body"),
        error_handling=ErrorHandling(max_retries=2, retry_interval_seconds=0),
    )

    result = node.run({}, config=RunnableConfig(egress=POLICY.model_copy(update={"timeout_seconds": 0.3})))

    assert result.status == RunnableStatus.FAILURE
    assert Handler.hits == 1


def test_send_reads_chunked_bodies(resolver, loopback_allowed, port):
    assert _send_sync(POLICY, f"http://public.test:{port}/chunked").content == b"ab" * 1000


@pytest.mark.asyncio
async def test_send_async_reads_chunked_bodies(resolver, loopback_allowed, port):
    assert (await _send_async(POLICY, f"http://public.test:{port}/chunked")).content == b"ab" * 1000


def test_send_decodes_every_gzip_member(resolver, loopback_allowed, port):
    assert _send_sync(POLICY, f"http://public.test:{port}/gzip-members").content == b"hello world"


@pytest.mark.asyncio
async def test_send_async_decodes_every_gzip_member(resolver, loopback_allowed, port):
    assert (await _send_async(POLICY, f"http://public.test:{port}/gzip-members")).content == b"hello world"


def test_send_accepts_an_empty_body_in_any_encoding(resolver, loopback_allowed, port):
    assert _send_sync(POLICY, f"http://public.test:{port}/empty-brotli").status_code == 204


@pytest.mark.asyncio
async def test_send_async_accepts_an_empty_body_in_any_encoding(resolver, loopback_allowed, port):
    assert (await _send_async(POLICY, f"http://public.test:{port}/empty-brotli")).status_code == 204


@pytest.fixture
def busy_resolver(monkeypatch):
    """A resolver pool whose only thread is taken by a lookup that does not end until the test does."""
    pool = ThreadPoolExecutor(max_workers=1)
    release = threading.Event()
    pool.submit(release.wait)
    monkeypatch.setattr(egress, "_resolver", pool)
    yield
    release.set()
    pool.shutdown()


def test_send_needs_no_lookup_for_an_address(loopback_allowed, port, busy_resolver):
    assert _send_sync(POLICY, f"http://127.0.0.1:{port}/ok").text == "hello"


@pytest.mark.asyncio
async def test_send_async_needs_no_lookup_for_an_address(loopback_allowed, port, busy_resolver):
    assert (await _send_async(POLICY, f"http://127.0.0.1:{port}/ok")).text == "hello"


def _raw_deflate(data: bytes) -> bytes:
    compressor = zlib.compressobj(wbits=-zlib.MAX_WBITS)
    return compressor.compress(data) + compressor.flush()


def _decode(content_encoding: str, chunks: list[bytes]) -> bytes:
    decoder = egress._BoundedDecoder(content_encoding, egress._Call(EgressPolicy()))
    body = b"".join(decoder.decode(chunk) for chunk in chunks)
    decoder.flush()
    return body


@pytest.mark.parametrize(
    "content_encoding, body, decoded",
    [
        ("gzip", gzip.compress(b"hello ", mtime=0) + gzip.compress(b"world", mtime=0), b"hello world"),
        ("gzip", gzip.compress(b"hello", mtime=0) + b"\x1ftrailing", b"hello"),
        ("gzip", gzip.compress(b"hello", mtime=0) + b"trailing", b"hello"),
        ("deflate", zlib.compress(b"hello"), b"hello"),
        ("deflate", zlib.compress(b"hello") + b"trailing", b"hello"),
        ("deflate", _raw_deflate(b"hello"), b"hello"),
    ],
    ids=["gzip members", "gzip, then a magic byte", "gzip, then data", "zlib", "zlib, then data", "raw deflate"],
)
def test_bounded_decoder_decodes_a_body_split_anywhere(content_encoding, body, decoded):
    for split in range(len(body) + 1):
        assert _decode(content_encoding, [body[:split], body[split:]]) == decoded, split
    assert _decode(content_encoding, [bytes([byte]) for byte in body]) == decoded


@pytest.mark.parametrize(
    "headers",
    [
        {"Host": "admin.internal"},
        {"Host": "internal.test:1"},
        {"Host": "internal.test@admin.internal"},
        {"X-Forwarded-Host": "admin.internal"},
        {"X-Original-URL": "/admin"},
        {"Forwarded": "host=admin.internal"},
    ],
)
def test_send_refuses_an_exempt_request_that_routes_elsewhere(resolver, port, headers):
    policy = POLICY.model_copy(update={"exempt_url_prefixes": (f"http://internal.test:{port}/exempt/",)})

    with pytest.raises(egress.EgressRefusedError):
        _send_sync(policy, f"http://internal.test:{port}/exempt/ok", headers=headers)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "headers",
    [
        {"Host": "admin.internal"},
        {"Host": "internal.test:1"},
        {"X-Forwarded-Host": "admin.internal"},
        {"X-Rewrite-URL": "/admin"},
    ],
)
async def test_send_async_refuses_an_exempt_request_that_routes_elsewhere(resolver, port, headers):
    policy = POLICY.model_copy(update={"exempt_url_prefixes": (f"http://internal.test:{port}/exempt/",)})

    with pytest.raises(egress.EgressRefusedError):
        await _send_async(policy, f"http://internal.test:{port}/exempt/ok", headers=headers)


def test_send_accepts_an_exempt_request_that_names_its_own_host(resolver, port):
    policy = POLICY.model_copy(update={"exempt_url_prefixes": (f"http://internal.test:{port}/exempt/",)})

    response = _send_sync(policy, f"http://internal.test:{port}/exempt/ok", headers={"Host": f"INTERNAL.test:{port}"})

    assert response.text == "hello"


@pytest.mark.asyncio
async def test_send_async_accepts_an_exempt_request_that_names_its_own_host(resolver, port):
    policy = POLICY.model_copy(update={"exempt_url_prefixes": (f"http://internal.test:{port}/exempt/",)})

    response = await _send_async(
        policy, f"http://internal.test:{port}/exempt/ok", headers={"Host": f"internal.test:{port}"}
    )

    assert response.text == "hello"


@pytest.mark.parametrize(
    "path",
    [
        "/redirect?to=http://internal.test:{port}/exempt/ok",
        "/exempt/redirect?to=http://internal.test:{port}/exempt/ok",
    ],
    ids=["from a public URL", "from an exempt URL"],
)
def test_send_checks_a_redirect_to_an_exempt_url_as_any_other(resolver, port, path):
    policy = POLICY.model_copy(update={"exempt_url_prefixes": (f"http://internal.test:{port}/exempt/",)})
    host = "internal.test" if path.startswith("/exempt") else "public.test"

    with pytest.raises(egress.EgressRefusedError):
        _send_sync(policy, f"http://{host}:{port}{path.format(port=port)}")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "path",
    [
        "/redirect?to=http://internal.test:{port}/exempt/ok",
        "/exempt/redirect?to=http://internal.test:{port}/exempt/ok",
    ],
    ids=["from a public URL", "from an exempt URL"],
)
async def test_send_async_checks_a_redirect_to_an_exempt_url_as_any_other(resolver, port, path):
    policy = POLICY.model_copy(update={"exempt_url_prefixes": (f"http://internal.test:{port}/exempt/",)})
    host = "internal.test" if path.startswith("/exempt") else "public.test"

    with pytest.raises(egress.EgressRefusedError):
        await _send_async(policy, f"http://{host}:{port}{path.format(port=port)}")


def test_send_lets_a_public_request_name_any_host(resolver, loopback_allowed, port):
    assert _send_sync(POLICY, f"http://public.test:{port}/ok", headers={"Host": "virtual.example"}).text == "hello"


@pytest.mark.parametrize(
    "content_encoding, body",
    [("gzip", gzip.compress(b"hello world", mtime=0)[:-2]), ("deflate", zlib.compress(b"hello world")[:-2])],
    ids=["gzip", "zlib"],
)
def test_bounded_decoder_refuses_a_body_that_ends_inside_its_compressed_data(content_encoding, body):
    for chunks in ([body], [bytes([byte]) for byte in body]):
        with pytest.raises(egress.EgressLimitError, match="ended before"):
            _decode(content_encoding, chunks)


def test_send_deadline_covers_the_lookup_of_an_exempt_host(resolver, port):
    policy = POLICY.model_copy(
        update={"timeout_seconds": 0.5, "exempt_url_prefixes": (f"http://slow.test:{port}/exempt/",)}
    )
    start = time.monotonic()

    with pytest.raises(egress.EgressLimitError, match="took longer"):
        _send_sync(policy, f"http://slow.test:{port}/exempt/ok")
    assert _within(start, 1.5)


@pytest.mark.asyncio
async def test_send_async_checks_the_deadline_after_the_client_returns(monkeypatch):
    clock = SimpleNamespace(value=0.0)
    monkeypatch.setattr(egress, "time", SimpleNamespace(monotonic=lambda: clock.value))

    async def request_that_ends_after_the_deadline(self, **kwargs):
        clock.value += 1
        return httpx.Response(200, content=b"hello")

    monkeypatch.setattr(httpx.AsyncClient, "request", request_that_ends_after_the_deadline)

    with pytest.raises(egress.EgressLimitError, match="took longer"):
        await egress.send_async(EgressPolicy(timeout_seconds=0.1), method="GET", url="http://public.test/")


@pytest.fixture
def full_resolver(monkeypatch):
    """A resolver that admits no more lookups."""
    slots = threading.BoundedSemaphore(1)
    slots.acquire()
    monkeypatch.setattr(egress, "_resolver_slots", slots)


def test_send_refuses_a_lookup_beyond_the_resolver_capacity(loopback_allowed, port, full_resolver):
    with pytest.raises(egress.EgressLimitError, match="Too many host name lookups"):
        _send_sync(POLICY, f"http://localhost:{port}/ok")
    assert _send_sync(POLICY, f"http://127.0.0.1:{port}/ok").text == "hello"


@pytest.mark.asyncio
async def test_send_async_refuses_a_lookup_beyond_the_resolver_capacity(loopback_allowed, port, full_resolver):
    with pytest.raises(egress.EgressLimitError, match="Too many host name lookups"):
        await _send_async(POLICY, f"http://localhost:{port}/ok")


def test_resolver_capacity_is_released_when_a_lookup_ends(resolver):
    for _ in range(egress.RESOLVER_CAPACITY + 1):
        with pytest.raises(egress.EgressRefusedError):
            egress.resolve_public("private.test", 80, timeout=5)


@pytest.mark.asyncio
@pytest.mark.parametrize("use_async", [False, True])
async def test_http_api_call_refusal_names_no_address(use_async):
    node = _node("http://10.1.2.3/")
    config = RunnableConfig(egress=POLICY)

    result = await node.run_async(input_data={}, config=config) if use_async else node.run_sync({}, config=config)

    assert "Request refused" in result.error.message
    assert "10.1.2.3" not in result.error.message
