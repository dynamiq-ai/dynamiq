import datetime
import gzip
import ipaddress
import socket
import ssl
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
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
from dynamiq.nodes.tools import HttpApiCall, ResponseType
from dynamiq.nodes.tools.http_api_call import EGRESS_ERROR_BODY_CHARS, HttpApiCallInputSchema
from dynamiq.runnables import RunnableConfig
from dynamiq.types.egress import EgressPolicy, set_default_egress_policy
from dynamiq.utils import egress

MAX_BYTES = 64 * 1024
GZIP_BOMB = gzip.compress(b"\0" * (64 * 1024 * 1024))


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

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
        elif path == "/brotli":
            self._send(200, b"\x0b\x02\x80hello\x03", {"Content-Encoding": "br"})
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
    }
    lookups: list[str] = []

    def getaddrinfo(host, port, family=0, type=0, proto=0, flags=0):
        if isinstance(host, bytes):
            host = host.decode()
        if host not in answers:
            return real_getaddrinfo(host, port, family, type, proto, flags)
        lookups.append(host)
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


@pytest.mark.parametrize("path", ["/gzip-bomb", "/redirect-big?to=/ok"])
def test_send_limits_decoded_and_redirect_bodies(resolver, loopback_allowed, port, path):
    with pytest.raises(egress.EgressLimitError):
        _send_sync(POLICY, f"http://public.test:{port}{path}")


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["/gzip-bomb", "/redirect-big?to=/ok", "/brotli"])
async def test_send_async_limits_decoded_and_redirect_bodies(resolver, loopback_allowed, port, path):
    with pytest.raises(egress.EgressLimitError):
        await _send_async(POLICY, f"http://public.test:{port}{path}")


@pytest.mark.asyncio
async def test_send_async_decodes_gzip(resolver, loopback_allowed, port):
    response = await _send_async(POLICY, f"http://public.test:{port}/gzip")

    assert response.content == b"hello"


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
    with pytest.raises(requests.TooManyRedirects):
        _send_sync(POLICY, f"http://public.test:{port}/loop")


@pytest.mark.asyncio
async def test_send_async_limits_redirects(resolver, loopback_allowed, port):
    with pytest.raises(httpx.TooManyRedirects):
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
    monkeypatch.setattr(egress, "_shared_ssl_context", lambda: client_context)

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
        ("http://catalyst.internal/v1/conversations/abc/agent-runs", True),
        ("http://catalyst.internal:80/v1/conversations/abc", True),
        ("http://CATALYST.internal./v1/conversations/abc/x", True),
        ("http://catalyst.internal/v1/conversations/abcd", False),
        ("http://catalyst.internal/v1/conversations/abc/../other", False),
        ("http://catalyst.internal/v1/conversations/abc/%2e%2e/other", False),
        ("http://catalyst.internal/v1/conversations/abc/a%2Fb", False),
        ("http://catalyst.internal:8080/v1/conversations/abc/x", False),
        ("https://catalyst.internal/v1/conversations/abc/x", False),
        ("http://user@catalyst.internal/v1/conversations/abc/x", False),
        ("http://catalyst.internal/v1/other", False),
        ("http://catalyst.internal:bad/v1/conversations/abc/x", False),
    ],
)
def test_egress_policy_is_exempt(url, exempt):
    policy = EgressPolicy(exempt_url_prefixes=("http://catalyst.internal/v1/conversations/abc/",))

    assert policy.is_exempt(url) is exempt


@pytest.mark.parametrize(
    "prefix",
    ["ftp://catalyst.internal/", "http:///v1/", "http://u:p@catalyst.internal/", "http://catalyst.internal/a/../b/"],
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
