"""Unit tests for the Cloudflare sandbox bridge HTTP client (no network).

Uses the ``requests_mock`` fixture provided by the requests-mock pytest plugin.
"""

import base64
import json

import pytest
import requests

from dynamiq.connections.cloudflare_sandbox import (
    CloudflareRateLimitError,
    CloudflareSandboxClient,
    CloudflareSandboxError,
    CloudflareSandboxInstance,
    resolve_workspace_path,
)

BASE_URL = "https://bridge.example.workers.dev"
SANDBOX_ID = "abcdefgh234567"


@pytest.fixture
def client():
    return CloudflareSandboxClient(url=BASE_URL, api_key="test-key")


def sse_body(events: list[tuple[str, str]]) -> str:
    """Build an SSE body the way the bridge does (event + data lines)."""
    return "".join(f"event: {event}\ndata: {data}\n\n" for event, data in events)


def b64(text: str) -> str:
    return base64.b64encode(text.encode()).decode()


class TestResolveWorkspacePath:
    def test_absolute_path_inside_workspace(self):
        assert resolve_workspace_path("/workspace/output/x.txt") == "/workspace/output/x.txt"

    def test_relative_path_is_anchored_to_workspace(self):
        assert resolve_workspace_path("output/x.txt") == "/workspace/output/x.txt"

    def test_workspace_root_allowed(self):
        assert resolve_workspace_path("/workspace") == "/workspace"

    def test_dot_segments_normalized(self):
        assert resolve_workspace_path("/workspace/a/../b/./c.txt") == "/workspace/b/c.txt"

    @pytest.mark.parametrize("path", ["/etc/passwd", "/workspace/../etc/passwd", "/home/user/x.txt"])
    def test_paths_escaping_workspace_rejected(self, path):
        with pytest.raises(ValueError, match="must stay inside /workspace"):
            resolve_workspace_path(path)


class TestClientRequests:
    def test_create_sandbox(self, client, requests_mock):
        requests_mock.post(f"{BASE_URL}/v1/sandbox", json={"id": SANDBOX_ID})

        assert client.create_sandbox() == SANDBOX_ID
        assert requests_mock.last_request.headers["Authorization"] == "Bearer test-key"

    def test_destroy_sandbox(self, client, requests_mock):
        requests_mock.delete(f"{BASE_URL}/v1/sandbox/{SANDBOX_ID}", status_code=204)

        client.destroy_sandbox(SANDBOX_ID)
        assert requests_mock.called

    @pytest.mark.parametrize("bad_id", ["NOT/VALID", "abcd\n", "", "abc1"])
    def test_invalid_sandbox_id_rejected_locally(self, client, bad_id):
        with pytest.raises(ValueError, match="Invalid Cloudflare sandbox id"):
            client.destroy_sandbox(bad_id)

    def test_is_running(self, client, requests_mock):
        requests_mock.get(f"{BASE_URL}/v1/sandbox/{SANDBOX_ID}/running", json={"running": True})

        assert client.is_running(SANDBOX_ID) is True

    def test_read_file(self, client, requests_mock):
        requests_mock.get(
            f"{BASE_URL}/v1/sandbox/{SANDBOX_ID}/file/workspace/output/data.bin", content=b"\x00\x01binary"
        )

        assert client.read_file(SANDBOX_ID, "/workspace/output/data.bin") == b"\x00\x01binary"

    def test_write_file(self, client, requests_mock):
        requests_mock.put(f"{BASE_URL}/v1/sandbox/{SANDBOX_ID}/file/workspace/input/a.txt", json={"ok": True})

        client.write_file(SANDBOX_ID, "/workspace/input/a.txt", b"hello")
        assert requests_mock.last_request.body == b"hello"

    def test_file_path_outside_workspace_rejected(self, client):
        with pytest.raises(ValueError, match="must stay inside /workspace"):
            client.read_file(SANDBOX_ID, "/etc/passwd")

    def test_file_url_keeps_uri_reserved_characters_raw(self, client, requests_mock):
        """decodeURI on the bridge never decodes : @ & = + $ , ; - they must be sent raw."""
        requests_mock.get(
            f"{BASE_URL}/v1/sandbox/{SANDBOX_ID}/file/workspace/output/report:v2,final+1.csv", content=b"x"
        )

        client.read_file(SANDBOX_ID, "/workspace/output/report:v2,final+1.csv")

        assert (
            requests_mock.last_request.path == f"/v1/sandbox/{SANDBOX_ID}/file/workspace/output/report:v2,final+1.csv"
        )

    def test_file_url_percent_encodes_unsafe_characters(self, client, requests_mock):
        """Characters like spaces and # must be percent-encoded so the URL stays intact."""
        requests_mock.get(f"{BASE_URL}/v1/sandbox/{SANDBOX_ID}/file/workspace/output/report%20%231.csv", content=b"x")

        client.read_file(SANDBOX_ID, "/workspace/output/report #1.csv")

        assert requests_mock.last_request.url.endswith("/file/workspace/output/report%20%231.csv")

    def test_health(self, client, requests_mock):
        requests_mock.get(f"{BASE_URL}/health", json={"ok": True})

        assert client.health() is True

    def test_health_false_on_http_error(self, client, requests_mock):
        requests_mock.get(f"{BASE_URL}/health", status_code=500, text="boom")

        assert client.health() is False

    def test_health_false_on_not_ok_body(self, client, requests_mock):
        requests_mock.get(f"{BASE_URL}/health", json={"ok": False, "errors": ["missing binding"]})

        assert client.health() is False

    def test_health_false_on_connection_error(self, client, requests_mock):
        requests_mock.get(f"{BASE_URL}/health", exc=requests.exceptions.ConnectTimeout)

        assert client.health() is False

    def test_create_session(self, client, requests_mock):
        requests_mock.post(f"{BASE_URL}/v1/sandbox/{SANDBOX_ID}/session", json={"id": "sess1"})

        assert client.create_session(SANDBOX_ID, cwd="/workspace", env={"A": "b"}) == "sess1"
        assert requests_mock.last_request.json() == {"cwd": "/workspace", "env": {"A": "b"}}

    def test_create_tunnel(self, client, requests_mock):
        tunnel = {"id": "t1", "port": 8000, "url": "https://x.trycloudflare.com", "hostname": "x.trycloudflare.com"}
        requests_mock.post(f"{BASE_URL}/v1/sandbox/{SANDBOX_ID}/tunnel/8000", json=tunnel)

        assert client.create_tunnel(SANDBOX_ID, 8000) == tunnel


class TestExec:
    def exec_url(self):
        return f"{BASE_URL}/v1/sandbox/{SANDBOX_ID}/exec"

    def test_exec_parses_sse_stream(self, client, requests_mock):
        body = sse_body(
            [
                ("stdout", b64("hello ")),
                ("stdout", b64("world\n")),
                ("stderr", b64("warning\n")),
                ("exit", json.dumps({"exit_code": 0})),
            ]
        )
        requests_mock.post(self.exec_url(), text=body, headers={"Content-Type": "text/event-stream"})

        result = client.exec(SANDBOX_ID, "echo hello world")

        assert result.exit_code == 0
        assert result.stdout == "hello world\n"
        assert result.stderr == "warning\n"

    def test_exec_builds_bash_argv_for_string_command(self, client, requests_mock):
        requests_mock.post(self.exec_url(), text=sse_body([("exit", json.dumps({"exit_code": 0}))]))

        client.exec(SANDBOX_ID, "echo hi", cwd="/workspace/output", timeout=30)

        request_body = requests_mock.last_request.json()
        assert request_body["argv"] == ["bash", "-c", "echo hi"]
        assert request_body["timeout_ms"] == 30000
        assert request_body["cwd"] == "/workspace/output"

    def test_exec_read_timeout_headroom(self, client, requests_mock):
        """The socket read timeout must exceed the command timeout so long commands are not cut off."""
        requests_mock.post(self.exec_url(), text=sse_body([("exit", json.dumps({"exit_code": 0}))]))

        client.exec(SANDBOX_ID, "sleep 500", timeout=600)

        assert requests_mock.last_request.timeout == (30.0, 720)

    def test_exec_env_is_prepended_via_env_command(self, client, requests_mock):
        requests_mock.post(self.exec_url(), text=sse_body([("exit", json.dumps({"exit_code": 0}))]))

        client.exec(SANDBOX_ID, "printenv FOO", env={"FOO": "bar baz"})

        assert requests_mock.last_request.json()["argv"] == ["env", "FOO=bar baz", "bash", "-c", "printenv FOO"]

    def test_exec_list_command_passed_as_argv(self, client, requests_mock):
        requests_mock.post(self.exec_url(), text=sse_body([("exit", json.dumps({"exit_code": 0}))]))

        client.exec(SANDBOX_ID, ["python3", "-V"])

        assert requests_mock.last_request.json()["argv"] == ["python3", "-V"]

    def test_exec_session_id_header(self, client, requests_mock):
        requests_mock.post(self.exec_url(), text=sse_body([("exit", json.dumps({"exit_code": 0}))]))

        client.exec(SANDBOX_ID, "true", session_id="sess1")

        assert requests_mock.last_request.headers["Session-Id"] == "sess1"

    def test_exec_nonzero_exit_code(self, client, requests_mock):
        body = sse_body([("stderr", b64("boom")), ("exit", json.dumps({"exit_code": 2}))])
        requests_mock.post(self.exec_url(), text=body)

        result = client.exec(SANDBOX_ID, "exit 2")

        assert result.exit_code == 2
        assert result.stderr == "boom"

    def test_exec_error_event_raises(self, client, requests_mock):
        body = sse_body([("error", json.dumps({"error": "exec failed: kaput", "code": "exec_transport_error"}))])
        requests_mock.post(self.exec_url(), text=body)

        with pytest.raises(CloudflareSandboxError, match="kaput"):
            client.exec(SANDBOX_ID, "true")

    def test_exec_error_event_mid_stream_raises(self, client, requests_mock):
        """A transport error arriving after some stdout must still raise."""
        body = sse_body(
            [
                ("stdout", b64("partial output")),
                ("error", json.dumps({"error": "container evicted", "code": "exec_transport_error"})),
            ]
        )
        requests_mock.post(self.exec_url(), text=body)

        with pytest.raises(CloudflareSandboxError, match="container evicted"):
            client.exec(SANDBOX_ID, "true")

    def test_exec_stream_without_trailing_blank_line(self, client, requests_mock):
        """The final event must be flushed even when the stream ends without a blank line."""
        body = sse_body([("stdout", b64("done\n")), ("exit", json.dumps({"exit_code": 0}))]).rstrip("\n")
        requests_mock.post(self.exec_url(), text=body)

        result = client.exec(SANDBOX_ID, "echo done")

        assert result.exit_code == 0
        assert result.stdout == "done\n"

    def test_exec_multibyte_character_split_across_chunks(self, client, requests_mock):
        """UTF-8 characters split across SSE chunk boundaries must survive intact."""
        encoded = "héllo".encode()
        first, second = encoded[:2], encoded[2:]  # split mid 'é' (0xC3 | 0xA9)
        body = sse_body(
            [
                ("stdout", base64.b64encode(first).decode()),
                ("stdout", base64.b64encode(second).decode()),
                ("exit", json.dumps({"exit_code": 0})),
            ]
        )
        requests_mock.post(self.exec_url(), text=body)

        result = client.exec(SANDBOX_ID, "echo héllo")

        assert result.stdout == "héllo"

    def test_exec_invalid_base64_falls_back_to_raw_data(self, client, requests_mock):
        body = sse_body([("stdout", "!!not-base64!!"), ("exit", json.dumps({"exit_code": 0}))])
        requests_mock.post(self.exec_url(), text=body)

        result = client.exec(SANDBOX_ID, "true")

        assert result.stdout == "!!not-base64!!"

    def test_exec_missing_exit_event_yields_none_exit_code(self, client, requests_mock):
        """A stream that dies before the exit event must not report success."""
        body = sse_body([("stdout", b64("partial"))])
        requests_mock.post(self.exec_url(), text=body)

        result = client.exec(SANDBOX_ID, "true")

        assert result.exit_code is None
        assert result.stdout == "partial"

    def test_exec_stream_disconnect_wrapped(self):
        """Connection drops mid-stream surface as CloudflareSandboxError, not raw requests errors."""

        class BrokenStreamResponse:
            encoding = None

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def iter_lines(self, decode_unicode=True):
                yield "event: stdout"
                yield f"data: {b64('partial')}"
                yield ""
                raise requests.exceptions.ChunkedEncodingError("connection broken mid-stream")

        with pytest.raises(CloudflareSandboxError, match="stream failed"):
            CloudflareSandboxClient._parse_exec_sse(BrokenStreamResponse())


class TestErrorMapping:
    def test_http_401_raises_sandbox_error(self, client, requests_mock):
        requests_mock.post(
            f"{BASE_URL}/v1/sandbox", status_code=401, json={"error": "Unauthorized", "code": "unauthorized"}
        )

        with pytest.raises(CloudflareSandboxError, match="Unauthorized") as exc_info:
            client.create_sandbox()
        assert not isinstance(exc_info.value, CloudflareRateLimitError)
        assert exc_info.value.status_code == 401

    def test_http_429_raises_rate_limit_error(self, client, requests_mock):
        requests_mock.post(f"{BASE_URL}/v1/sandbox", status_code=429, json={"error": "slow down"})

        with pytest.raises(CloudflareRateLimitError):
            client.create_sandbox()

    def test_capacity_exceeded_503_raises_rate_limit_error(self, client, requests_mock):
        requests_mock.post(
            f"{BASE_URL}/v1/sandbox",
            status_code=503,
            json={"error": "instance limit reached", "code": "capacity_exceeded"},
        )

        with pytest.raises(CloudflareRateLimitError, match="instance limit reached"):
            client.create_sandbox()

    def test_non_json_error_body(self, client, requests_mock):
        requests_mock.post(f"{BASE_URL}/v1/sandbox", status_code=502, text="Bad Gateway")

        with pytest.raises(CloudflareSandboxError, match="HTTP 502"):
            client.create_sandbox()

    def test_missing_url_raises(self):
        with pytest.raises(ValueError, match="bridge URL is required"):
            CloudflareSandboxClient(url="", api_key="k")


class TestSandboxInstance:
    def test_instance_delegates_to_client(self, client, requests_mock):
        requests_mock.post(
            f"{BASE_URL}/v1/sandbox/{SANDBOX_ID}/exec",
            text=sse_body([("stdout", b64("ok")), ("exit", json.dumps({"exit_code": 0}))]),
        )
        requests_mock.put(f"{BASE_URL}/v1/sandbox/{SANDBOX_ID}/file/workspace/f.txt", json={"ok": True})
        requests_mock.get(f"{BASE_URL}/v1/sandbox/{SANDBOX_ID}/file/workspace/f.txt", content=b"data")
        requests_mock.delete(f"{BASE_URL}/v1/sandbox/{SANDBOX_ID}", status_code=204)

        instance = CloudflareSandboxInstance(client, SANDBOX_ID)

        assert instance.exec("echo ok").stdout == "ok"
        instance.write_file("/workspace/f.txt", b"data")
        assert instance.read_file("/workspace/f.txt") == b"data"
        instance.destroy()
        assert requests_mock.call_count == 4
