"""Tool bridge broker, run inside the sandbox by the agent process for one shell command.

Its standard streams are its only channel to the agent process. Code in the sandbox reaches it on a Unix socket
(``dynamiq_tools``), and the template's CLIs (``tools-cli``, ``sql-cli``) on a local HTTP port that answers the
platform routes they call. Each request goes out as one line on stdout, framed with the command's nonce; the reply
comes back on stdin as JSON lines ``{"id", "chunk", "last"}``, whose chunks joined make the reply. Every request
carries the token from ``DYNAMIQ_TOOLS_TOKEN``. The broker exits when its stdin closes.

Standard library only: it runs on whatever Python the sandbox template has.
"""

import base64
import json
import os
import socket
import sys
import threading
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

DEFAULT_TIMEOUT = 300
MAX_TIMEOUT = 3600

_pending: dict[str, list] = {}  # request id -> [Event, reply chunks, reply]
_out_lock = threading.Lock()


def emit(nonce: str, kind: str, body: dict) -> None:
    frame = base64.b64encode(json.dumps({"kind": kind, **body}).encode()).decode()
    with _out_lock:
        os.write(1, f"\x1eDQB:{nonce}:{frame}\n".encode())


def read_replies() -> None:
    for line in sys.stdin:
        try:
            part = json.loads(line)
            slot = _pending.get(part["id"])
            if slot is None:
                continue
            slot[1].append(part["chunk"])
            if part.get("last", True):
                slot[2] = json.loads(base64.b64decode("".join(slot[1])))
                slot[0].set()
        except (ValueError, KeyError, TypeError) as e:
            print(f"broker: bad reply line: {type(e).__name__}", file=sys.stderr, flush=True)
    os._exit(0)


def relay(nonce: str, request: dict, timeout: float) -> dict:
    rid = uuid.uuid4().hex
    slot = _pending[rid] = [threading.Event(), [], None]
    try:
        emit(nonce, "req", {"id": rid, **request})
        if not slot[0].wait(timeout):
            return {"ok": False, "error": "BridgeTimeout", "reason": f"no reply within {timeout:g} seconds"}
        return slot[2]
    finally:
        _pending.pop(rid, None)


def clamp_timeout(value) -> float:
    try:
        return max(1.0, min(float(value), MAX_TIMEOUT))
    except (TypeError, ValueError):
        return DEFAULT_TIMEOUT


def answer(f, reply: dict) -> None:
    f.write((json.dumps(reply) + "\n").encode())


def serve_socket(conn: socket.socket, nonce: str, token: str, max_bytes: int) -> None:
    with conn, conn.makefile("rwb") as f:
        line = f.readline(max_bytes + 1)
        if len(line) > max_bytes:
            return answer(f, {"ok": False, "error": "RequestTooLarge"})
        try:
            request = json.loads(line)
        except ValueError:
            return answer(f, {"ok": False, "error": "BadRequest"})
        if not isinstance(request, dict) or request.pop("token", None) != token:
            return answer(f, {"ok": False, "error": "BadToken"})
        if request.get("op") not in ("list", "call"):
            return answer(f, {"ok": False, "error": "UnknownOperation"})
        timeout = clamp_timeout(request.pop("timeout", DEFAULT_TIMEOUT))
        relayed = {"op": request["op"], "tool": request.get("tool"), "input": request.get("input")}
        answer(f, relay(nonce, relayed, timeout))


def make_http_handler(nonce: str, token: str, max_bytes: int):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):  # stdout carries frames only
            pass

        def _answer(self, status: int, body) -> None:
            data = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _relay(self) -> None:
            # A browser page that rebinds its own name to 127.0.0.1 sends a Host header of that name.
            host = (self.headers.get("Host") or "").rsplit(":", 1)[0]
            if host not in ("127.0.0.1", "localhost"):
                return self._answer(403, {"error": {"code": "forbidden", "message": "Bad Host header."}})
            if self.headers.get("Authorization", "") != f"Bearer {token}":
                return self._answer(401, {"error": {"code": "unauthorized", "message": "Bad token."}})
            length = int(self.headers.get("Content-Length") or 0)
            if length > max_bytes:
                return self._answer(413, {"error": {"code": "too_large", "message": "Request too large."}})
            try:
                body = json.loads(self.rfile.read(length)) if length else None
            except ValueError:
                return self._answer(400, {"error": {"code": "bad_request", "message": "The body is not JSON."}})
            request = {"op": "http", "method": self.command, "path": urlsplit(self.path).path, "body": body}
            reply = relay(nonce, request, DEFAULT_TIMEOUT)
            if reply.get("ok") and "status" in reply:
                return self._answer(int(reply["status"]), reply.get("body"))
            error = reply.get("error", "BridgeError")
            return self._answer(502, {"error": {"code": error, "message": reply.get("reason") or error}})

        do_GET = do_POST = _relay

    return Handler


def main() -> None:
    sock_path, nonce, max_bytes = sys.argv[1], sys.argv[2], int(sys.argv[3])
    token = os.environ["DYNAMIQ_TOOLS_TOKEN"]
    if os.path.exists(sock_path):
        os.unlink(sock_path)
    server = socket.socket(socket.AF_UNIX)
    server.bind(sock_path)
    os.chmod(sock_path, 0o600)
    server.listen(64)
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), make_http_handler(nonce, token, max_bytes))
    httpd.daemon_threads = True
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    threading.Thread(target=read_replies, daemon=True).start()
    emit(nonce, "ready", {"http_port": httpd.server_address[1]})
    while True:
        conn, _ = server.accept()
        threading.Thread(target=serve_socket, args=(conn, nonce, token, max_bytes), daemon=True).start()


if __name__ == "__main__":
    main()
