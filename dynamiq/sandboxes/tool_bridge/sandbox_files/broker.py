"""Tool bridge broker, run inside the sandbox by the agent process for one shell command.

Its standard streams are its only channel to the agent process. Code in the sandbox reaches it on a Unix socket
(``dynamiq_tools``). Each request goes out as one line on stdout, framed with the command's nonce; the reply comes
back on stdin as JSON lines ``{"id", "chunk", "last"}``, whose chunks joined make the reply. Every request carries
the token from ``DYNAMIQ_TOOLS_TOKEN``. The broker exits when its stdin closes.

Standard library only: it runs on whatever Python the sandbox template has.
"""

import base64
import json
import os
import socket
import sys
import threading
import uuid

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


def main() -> None:
    sock_path, nonce, max_bytes = sys.argv[1], sys.argv[2], int(sys.argv[3])
    token = os.environ["DYNAMIQ_TOOLS_TOKEN"]
    if os.path.exists(sock_path):
        os.unlink(sock_path)
    server = socket.socket(socket.AF_UNIX)
    server.bind(sock_path)
    os.chmod(sock_path, 0o600)
    server.listen(64)
    threading.Thread(target=read_replies, daemon=True).start()
    emit(nonce, "ready", {})
    while True:
        conn, _ = server.accept()
        threading.Thread(target=serve_socket, args=(conn, nonce, token, max_bytes), daemon=True).start()


if __name__ == "__main__":
    main()
