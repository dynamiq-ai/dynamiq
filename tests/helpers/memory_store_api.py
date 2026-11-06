"""A stand-in for the platform's memory store API, for tests.

``MemoryStoreAPISimulator`` stands in for the server, so everything above the socket is real: the same
client, URLs, query strings and status codes.
"""

import json as json_lib
from datetime import datetime, timezone

INVALID_INPUT = "The request could not be processed due to invalid input."


class _Response:
    """The subset of a `requests` response that DynamiqFileStore reads."""

    def __init__(self, status_code: int, payload=None):
        self.status_code = status_code
        self._payload = payload
        self.text = json_lib.dumps(payload) if payload is not None else ""
        self.content = self.text.encode()

    def json(self):
        if self._payload is None:
            raise ValueError("response has no JSON body")
        return self._payload


def _error(status_code: int, code: str, message: str, details: dict | None = None) -> _Response:
    """An error in the platform's shape: {"error": {"code", "message", "details"}}."""
    error = {"code": code, "message": message}
    if details:
        error["details"] = details
    return _Response(status_code, {"error": error})


class MemoryStoreAPISimulator:
    """In-process implementation of the platform's Memory Store API.

    Deliberately strict: it enforces the parts a client is most likely to get wrong, such as the
    user_id on every call, safe paths and the content limit, and answers errors the way the
    platform does, so a client change that breaks the contract fails here rather than in production.
    """

    max_content_length = 8192

    def __init__(self):
        # Keyed by (store_id, user_id, path) - the three things the real API scopes by. Dropping
        # either of the first two would let genuinely separate memories share one keyspace, which
        # is exactly the collision the composite exists to prevent.
        self.memories: dict[tuple[str, str, str], str] = {}
        self.calls: list[tuple[str, str]] = []

    def request(self, verb, url, headers=None, params=None, json=None, timeout=None):
        self.calls.append((verb, url))

        if not (headers or {}).get("Authorization"):
            return _error(401, "unauthorized", "Missing bearer token.")

        params, body = params or {}, json or {}
        source = body if verb == "PUT" else params

        # Required on every call: the client must never omit it.
        if not source.get("user_id"):
            return _error(400, "bad_request", INVALID_INPUT, {"user_id": "cannot be blank"})

        path = source.get("path")
        if path is not None and not self._path_is_safe(path):
            return _error(400, "bad_input", "Path must not start with a slash or contain '..' segments.")

        scope = (url.split("/memory-stores/")[1].split("/")[0], source["user_id"])
        if url.endswith("/files/content"):
            return self._read(scope, path)
        if verb == "PUT":
            return self._write(scope, body)
        if verb == "DELETE":
            return self._delete(scope, path)
        return self._list(scope, params.get("path") or "")

    def _read(self, scope, path):
        if (*scope, path) not in self.memories:
            return _error(404, "not_found", "Memory not found.")
        return _Response(200, {"data": self._as_json(scope, path, include_content=True)})

    def _write(self, scope, body):
        path, content = body["path"], body["content"]
        if not content:
            return _error(400, "bad_request", INVALID_INPUT, {"content": "cannot be blank"})
        if len(content) > self.max_content_length:
            details = {"content": f"the length must be between 1 and {self.max_content_length}"}
            return _error(400, "bad_request", INVALID_INPUT, details)
        self.memories[(*scope, path)] = content  # upsert: no overwrite flag, no 409
        return _Response(200, {"data": self._as_json(scope, path)})

    def _delete(self, scope, path):
        if (*scope, path) not in self.memories:
            return _error(404, "not_found", "Memory not found.")
        del self.memories[(*scope, path)]
        return _Response(200, {"data": {"deleted": True}})

    def _list(self, scope, prefix):
        return _Response(
            200,
            {
                "data": [
                    self._as_json(scope, path)
                    for store_id, user_id, path in self.memories
                    if (store_id, user_id) == tuple(scope) and path.startswith(prefix)
                ]
            },
        )

    @staticmethod
    def _path_is_safe(path: str) -> bool:
        if not path:
            return True
        if path.startswith("/") or ".." in path.split("/"):
            return False
        return "%2e" not in path.lower()

    def _as_json(self, scope, path, include_content=False):
        payload = {
            "path": path,
            "size": len(self.memories[(*scope, path)]),
            "updated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        }
        if include_content:
            payload["content"] = self.memories[(*scope, path)]
        return payload
