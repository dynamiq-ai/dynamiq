"""A stand-in for the platform's artifact API, for tests.

``ArtifactAPISimulator`` stands in for the platform, so everything above the socket is real: the
same client, URLs, multipart bodies, If-Match headers and status codes.
"""

import hashlib
import io
import json as json_lib
import mimetypes
import posixpath
import zipfile
from datetime import datetime, timezone
from uuid import uuid4

from dynamiq.artifacts import TEXT_KINDS, ArtifactKind

API_URL = "https://artifacts.simulated"
KINDS = {kind.value for kind in ArtifactKind}
TEXT_KIND_VALUES = {kind.value for kind in TEXT_KINDS}

MAX_BYTES = {"html": 16 << 20, "bundle": 100 << 20}
DEFAULT_MAX_BYTES = 25 << 20
KIND_MIME_TYPES = {
    "html": "text/html",
    "markdown": "text/markdown",
    "svg": "image/svg+xml",
    "mermaid": "text/plain",
    "json": "application/json",
    "chart": "application/json",
    "csv": "text/csv",
    "pdf": "application/pdf",
    "bundle": "application/zip",
}
IMAGE_SIGNATURES = (b"\x89PNG\r\n\x1a\n", b"\xff\xd8\xff", b"GIF87a", b"GIF89a")
PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 16


class _Response:
    """The subset of a `requests` response that the Dynamiq artifact backend reads."""

    def __init__(self, status_code: int, payload=None, raw: bytes | None = None):
        self.status_code = status_code
        self._payload = payload
        if raw is not None:
            self.content, self.text = raw, raw.decode(errors="replace")
        else:
            self.text = json_lib.dumps(payload) if payload is not None else ""
            self.content = self.text.encode()

    def json(self):
        if self._payload is None:
            raise ValueError("response has no JSON body")
        return self._payload


def _error(status_code: int, code: str, message: str, details: dict | None = None) -> _Response:
    error = {"code": code, "message": message}
    if details:
        error["details"] = details
    return _Response(status_code, {"error": error})


def _checksum(raw: bytes) -> str:
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _invalid_input(message: str) -> _Response:
    return _error(400, "bad_request", message)


def _check_content(kind: str, content: bytes, entry_path: str | None) -> tuple[_Response | None, str | None]:
    """Validate a version's bytes against the artifact's kind, as the platform does.

    Returns the error response, or None and the bundle's entry path (index.html unless given).
    """
    if len(content) > MAX_BYTES.get(kind, DEFAULT_MAX_BYTES):
        return _error(413, "payload_too_large", "The content is too large."), None
    if not content:
        return _invalid_input("The content is empty."), None
    if kind in TEXT_KIND_VALUES:
        try:
            text = content.decode("utf-8")
        except UnicodeDecodeError:
            return _invalid_input("The content is not valid UTF-8 text."), None
        if kind in ("json", "chart"):
            try:
                spec = json_lib.loads(text)
            except ValueError:
                return _invalid_input("The content is not valid JSON."), None
            if kind == "chart" and "vega-lite" not in str((spec if isinstance(spec, dict) else {}).get("$schema")):
                return _invalid_input('A chart must be a Vega-Lite spec with a "$schema" naming Vega-Lite.'), None
    if kind == "image" and not (
        content.startswith(IMAGE_SIGNATURES) or (content[:4] == b"RIFF" and content[8:12] == b"WEBP")
    ):
        return _invalid_input("The content is not an image."), None
    if kind == "pdf" and not content.startswith(b"%PDF-"):
        return _invalid_input("The content is not a PDF."), None
    if kind != "bundle":
        return None, None
    try:
        names = [i.filename for i in zipfile.ZipFile(io.BytesIO(content)).infolist() if not i.is_dir()]
    except zipfile.BadZipFile:
        return _invalid_input("The bundle is not a valid zip archive."), None
    if any(n.startswith("/") or ".." in n.split("/") for n in names):
        return _invalid_input("The bundle has an entry outside the archive or a symbolic link."), None
    entry = posixpath.normpath(entry_path) if entry_path else "index.html"
    if entry not in names:
        return _invalid_input("The bundle has no file at entry_path."), None
    return None, entry


def _mime_type(kind: str, file_name: str, declared: str | None) -> str:
    """The declared type, else the one the kind fixes, else a guess from the file name."""
    return declared or KIND_MIME_TYPES.get(kind) or mimetypes.guess_type(file_name)[0] or "application/octet-stream"


class ArtifactAPISimulator:
    """In-process implementation of the platform's /v1/artifacts routes.

    Deliberately strict where the platform is: bearer token, known kinds, text kinds only as JSON,
    content validated against the artifact's kind, a MIME type and bundle entry page per version,
    immutable versions, a repeated upload returning the latest version, and If-Match on the latest
    version id.
    """

    def __init__(self):
        self.artifacts: dict[str, dict] = {}
        self.calls: list[tuple[str, str]] = []
        self.if_matches: list[str | None] = []

    def request(self, verb, url, headers=None, params=None, json=None, data=None, files=None, timeout=None):
        path = url.split("/v1/artifacts", 1)[1]
        self.calls.append((verb, path))
        headers, params = headers or {}, params or {}

        if not headers.get("Authorization"):
            return _error(401, "unauthorized", "The request requires authentication.")

        parts = [p for p in path.split("/") if p]
        client = headers.get("User-Agent")
        if verb == "POST" and not parts:
            return self._create(dict(json or {}), client, upload=False)
        if verb == "POST" and parts == ["upload"]:
            return self._create(self._upload_body(data, files), client, upload=True)
        if verb == "GET" and not parts:
            return self._list(params)
        if verb == "GET" and len(parts) == 1:
            return self._get(parts[0])
        if verb == "POST" and len(parts) == 2 and parts[1] == "versions":
            return self._add(parts[0], dict(json or {}), headers.get("If-Match"), client, upload=False)
        if verb == "POST" and len(parts) == 3 and parts[1:] == ["versions", "upload"]:
            return self._add(parts[0], self._upload_body(data, files), headers.get("If-Match"), client, upload=True)
        if verb == "GET" and len(parts) == 2 and parts[1] == "versions":
            return self._versions(parts[0])
        if verb == "GET" and len(parts) == 4 and parts[1] == "versions" and parts[3] == "download":
            return self._download(parts[0], parts[2])
        if verb in ("POST", "DELETE") and len(parts) == 2 and parts[1] == "share":
            return self._share(parts[0], dict(json or {}), revoke=verb == "DELETE")
        return _error(404, "not_found", f"no route for {verb} {path}")

    @staticmethod
    def _upload_body(data, files) -> dict:
        """The JSON fields of the `data` part, plus the file's name and bytes."""
        body = json_lib.loads((data or {}).get("data") or "{}")
        name, content, _ = files["file"]
        body["_file_name"] = name
        body["content"] = content if isinstance(content, bytes) else content.read()
        return body

    def _create(self, body, client, upload):
        file_name = body.get("file_name") or body.get("_file_name")
        missing = {k: "cannot be blank" for k in ("name", "kind") if not body.get(k)}
        if not file_name:
            missing["file_name"] = "cannot be blank"
        if not upload and not body.get("content"):
            missing["content"] = "cannot be blank"
        if missing:
            return _error(400, "bad_request", "The request could not be processed due to invalid input.", missing)
        if body["kind"] not in KINDS or (not upload and body["kind"] not in TEXT_KIND_VALUES):
            return _error(
                400,
                "bad_request",
                "The request could not be processed due to invalid input.",
                {"kind": "must be a valid value"},
            )
        content = body["content"].encode() if isinstance(body["content"], str) else body["content"]
        problem, entry_path = _check_content(body["kind"], content, body.get("entry_path"))
        if problem:
            return problem
        artifact_id = str(uuid4())
        self.artifacts[artifact_id] = {
            "id": artifact_id,
            "store_id": body.get("store_id"),
            "user_id": body.get("user_id"),
            "file_name": file_name,
            "name": body["name"],
            "kind": body["kind"],
            "visibility": "private",
            "versions": [],
            "share": None,
        }
        mime_type = _mime_type(body["kind"], file_name, body.get("mime_type"))
        self._record(artifact_id, content, body, client, mime_type, entry_path)
        return _Response(201, {"data": self._as_json(artifact_id)})

    def _add(self, artifact_id, body, if_match, client, upload):
        artifact = self.artifacts.get(artifact_id)
        if artifact is None:
            return _error(404, "not_found", "The requested resource was not found.")
        latest = artifact["versions"][-1]
        self.if_matches.append(if_match)
        if if_match and if_match.strip('"') != latest["id"]:
            return _error(
                412,
                "precondition_failed",
                f"The artifact has a newer version than If-Match. The latest version is {latest['version']}.",
            )
        content = body.get("content")
        if content is None:
            return _error(
                400,
                "bad_request",
                "The request could not be processed due to invalid input.",
                {"content": "cannot be blank"},
            )
        content = content.encode() if isinstance(content, str) else content
        problem, entry_path = _check_content(artifact["kind"], content, body.get("entry_path"))
        if problem:
            return problem
        repeats = _checksum(content) == latest["checksum"] and all(
            body.get(field) is None or body[field] == latest[field]
            for field in ("name", "description", "mime_type", "entry_path")
        )
        if repeats:
            return _Response(200, {"data": self._version_json(artifact_id, latest)})
        mime_type = body.get("mime_type") or KIND_MIME_TYPES.get(artifact["kind"]) or latest["mime_type"]
        version = self._record(artifact_id, content, body, client, mime_type, entry_path)
        return _Response(201, {"data": self._version_json(artifact_id, version)})

    def _get(self, artifact_id):
        if artifact_id not in self.artifacts:
            return _error(404, "not_found", "The requested resource was not found.")
        return _Response(200, {"data": self._as_json(artifact_id)})

    def _versions(self, artifact_id):
        artifact = self.artifacts.get(artifact_id)
        if artifact is None:
            return _error(404, "not_found", "The requested resource was not found.")
        versions = [self._version_json(artifact_id, v) for v in reversed(artifact["versions"])]
        return _Response(200, {"data": versions})

    def _download(self, artifact_id, version_id):
        artifact = self.artifacts.get(artifact_id)
        if artifact is None:
            return _error(404, "not_found", "The requested resource was not found.")
        if version_id == "latest":
            return _Response(200, raw=artifact["versions"][-1]["content"])
        for version in artifact["versions"]:
            if version["id"] == version_id:
                return _Response(200, raw=version["content"])
        return _error(404, "not_found", "The requested resource was not found.")

    def _list(self, params):
        found = [a for a in self.artifacts.values() if a["store_id"] == params.get("store_id")]
        if params.get("user_id"):
            found = [a for a in found if a["user_id"] == params["user_id"]]
        if params.get("kind"):
            found = [a for a in found if a["kind"] == params["kind"]]
        found.sort(key=lambda a: a["versions"][-1]["created_at"], reverse=True)
        return _Response(200, {"data": [self._as_json(a["id"]) for a in found[: int(params.get("page_size", 25))]]})

    def _share(self, artifact_id, body, revoke):
        artifact = self.artifacts.get(artifact_id)
        if artifact is None:
            return _error(404, "not_found", "The requested resource was not found.")
        if revoke:
            artifact["share"], artifact["visibility"] = None, "private"
            return _Response(200, {"message": "The object was deleted."})
        share_id = (artifact["share"] or {}).get("id") or str(uuid4())
        artifact["share"] = {
            "id": share_id,
            "artifact_id": artifact_id,
            "pinned_version_id": body.get("pinned_version_id"),
            "expires_at": body.get("expires_at"),
            "url": f"https://app.simulated/a/{share_id}",
        }
        artifact["visibility"] = "link"
        return _Response(200, {"data": artifact["share"]})

    def _record(
        self,
        artifact_id,
        content: bytes,
        body: dict,
        client: str | None,
        mime_type: str,
        entry_path: str | None,
    ) -> dict:
        artifact = self.artifacts[artifact_id]
        versions = artifact["versions"]
        version = {
            "id": str(uuid4()),
            "version": len(versions) + 1,
            "name": body.get("name") or artifact["name"],
            "description": body.get("description"),
            "mime_type": mime_type,
            "entry_path": entry_path,
            "content": content,
            "checksum": _checksum(content),
            "client": client,
            "created_at": _now(),
        }
        versions.append(version)
        artifact["name"] = version["name"]
        return version

    def _version_json(self, artifact_id, version):
        return {
            "id": version["id"],
            "artifact_id": artifact_id,
            "version": version["version"],
            "name": version["name"],
            "description": version["description"],
            "mime_type": version["mime_type"],
            "size": len(version["content"]),
            "checksum": version["checksum"],
            "entry_path": version["entry_path"],
            "source": {"client": version["client"]},
            "created_at": version["created_at"],
        }

    def _as_json(self, artifact_id):
        artifact = self.artifacts[artifact_id]
        latest = artifact["versions"][-1]
        return {
            **{k: artifact[k] for k in ("id", "store_id", "user_id", "file_name", "name", "kind")},
            "mime_type": latest["mime_type"],
            "visibility": artifact["visibility"],
            "latest_version_id": latest["id"],
            "latest_version": self._version_json(artifact_id, latest),
            "url": f"https://app.simulated/artifacts/{artifact_id}",
        }

    def latest_text(self, artifact_id) -> str:
        return self.artifacts[artifact_id]["versions"][-1]["content"].decode(errors="replace")
