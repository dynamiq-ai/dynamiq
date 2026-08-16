import json
import os
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

__all__ = ["Settings", "DYNAMIQ_BASE_URL"]

_XDG_CONFIG_HOME = Path(os.getenv("XDG_CONFIG_HOME", os.path.join(Path.home(), ".config")))
_CONFIG_FILE_PATH = Path(os.path.join(_XDG_CONFIG_HOME, "dynamiq", "config.json"))
_CREDS_FILE_PATH = Path(os.path.join(_XDG_CONFIG_HOME, "dynamiq", "credentials.json"))
DYNAMIQ_BASE_URL = "https://api.getdynamiq.ai"
# Expected structure of `.dynamiq/config.json`:
# {
#   "org_id": "your-org-id",
#   "project_id": "your-project-id"
# }

# Expected structure of `.dynamiq/credentials.json`:
# {
#   "api_key": "your-api-key-here",
#   "api_host": "https://api.getdynamiq.ai"
# }


def _restrict_permissions(path: Path, mode: int) -> None:
    """Tighten a path's mode, ignoring filesystems that cannot express it.

    Best effort: Windows and some mounts raise on chmod, and refusing to read or write
    credentials there would be worse than leaving the mode alone.
    """
    try:
        os.chmod(path, mode)
    except OSError:
        pass


def _write_private_text(path: Path, text: str) -> None:
    """Write text to a file only its owner can read.

    ``Path.write_text`` leaves the mode to the umask, which is 0022 on most systems and
    lands an API key at 0644. A new file is created 0600; an existing one keeps its old
    mode until the chmod below, since ``O_CREAT`` does not touch the mode of a file that
    already exists. What closes that window is the caller tightening the parent
    directory to 0700 *before* calling this — do not reorder those two steps.

    Best effort: filesystems without POSIX permissions (Windows, some mounts) raise on
    chmod, and failing to save credentials there would be worse than saving them
    without a mode change.
    """
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    # Wrapped in a file object so a short write (a quota or disk-full boundary) raises
    # instead of silently truncating the credentials.
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(text)
    _restrict_permissions(path, 0o600)


class Settings(BaseModel):
    org_id: str | None = Field(default=None)
    project_id: str | None = Field(default=None)

    api_host: str | None = Field(default=DYNAMIQ_BASE_URL)
    api_key: str | None = Field(default=None)

    model_config = ConfigDict(extra="forbid")

    @property
    def base_url(self) -> str:
        return self.api_host.rstrip("/")

    def __str__(self) -> str:
        return f"Settings(org={self.org_id}, project={self.project_id}, host={self.api_host})"

    @classmethod
    def _from_env(cls) -> dict[str, Any]:
        """Pick just the env-vars we care about."""
        return {
            k: v
            for k, v in {
                "api_host": os.getenv("DYNAMIQ_API_HOST"),
                "api_key": os.getenv("DYNAMIQ_API_KEY"),
            }.items()
            if v is not None
        }

    @classmethod
    def load_settings(cls):
        disk: dict[str, Any] = {}
        env: dict = cls._from_env()
        if _CONFIG_FILE_PATH.exists():
            try:
                disk = json.loads(_CONFIG_FILE_PATH.read_text())
            except json.JSONDecodeError as exc:
                raise SystemExit(f"❌ Corrupted config file at {_CONFIG_FILE_PATH}: {exc}") from exc
        if _CREDS_FILE_PATH.exists():
            # Repair on read as well as on write: a file an older version created at 0644
            # would otherwise stay world-readable until the user happens to save again.
            _restrict_permissions(_CREDS_FILE_PATH.parent, 0o700)
            _restrict_permissions(_CREDS_FILE_PATH, 0o600)
            try:
                env = json.loads(_CREDS_FILE_PATH.read_text())
            except json.JSONDecodeError as exc:
                raise SystemExit(f"❌ Corrupted credentials file at {_CREDS_FILE_PATH}: {exc}") from exc

        merged = {**disk, **env}
        try:
            return cls.model_validate(merged)
        except ValidationError as exc:
            raise SystemExit(f"❌ Invalid configuration: {exc}") from exc

    def save_settings(self) -> None:
        _CONFIG_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)
        payload = self.model_dump(include={"org_id", "project_id"})
        _CONFIG_FILE_PATH.write_text(json.dumps(payload, indent=2))

        _CREDS_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)
        _restrict_permissions(_CREDS_FILE_PATH.parent, 0o700)
        payload = self.model_dump(include={"api_key", "api_host"})
        _write_private_text(_CREDS_FILE_PATH, json.dumps(payload, indent=2))
