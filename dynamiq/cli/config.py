import json
import os
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, ValidationError

__all__ = ["Settings", "load_settings", "save_settings"]

_XDG_CONFIG_HOME = Path(os.getenv("XDG_CONFIG_HOME", Path.home() / ".config"))
_CFG_PATH = _XDG_CONFIG_HOME / "dynamiq" / "config.json"
_CREDS_PATH = _XDG_CONFIG_HOME / "dynamiq" / "credentials.json"
DYNAMIQ_BASE_URL = "https://api.us-east-1.aws.getdynamiq.ai"


class Settings(BaseModel):
    org_id: str | None = Field(default=None)
    project_id: str | None = Field(default=None)

    api_host: str | None = Field(default=DYNAMIQ_BASE_URL)
    api_key: str | None = Field(default=None)

    model_config = dict(extra="forbid")

    @property
    def base_url(self) -> str:
        return self.api_host.rstrip("/")

    def __str__(self) -> str:
        return f"Settings(org={self.org_id}, project={self.project_id}, host={self.api_host})"


def _from_env() -> dict[str, Any]:
    """Pick just the env-vars we care about."""
    return {
        k: v
        for k, v in {
            "api_host": os.getenv("DYNAMIQ_API_HOST"),
            "api_key": os.getenv("DYNAMIQ_API_KEY"),
        }.items()
        if v is not None
    }


def load_settings() -> Settings:
    disk: dict[str, Any] = {}
    env: dict = _from_env()
    if _CFG_PATH.exists():
        try:
            disk = json.loads(_CFG_PATH.read_text())
        except json.JSONDecodeError as exc:
            raise SystemExit(f"❌ Corrupted config file at {_CFG_PATH}: {exc}") from exc
    if _CREDS_PATH.exists():
        try:
            env = json.loads(_CREDS_PATH.read_text())
        except json.JSONDecodeError as exc:
            raise SystemExit(f"❌ Corrupted credentials file at {_CREDS_PATH}: {exc}") from exc

    merged = {**disk, **env}
    try:
        return Settings.model_validate(merged)
    except ValidationError as exc:
        raise SystemExit(f"❌ Invalid configuration: {exc}") from exc


def save_settings(settings: Settings) -> None:
    _CFG_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = settings.model_dump(include={"org_id", "project_id"})
    _CFG_PATH.write_text(json.dumps(payload, indent=2))

    _CREDS_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = settings.model_dump(include={"api_key", "api_host"})
    _CREDS_PATH.write_text(json.dumps(payload, indent=2))
