import json
import os
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

__all__ = ["Settings", "DYNAMIQ_BASE_URL"]

_XDG_CONFIG_HOME = Path(os.getenv("XDG_CONFIG_HOME", os.path.join(Path.home(), ".config")))
_CONFIG_FILE_PATH = Path(os.path.join(_XDG_CONFIG_HOME, "dynamiq", "config.json"))
_CREDS_FILE_PATH = Path(os.path.join(_XDG_CONFIG_HOME, "dynamiq", "credentials.json"))
DYNAMIQ_BASE_URL = "https://api.us-east-1.aws.getdynamiq.ai"
# Expected structure of `.dynamiq/config.json`:
# {
#   "org_id": "your-org-id",
#   "project_id": "your-project-id"
# }

# Expected structure of `.dynamiq/credentials.json`:
# {
#   "api_key": "your-api-key-here",
#   "api_host": "https://api.us-east-1.aws.getdynamiq.ai"
# }


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
        payload = self.model_dump(include={"api_key", "api_host"})
        _CREDS_FILE_PATH.write_text(json.dumps(payload, indent=2))
