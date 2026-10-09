import json
import os
from pathlib import Path
from typing import Any

import click
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, ValidationError

__all__ = [
    "Settings",
    "DYNAMIQ_BASE_URL",
    "SOURCE_CONFIG_FILE",
    "SOURCE_CREDENTIALS_FILE",
    "SOURCE_DEFAULT",
    "SOURCE_EXPLICIT",
    "SOURCE_NOT_SET",
    "env_source",
]

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

SOURCE_CONFIG_FILE = "config file"
SOURCE_CREDENTIALS_FILE = "credentials file"
SOURCE_DEFAULT = "default"
SOURCE_EXPLICIT = "set explicitly"
SOURCE_NOT_SET = "not set"

_CONFIG_FILE_KEYS = ("org_id", "project_id")
_CREDS_FILE_KEYS = ("api_key", "api_host")

# Each setting and the env vars that can supply it; the first one set wins.
# DYNAMIQ_API_TOKEN / DYNAMIQ_API_BASE_URL are what catalyst injects into sandboxes;
# DYNAMIQ_API_KEY / DYNAMIQ_API_HOST are kept as fallbacks for existing setups.
_ENV_VARS: dict[str, tuple[str, ...]] = {
    "api_host": ("DYNAMIQ_API_BASE_URL", "DYNAMIQ_API_HOST"),
    "api_key": ("DYNAMIQ_API_TOKEN", "DYNAMIQ_API_KEY"),
    "org_id": ("DYNAMIQ_ORG_ID",),
    "project_id": ("DYNAMIQ_PROJECT_ID",),
}


def env_source(var: str) -> str:
    """The source label for a value read from env var `var`."""
    return f"env {var}"


def _read_json(path: Path, label: str) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise SystemExit(f"❌ Corrupted {label} at {path}: {exc}") from exc


class Settings(BaseModel):
    org_id: str | None = Field(default=None)
    project_id: str | None = Field(default=None)

    api_host: str | None = Field(default=DYNAMIQ_BASE_URL)
    api_key: str | None = Field(default=None)

    model_config = ConfigDict(extra="forbid")

    # Where each loaded value came from: "env <VAR>", "config file" or "credentials file".
    _sources: dict[str, str] = PrivateAttr(default_factory=dict)
    # What the files held at load time; None when the instance was built directly rather
    # than loaded, in which case saving writes every field as it always has.
    _stored: dict[str, Any] | None = PrivateAttr(default=None)
    # Fields assigned after construction: the values a command asked to keep.
    _assigned: set[str] = PrivateAttr(default_factory=set)

    def __setattr__(self, name: str, value: Any) -> None:
        if name in type(self).model_fields:
            self._assigned.add(name)
        super().__setattr__(name, value)

    @property
    def base_url(self) -> str:
        return self.api_host.rstrip("/")

    def __str__(self) -> str:
        return f"Settings(org={self.org_id}, project={self.project_id}, host={self.api_host})"

    def source_of(self, field: str) -> str:
        """Where `field`'s effective value came from.

        "env <VAR>", "config file" or "credentials file" for loaded values; "set explicitly"
        for a value passed to the constructor or assigned afterwards; "default" for a field left at its default;
        "not set" when there is no value at all.
        """
        if field not in type(self).model_fields:
            raise KeyError(field)
        if field in self._assigned:
            return SOURCE_EXPLICIT
        if field in self._sources:
            return self._sources[field]
        if getattr(self, field) is None:
            return SOURCE_NOT_SET
        if field in self.model_fields_set:
            return SOURCE_EXPLICIT
        return SOURCE_DEFAULT

    @property
    def sources(self) -> dict[str, str]:
        """`source_of` for every field."""
        return {field: self.source_of(field) for field in type(self).model_fields}

    def stored_value(self, field: str) -> Any:
        """What the files on disk hold for `field`, ignoring env overrides."""
        return (self._stored or {}).get(field)

    @classmethod
    def _env_with_names(cls) -> dict[str, tuple[str, str]]:
        """{field: (env var name, value)} for every setting the environment supplies."""
        found: dict[str, tuple[str, str]] = {}
        for field, names in _ENV_VARS.items():
            for name in names:
                value = os.getenv(name)
                if value:
                    found[field] = (name, value)
                    break
        return found

    @classmethod
    def _from_env(cls) -> dict[str, Any]:
        """Pick just the env-vars we care about."""
        return {field: value for field, (_, value) in cls._env_with_names().items()}

    @classmethod
    def load_settings(cls, warn: bool = True):
        """Merge config file, credentials file and env, in rising precedence.

        Env beats the stored files (catalyst injects env into sandboxes). When an env var
        replaces a different value stored on disk it is said on stderr - a leftover
        DYNAMIQ_PROJECT_ID otherwise points every command at another project with no hint
        why. stdout stays clean for commands that print JSON.
        """
        disk = _read_json(_CONFIG_FILE_PATH, "config file")
        creds = _read_json(_CREDS_FILE_PATH, "credentials file")

        sources: dict[str, str] = {key: SOURCE_CONFIG_FILE for key in disk}
        sources.update({key: SOURCE_CREDENTIALS_FILE for key in creds})
        stored = {**disk, **creds}

        merged = dict(stored)
        for field, (name, value) in cls._env_with_names().items():
            previous = stored.get(field)
            if warn and previous is not None and previous != value:
                click.echo(f"⚠️  {name} overrides {field} from {sources[field]}", err=True)
            merged[field] = value
            sources[field] = env_source(name)

        # A key stored as null is the same as no key.
        sources = {key: source for key, source in sources.items() if merged.get(key) is not None}

        try:
            settings = cls.model_validate(merged)
        except ValidationError as exc:
            raise SystemExit(f"❌ Invalid configuration: {exc}") from exc
        settings._sources = sources
        settings._stored = stored
        return settings

    def _payload(self, keys: tuple[str, ...]) -> dict[str, Any]:
        if self._stored is None:
            return {key: getattr(self, key) for key in keys}
        payload = {}
        for key in keys:
            value = getattr(self, key) if key in self._assigned else self._stored.get(key)
            if value is not None:
                payload[key] = value
        return payload

    def save_settings(self) -> None:
        """Write what the files already held plus the fields assigned since loading.

        A value that only came from an env var is never written: `org set` in a shell with
        DYNAMIQ_API_TOKEN exported must not copy that token into credentials.json in
        plaintext, nor pin an env-supplied host or project for every later run.
        """
        _CONFIG_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)
        _CONFIG_FILE_PATH.write_text(json.dumps(self._payload(_CONFIG_FILE_KEYS), indent=2))

        _CREDS_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)
        _CREDS_FILE_PATH.write_text(json.dumps(self._payload(_CREDS_FILE_KEYS), indent=2))
