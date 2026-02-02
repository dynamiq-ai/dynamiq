"""Skills configuration: single backend (source + implied executor)."""

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, field_validator

from dynamiq.skills.executor import SkillExecutor
from dynamiq.skills.models import SkillWhitelistEntry
from dynamiq.skills.sources.base import SkillSource


class SkillsBackendType(str, Enum):
    """Skills backend type: Local (FileStore) or Dynamiq (API)."""

    Local = "Local"
    Dynamiq = "Dynamiq"


def _normalize_backend_type(v: Any) -> SkillsBackendType:
    """Normalize backend type string or enum to SkillsBackendType (accepts class path or name)."""
    if isinstance(v, SkillsBackendType):
        return v
    s = (v or "").strip() if isinstance(v, str) else str(v)
    if s in (SkillsBackendType.Local.value, SkillsBackendType.Dynamiq.value):
        return SkillsBackendType(s)
    if "Dynamiq" in s or "registry" in s.lower():
        return SkillsBackendType.Dynamiq
    return SkillsBackendType.Local


class SkillsBackendConfig(BaseModel):
    """Backend for skills: source (Dynamiq API or Local FileStore).

    Executor is derived from backend: Local backend implies filestore executor
    (subprocess + same FileStore). Dynamiq backend has no local executor (run_script
    not available unless a sandbox executor is added later).
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    type: SkillsBackendType = Field(
        ...,
        description="Backend type: 'Local' (FileStore) or 'Dynamiq' (API).",
    )
    connection: Any = Field(
        default=None,
        description="Connection instance for Dynamiq backend (resolved by loader).",
    )
    file_store: Any = Field(
        default=None,
        description="FileStore for Local backend (or agent file_store_backend when omitted).",
    )
    skills_prefix: str = Field(default=".skills/", description="Prefix for skills (Local).")
    local_skills_dir: str | None = Field(
        default=".skills",
        description="Local directory to sync into FileStore for Local backend (e.g. .skills). "
        "When set, synced during init so skills are available without manual upload. None disables sync.",
    )

    @field_validator("type", mode="before")
    @classmethod
    def normalize_type(cls, v: Any) -> SkillsBackendType:
        return _normalize_backend_type(v)


class SkillsConfig(BaseModel):
    """Unified skills configuration: enabled flag + backend (source) + optional whitelist.

    When enabled is True, backend is required. Executor is derived from backend
    (Local -> filestore executor, Dynamiq -> no executor).
    Accepts dict from YAML/JSON; resolved to (source, executor) at init via resolve_skills_config().
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    enabled: bool = Field(
        default=False,
        description="Whether skills support is enabled. When True, backend is required.",
    )
    backend: SkillsBackendConfig | None = Field(
        default=None,
        description="Backend (source): Dynamiq or Local. Required when enabled is True.",
    )
    whitelist: list[SkillWhitelistEntry] | None = Field(
        default=None,
        description="Optional whitelist (Dynamiq: required list; Local: optional filter).",
    )


def resolve_skills_config(
    skills: SkillsConfig | dict | None,
    file_store: Any,
) -> tuple[SkillSource, SkillExecutor | None] | None:
    """Resolve skills config to (source, executor). Returns None if skills disabled or invalid.

    - skills None -> None (disabled).
    - enabled False or missing -> None (disabled).
    - backend Dynamiq: source = DynamiqSkillSource(connection, whitelist); executor = None.
    - backend Local: source = FileStoreSkillSource(file_store); executor = SkillExecutor(file_store)
      (executor derived from backend).
    """
    if skills is None:
        return None
    if isinstance(skills, dict):
        if not skills.get("enabled") or not skills.get("backend"):
            return None
        backend_data = skills.get("backend")
        whitelist_data = skills.get("whitelist")
    else:
        if not skills.enabled or not skills.backend:
            return None
        backend_data = skills.backend
        whitelist_data = skills.whitelist

    if isinstance(backend_data, dict):
        backend = SkillsBackendConfig.model_validate(backend_data)
    else:
        backend = backend_data

    whitelist_entries: list[SkillWhitelistEntry] | None = None
    if whitelist_data:
        whitelist_entries = [
            SkillWhitelistEntry.model_validate(e) if isinstance(e, dict) else e for e in whitelist_data
        ]

    if backend.type == SkillsBackendType.Dynamiq:
        from dynamiq.skills.registry.dynamiq import DynamiqSkillSource

        if not backend.connection:
            raise ValueError("Skills backend type Dynamiq requires backend.connection")
        source: SkillSource = DynamiqSkillSource(connection=backend.connection, whitelist=whitelist_entries)
        executor: SkillExecutor | None = None
    else:
        fs = backend.file_store or file_store
        if not fs:
            raise ValueError("Skills backend type Local requires file_store (or agent file_store_backend)")
        from dynamiq.skills.sources.filestore import FileStoreSkillSource
        from dynamiq.skills.utils import sync_local_skills_to_filestore

        prefix = backend.skills_prefix or ".skills/"
        if backend.local_skills_dir:
            sync_local_skills_to_filestore(
                fs,
                local_dir=backend.local_skills_dir,
                prefix=prefix,
            )
        source = FileStoreSkillSource(
            file_store=fs,
            skills_prefix=backend.skills_prefix or ".skills/",
        )
        if whitelist_entries:
            source = _FilteringSkillSourceWrapper(source, whitelist_entries)
        executor = SkillExecutor(
            file_store=fs,
            skills_prefix=backend.skills_prefix or ".skills/",
            default_timeout_seconds=120,
            cleanup_work_dir=True,
        )

    return (source, executor)


class _FilteringSkillSourceWrapper(SkillSource):
    """Wraps a Local source to filter by whitelist (name match)."""

    _inner: SkillSource = PrivateAttr()
    _whitelist_names: set = PrivateAttr()

    def __init__(self, inner: SkillSource, whitelist: list[SkillWhitelistEntry], **kwargs):
        super().__init__(name="FilteringSkillSource", **kwargs)
        object.__setattr__(self, "_inner", inner)
        object.__setattr__(self, "_whitelist_names", {e.name for e in whitelist})

    def discover_skills(self):
        refs = self._inner.discover_skills()
        return [r for r in refs if r.name in self._whitelist_names]

    def load_skill(self, name: str):
        return self._inner.load_skill(name) if name in self._whitelist_names else None

    def load_skill_content(self, name, section=None, line_start=None, line_end=None):
        return (
            self._inner.load_skill_content(name, section=section, line_start=line_start, line_end=line_end)
            if name in self._whitelist_names
            else None
        )
