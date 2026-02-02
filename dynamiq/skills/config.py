"""Skills configuration: Dynamiq registry backend only."""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from dynamiq.skills.models import SkillWhitelistEntry
from dynamiq.skills.sources.base import SkillSource


class SkillsBackendType:
    """Skills backend: Dynamiq (API) only."""

    Dynamiq = "Dynamiq"


class SkillsBackendConfig(BaseModel):
    """Backend for skills: Dynamiq API registry."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    type: str = Field(
        default=SkillsBackendType.Dynamiq,
        description="Backend type: 'Dynamiq' (API registry).",
    )
    connection: Any = Field(
        default=None,
        description="Connection instance for Dynamiq backend (resolved by loader).",
    )

    @field_validator("type", mode="before")
    @classmethod
    def normalize_type(cls, v: Any) -> str:
        if v is None or (isinstance(v, str) and v.strip() == ""):
            return SkillsBackendType.Dynamiq
        s = str(v).strip()
        if "Dynamiq" in s or "registry" in s.lower():
            return SkillsBackendType.Dynamiq
        return SkillsBackendType.Dynamiq


class SkillsConfig(BaseModel):
    """Unified skills configuration: enabled flag + backend (Dynamiq) + optional whitelist.

    When enabled is True, backend is required. Only Dynamiq (API registry) backend is supported.
    Accepts dict from YAML/JSON; resolved to source at init via resolve_skills_config().
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    enabled: bool = Field(
        default=False,
        description="Whether skills support is enabled. When True, backend is required.",
    )
    backend: SkillsBackendConfig | None = Field(
        default=None,
        description="Backend (Dynamiq registry). Required when enabled is True.",
    )
    whitelist: list[SkillWhitelistEntry] | None = Field(
        default=None,
        description="Whitelist of skills (id, name, description, version_id). Required for Dynamiq backend.",
    )


def resolve_skills_config(skills: SkillsConfig | dict | None) -> tuple[SkillSource, None] | None:
    """Resolve skills config to (source, None). Returns None if skills disabled or invalid.

    Only Dynamiq backend is supported: source = DynamiqSkillSource(connection, whitelist).
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

    from dynamiq.skills.registry.dynamiq import DynamiqSkillSource

    if not backend.connection:
        raise ValueError("Skills backend requires connection (Dynamiq API).")
    source: SkillSource = DynamiqSkillSource(connection=backend.connection, whitelist=whitelist_entries)
    return (source, None)
