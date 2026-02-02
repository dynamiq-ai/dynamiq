"""Skills configuration: Dynamiq registry backend only."""

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from dynamiq.skills.models import SkillWhitelistEntry
from dynamiq.skills.sources.base import SkillSource


class SkillsBackendType(str, Enum):
    """Skills backend: Dynamiq (API) only."""

    Dynamiq = "Dynamiq"


class SkillsBackendConfig(BaseModel):
    """Backend for skills: Dynamiq API registry."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    type: SkillsBackendType = Field(
        default=SkillsBackendType.Dynamiq,
        description="Backend type: 'Dynamiq' (API registry).",
    )
    connection: Any = Field(
        default=None,
        description="Connection instance for Dynamiq backend (resolved by loader).",
    )


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


def resolve_skills_config(skills: SkillsConfig | dict | None) -> SkillSource | None:
    """Resolve skills config to a skill source. Returns None if skills disabled or invalid.

    Only Dynamiq backend is supported: DynamiqSkillSource(connection, whitelist).
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
    return DynamiqSkillSource(connection=backend.connection, whitelist=whitelist_entries)
