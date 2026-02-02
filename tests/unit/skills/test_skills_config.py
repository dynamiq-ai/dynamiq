"""Unit tests for SkillsConfig and resolve_skills_config (Dynamiq backend only)."""

from unittest.mock import MagicMock

import pytest

from dynamiq.skills.config import SkillsBackendConfig, SkillsBackendType, SkillsConfig, resolve_skills_config
from dynamiq.skills.registry.dynamiq import DynamiqSkillSource


def test_skills_backend_type():
    """SkillsBackendType has Dynamiq; type normalizes to Dynamiq."""
    assert SkillsBackendType.Dynamiq == "Dynamiq"
    backend = SkillsBackendConfig.model_validate({"type": "Dynamiq"})
    assert backend.type == SkillsBackendType.Dynamiq
    backend_normalized = SkillsBackendConfig.model_validate({"type": "  "})
    assert backend_normalized.type == SkillsBackendType.Dynamiq


def test_skills_config_defaults():
    """SkillsConfig has enabled=False and backend=None by default."""
    cfg = SkillsConfig()
    assert cfg.enabled is False
    assert cfg.backend is None
    assert cfg.whitelist is None


def test_skills_config_enabled_with_backend():
    """SkillsConfig accepts enabled=True and backend (Dynamiq)."""
    backend = SkillsBackendConfig(type=SkillsBackendType.Dynamiq, connection=MagicMock())
    cfg = SkillsConfig(enabled=True, backend=backend)
    assert cfg.enabled is True
    assert cfg.backend is backend


def test_skills_config_from_dict():
    """SkillsConfig can be built from dict (YAML shape); type string coerced to Dynamiq."""
    cfg = SkillsConfig.model_validate(
        {
            "enabled": True,
            "backend": {
                "type": "Dynamiq",
            },
        }
    )
    assert cfg.enabled is True
    assert cfg.backend is not None
    assert cfg.backend.type == SkillsBackendType.Dynamiq


def test_resolve_skills_config_none():
    """resolve_skills_config returns None when skills is None."""
    assert resolve_skills_config(None) is None


def test_resolve_skills_config_disabled_dict():
    """resolve_skills_config returns None when dict has enabled=False or no backend."""
    assert resolve_skills_config({"enabled": False, "backend": {"type": "Dynamiq"}}) is None
    assert resolve_skills_config({"enabled": True}) is None
    assert resolve_skills_config({}) is None


def test_resolve_skills_config_disabled_skills_config():
    """resolve_skills_config returns None when SkillsConfig.enabled is False."""
    backend = SkillsBackendConfig(type=SkillsBackendType.Dynamiq, connection=MagicMock())
    cfg = SkillsConfig(enabled=False, backend=backend)
    assert resolve_skills_config(cfg) is None


def test_resolve_skills_config_dynamiq_backend_missing_connection_raises():
    """resolve_skills_config raises when Dynamiq backend has no connection."""
    skills_dict = {
        "enabled": True,
        "backend": {"type": "Dynamiq"},
    }
    with pytest.raises(ValueError, match="connection"):
        resolve_skills_config(skills_dict)


def test_resolve_skills_config_dynamiq_backend():
    """resolve_skills_config returns source for Dynamiq backend with connection."""
    conn = MagicMock()
    skills_dict = {
        "enabled": True,
        "backend": {"type": "Dynamiq", "connection": conn},
        "whitelist": [],
    }
    source = resolve_skills_config(skills_dict)
    assert source is not None
    assert isinstance(source, DynamiqSkillSource)
    assert source.connection is conn


def test_resolve_skills_config_dynamiq_skills_config_instance():
    """resolve_skills_config works with SkillsConfig instance (Dynamiq backend)."""
    conn = MagicMock()
    backend = SkillsBackendConfig(type=SkillsBackendType.Dynamiq, connection=conn)
    cfg = SkillsConfig(enabled=True, backend=backend)
    source = resolve_skills_config(cfg)
    assert source is not None
    assert isinstance(source, DynamiqSkillSource)
