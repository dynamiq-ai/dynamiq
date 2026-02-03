"""Unit tests for SkillsConfig (source-based: Dynamiq and Local registries)."""

import pytest

from dynamiq.connections import Dynamiq as DynamiqConnection
from dynamiq.skills.config import SkillsConfig
from dynamiq.skills.models import SkillRegistryError
from dynamiq.skills.registries import Dynamiq, Local


def test_skills_config_defaults():
    """SkillsConfig has enabled=False and source=None by default."""
    cfg = SkillsConfig()
    assert cfg.enabled is False
    assert cfg.source is None


def test_skills_config_enabled_without_source_raises():
    """SkillsConfig raises when enabled=True and source is None."""
    with pytest.raises(SkillRegistryError, match="enabled but no source"):
        SkillsConfig(enabled=True, source=None)


def test_skills_config_get_skills_metadata_disabled():
    """get_skills_metadata returns [] when disabled."""
    cfg = SkillsConfig(enabled=False)
    assert cfg.get_skills_metadata() == []


def test_skills_config_get_skills_metadata_no_source():
    """get_skills_metadata returns [] when source is None (enabled=False)."""
    cfg = SkillsConfig()
    assert cfg.get_skills_metadata() == []


def test_skills_config_get_skill_instructions_disabled_raises():
    """get_skill_instructions raises when skills disabled."""
    cfg = SkillsConfig(enabled=False)
    with pytest.raises(SkillRegistryError, match="disabled"):
        cfg.get_skill_instructions("any")


def test_skills_config_source_malformed_type_raises():
    """SkillsConfig raises when source type string has no module path (no dot)."""
    with pytest.raises(SkillRegistryError, match="fully qualified class name"):
        SkillsConfig.model_validate(
            {
                "enabled": True,
                "source": {
                    "type": "Local",
                    "base_path": "/tmp",
                    "whitelist": [],
                },
            }
        )


def test_skills_config_source_resolved_from_dict_dynamiq():
    """SkillsConfig resolves source from dict with type dynamiq.skills.registries.Dynamiq."""
    conn = DynamiqConnection(url="https://api.example.com", api_key="test-key")
    cfg = SkillsConfig.model_validate(
        {
            "enabled": True,
            "source": {
                "type": "dynamiq.skills.registries.Dynamiq",
                "connection": conn,
                "whitelist": [
                    {"id": "sid", "version_id": "vid", "name": "foo", "description": "Foo skill"},
                ],
            },
        }
    )
    assert cfg.enabled is True
    assert cfg.source is not None
    assert isinstance(cfg.source, Dynamiq)
    assert cfg.source.connection is conn
    assert len(cfg.source.whitelist) == 1
    assert cfg.source.whitelist[0].name == "foo"
    metadata = cfg.get_skills_metadata()
    assert len(metadata) == 1
    assert metadata[0].name == "foo"
    assert metadata[0].description == "Foo skill"


def test_skills_config_source_resolved_from_dict_local():
    """SkillsConfig resolves source from dict with type dynamiq.skills.registries.Local."""
    cfg = SkillsConfig.model_validate(
        {
            "enabled": True,
            "source": {
                "type": "dynamiq.skills.registries.Local",
                "base_path": "~/.dynamiq/skills",
                "whitelist": [
                    {"name": "local-skill", "description": "Local skill"},
                ],
            },
        }
    )
    assert cfg.enabled is True
    assert cfg.source is not None
    assert isinstance(cfg.source, Local)
    assert cfg.source.base_path == "~/.dynamiq/skills"
    assert len(cfg.source.whitelist) == 1
    assert cfg.source.whitelist[0].name == "local-skill"
    metadata = cfg.get_skills_metadata()
    assert len(metadata) == 1
    assert metadata[0].name == "local-skill"
    assert metadata[0].description == "Local skill"


def test_skills_config_source_instance_unchanged():
    """When source is already a BaseSkillRegistry instance, it is not re-resolved."""
    registry = Local(base_path="/tmp/skills", whitelist=[])
    cfg = SkillsConfig(enabled=True, source=registry)
    assert cfg.source is registry
