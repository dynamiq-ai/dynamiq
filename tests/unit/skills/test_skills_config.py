"""Unit tests for SkillsConfig and resolve_skills_config."""

from dynamiq.skills.config import SkillsBackendConfig, SkillsBackendType, SkillsConfig, resolve_skills_config
from dynamiq.storages.file.in_memory import InMemoryFileStore


def test_skills_backend_type_enum():
    """SkillsBackendType enum has Local and Dynamiq; strings coerce to enum."""
    assert SkillsBackendType.Local.value == "Local"
    assert SkillsBackendType.Dynamiq.value == "Dynamiq"
    backend = SkillsBackendConfig.model_validate({"type": "Dynamiq"})
    assert backend.type == SkillsBackendType.Dynamiq
    backend_local = SkillsBackendConfig.model_validate({"type": "Local"})
    assert backend_local.type == SkillsBackendType.Local


def test_skills_config_defaults():
    """SkillsConfig has enabled=False and backend=None by default."""
    cfg = SkillsConfig()
    assert cfg.enabled is False
    assert cfg.backend is None
    assert cfg.whitelist is None


def test_skills_config_enabled_with_backend():
    """SkillsConfig accepts enabled=True and backend."""
    backend = SkillsBackendConfig(
        type=SkillsBackendType.Local,
        file_store=InMemoryFileStore(),
    )
    cfg = SkillsConfig(enabled=True, backend=backend)
    assert cfg.enabled is True
    assert cfg.backend is backend


def test_skills_config_from_dict():
    """SkillsConfig can be built from dict (YAML shape); type string coerced to enum."""
    cfg = SkillsConfig.model_validate(
        {
            "enabled": True,
            "backend": {
                "type": "Local",
            },
        }
    )
    assert cfg.enabled is True
    assert cfg.backend is not None
    assert cfg.backend.type == SkillsBackendType.Local


def test_resolve_skills_config_none():
    """resolve_skills_config returns None when skills is None."""
    assert resolve_skills_config(None, InMemoryFileStore()) is None


def test_resolve_skills_config_disabled_dict():
    """resolve_skills_config returns None when dict has enabled=False or no backend."""
    fs = InMemoryFileStore()
    assert resolve_skills_config({"enabled": False, "backend": {"type": "x"}}, fs) is None
    assert resolve_skills_config({"enabled": True}, fs) is None
    assert resolve_skills_config({}, fs) is None


def test_resolve_skills_config_disabled_skills_config():
    """resolve_skills_config returns None when SkillsConfig.enabled is False."""
    fs = InMemoryFileStore()
    backend = SkillsBackendConfig(
        type=SkillsBackendType.Local,
        file_store=fs,
    )
    cfg = SkillsConfig(enabled=False, backend=backend)
    assert resolve_skills_config(cfg, fs) is None


def test_resolve_skills_config_local_backend():
    """resolve_skills_config returns (source, executor) for Local backend with file_store."""
    fs = InMemoryFileStore()
    skills_dict = {
        "enabled": True,
        "backend": {
            "type": SkillsBackendType.Local,
        },
    }
    result = resolve_skills_config(skills_dict, fs)
    assert result is not None
    source, executor = result
    assert source is not None
    assert executor is not None
    assert source.name == "FileStoreSkillSource"
    from dynamiq.skills.executor import SkillExecutor

    assert isinstance(executor, SkillExecutor)


def test_resolve_skills_config_local_backend_skills_config_instance():
    """resolve_skills_config works with SkillsConfig instance (Local backend)."""
    fs = InMemoryFileStore()
    backend = SkillsBackendConfig(
        type=SkillsBackendType.Local,
        file_store=fs,
    )
    cfg = SkillsConfig(enabled=True, backend=backend)
    result = resolve_skills_config(cfg, fs)
    assert result is not None
    source, executor = result
    assert source is not None
    assert executor is not None
