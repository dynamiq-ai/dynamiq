"""Unit tests for skill registries (Dynamiq and FileSystem)."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from dynamiq.skills.registries import Dynamiq, FileSystem
from dynamiq.skills.registries.dynamiq import DynamiqSkillEntry
from dynamiq.skills.registries.filesystem import FileSystemSkillEntry
from dynamiq.skills.types import SkillRegistryError


class TestFileSystemRegistry:
    """Tests for FileSystem skill registry."""

    def test_get_skills_metadata(self):
        """FileSystem.get_skills_metadata returns metadata from allowed_skills."""
        registry = FileSystem(
            base_path="/nonexistent",
            allowed_skills=[
                FileSystemSkillEntry(name="a", description="A"),
                FileSystemSkillEntry(name="b", description=None),
            ],
        )
        metadata = registry.get_skills_metadata()
        assert len(metadata) == 2
        assert metadata[0].name == "a"
        assert metadata[0].description == "A"
        assert metadata[1].name == "b"
        assert metadata[1].description is None

    def test_get_skill_instructions_not_found_raises(self):
        """FileSystem.get_skill_instructions raises when skill path does not exist."""
        with tempfile.TemporaryDirectory() as tmp:
            registry = FileSystem(
                base_path=tmp, allowed_skills=[FileSystemSkillEntry(name="missing", description=None)]
            )
            with pytest.raises(SkillRegistryError, match="not found"):
                registry.get_skill_instructions("missing")

    def test_get_skill_instructions_found(self):
        """FileSystem.get_skill_instructions returns content from SKILL.md."""
        with tempfile.TemporaryDirectory() as tmp:
            skill_dir = Path(tmp) / "my-skill"
            skill_dir.mkdir()
            (skill_dir / "SKILL.md").write_text("# My Skill\n\nDo something.", encoding="utf-8")
            registry = FileSystem(
                base_path=tmp,
                allowed_skills=[FileSystemSkillEntry(name="my-skill", description="My skill")],
            )
            instructions = registry.get_skill_instructions("my-skill")
            assert instructions.name == "my-skill"
            assert instructions.description == "My skill"
            assert instructions.instructions == "# My Skill\n\nDo something."

    def test_get_skill_instructions_not_in_allowed_skills_raises(self):
        """FileSystem.get_skill_instructions raises when skill name not in allowed_skills."""
        with tempfile.TemporaryDirectory() as tmp:
            skill_dir = Path(tmp) / "other-skill"
            skill_dir.mkdir()
            (skill_dir / "SKILL.md").write_text("content", encoding="utf-8")
            registry = FileSystem(
                base_path=tmp,
                allowed_skills=[FileSystemSkillEntry(name="my-skill", description=None)],
            )
            with pytest.raises(SkillRegistryError, match="not in allowed skills"):
                registry.get_skill_instructions("other-skill")

    def test_get_skill_instructions_path_traversal_raises(self):
        """FileSystem.get_skill_instructions rejects skill names with path components."""
        with tempfile.TemporaryDirectory() as tmp:
            for invalid_name in ("../other", "foo/bar", "..", "a\\b"):
                registry = FileSystem(
                    base_path=tmp,
                    allowed_skills=[FileSystemSkillEntry(name=invalid_name, description=None)],
                )
                with pytest.raises(SkillRegistryError, match="Invalid skill name|outside base"):
                    registry.get_skill_instructions(invalid_name)

    def test_type_computed_field(self):
        """FileSystem registry has type computed field with module and class name."""
        registry = FileSystem(base_path="/tmp", allowed_skills=[])
        assert "FileSystem" in registry.type
        assert "dynamiq.skills.registries" in registry.type


class TestDynamiqRegistry:
    """Tests for Dynamiq skill registry."""

    def test_get_skills_metadata(self):
        """Dynamiq.get_skills_metadata returns metadata from allowed_skills."""
        conn = MagicMock()
        registry = Dynamiq.model_construct(
            connection=conn,
            allowed_skills=[
                DynamiqSkillEntry(
                    id="i1",
                    version_id="v1",
                    name="skill1",
                    description="First",
                ),
            ],
        )
        metadata = registry.get_skills_metadata()
        assert len(metadata) == 1
        assert metadata[0].name == "skill1"
        assert metadata[0].description == "First"

    def test_get_skill_instructions_not_in_allowed_skills_raises(self):
        """Dynamiq.get_skill_instructions raises when skill name not in allowed_skills."""
        conn = MagicMock()
        registry = Dynamiq.model_construct(connection=conn, allowed_skills=[])
        with pytest.raises(SkillRegistryError, match="not in allowed skills"):
            registry.get_skill_instructions("unknown")

    def test_get_skill_instructions_success(self):
        """Dynamiq.get_skill_instructions fetches and returns instructions."""
        conn = MagicMock()
        conn.conn_params = {"api_base": "https://api.example.com"}
        resp = MagicMock()
        resp.status_code = 200
        resp.content = b"# Skill content"
        resp.text = "# Skill content"
        resp.json.side_effect = ValueError("not JSON")
        client = MagicMock()
        client.request.return_value = resp
        conn.connect.return_value = client

        registry = Dynamiq.model_construct(
            connection=conn,
            allowed_skills=[
                DynamiqSkillEntry(
                    id="skill-id",
                    version_id="ver-id",
                    name="my-skill",
                    description="My skill",
                ),
            ],
        )
        instructions = registry.get_skill_instructions("my-skill")
        assert instructions.name == "my-skill"
        assert instructions.description == "My skill"
        assert instructions.instructions == "# Skill content"
        client.request.assert_called_once()
        call_args = client.request.call_args
        assert "/v1/skills/skill-id/versions/ver-id/instructions" in call_args[0][1]

    def test_get_skill_instructions_success_json_response(self):
        """Dynamiq.get_skill_instructions accepts JSON response with instructions and optional metadata."""
        conn = MagicMock()
        conn.conn_params = {"api_base": "https://api.example.com"}
        resp = MagicMock()
        resp.status_code = 200
        resp.content = b'{"instructions": "# JSON skill", "metadata": {"version": "1.0"}}'
        resp.headers = {"content-type": "application/json"}
        resp.json.return_value = {
            "instructions": "# JSON skill",
            "metadata": {"version": "1.0", "content_type": "markdown"},
        }
        client = MagicMock()
        client.request.return_value = resp
        conn.connect.return_value = client

        registry = Dynamiq.model_construct(
            connection=conn,
            allowed_skills=[
                DynamiqSkillEntry(
                    id="skill-json",
                    version_id="ver-id",
                    name="json-skill",
                    description="JSON skill",
                ),
            ],
        )
        instructions = registry.get_skill_instructions("json-skill")
        assert instructions.name == "json-skill"
        assert instructions.instructions == "# JSON skill"
        assert instructions.metadata == {"version": "1.0", "content_type": "markdown"}

    def test_get_skill_instructions_lookup_by_name(self):
        """Dynamiq.get_skill_instructions finds entry by name and fetches instructions."""
        conn = MagicMock()
        conn.conn_params = {"api_base": "https://api.example.com"}
        resp = MagicMock()
        resp.status_code = 200
        resp.content = b"# Skill by id"
        resp.text = "# Skill by id"
        resp.json.side_effect = ValueError("not JSON")
        client = MagicMock()
        client.request.return_value = resp
        conn.connect.return_value = client

        registry = Dynamiq.model_construct(
            connection=conn,
            allowed_skills=[
                DynamiqSkillEntry(
                    id="skill-by-id",
                    version_id="ver-id",
                    name="skill-by-id",
                    description="Cached description",
                ),
            ],
        )
        metadata = registry.get_skills_metadata()
        assert metadata[0].name == "skill-by-id"
        instructions = registry.get_skill_instructions("skill-by-id")
        assert instructions.name == "skill-by-id"
        assert instructions.description == "Cached description"
        assert instructions.instructions == "# Skill by id"
        client.request.assert_called_once()
        call_args = client.request.call_args
        assert "/v1/skills/skill-by-id/versions/ver-id/instructions" in call_args[0][1]
