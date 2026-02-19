"""Unit tests for skill ingestion into sandbox."""

import zipfile
from io import BytesIO
from unittest.mock import MagicMock, patch

import pytest

from dynamiq.skills.registries.dynamiq import Dynamiq, DynamiqSkillEntry
from dynamiq.skills.types import SkillRegistryError
from dynamiq.skills.utils import _upload_zip_to_sandbox, ingest_skills_into_sandbox


def _make_zip(files: dict[str, bytes]) -> bytes:
    """Build a zip archive from {path: content} and return bytes."""
    buf = BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for path, content in files.items():
            zf.writestr(path, content)
    return buf.getvalue()


class TestIngestSkillsIntoSandbox:
    """Tests for ingest_skills_into_sandbox."""

    def test_uses_custom_sandbox_skills_base_path(self):
        """When sandbox_skills_base_path is set, uploads go to that base."""
        zip_bytes = _make_zip({"scripts/run.py": b"print(1)"})
        sandbox = MagicMock()
        sandbox.base_path = "/home/user"
        registry = Dynamiq.model_construct(
            connection=MagicMock(),
            skills=[
                DynamiqSkillEntry(id="id1", version_id="v1", name="my-skill", description=None),
            ],
        )
        with patch(
            "dynamiq.skills.registries.dynamiq.Dynamiq.download_skill_archive", return_value=zip_bytes
        ) as mock_download:
            result = ingest_skills_into_sandbox(
                sandbox,
                registry,
                sandbox_skills_base_path="/opt/custom/skills",
            )

        assert result == ["my-skill"]
        mock_download.assert_called_once_with("id1", "v1")
        upload_calls = sandbox.upload_file.call_args_list
        assert len(upload_calls) == 1
        assert upload_calls[0][1]["destination_path"] == "/opt/custom/skills/my-skill/skill.zip"
        assert sandbox.run_command_shell.called
        assert "unzip" in sandbox.run_command_shell.call_args[0][0]

    def test_uses_default_base_when_sandbox_skills_base_path_none(self):
        """When sandbox_skills_base_path is None, base is {sandbox.base_path}/skills."""
        zip_bytes = _make_zip({"SKILL.md": b"# Skill"})
        sandbox = MagicMock()
        sandbox.base_path = "/home/agent"
        registry = Dynamiq.model_construct(
            connection=MagicMock(),
            skills=[
                DynamiqSkillEntry(id="id1", version_id="v1", name="default-skill", description=None),
            ],
        )
        with patch("dynamiq.skills.registries.dynamiq.Dynamiq.download_skill_archive", return_value=zip_bytes):
            result = ingest_skills_into_sandbox(sandbox, registry)

        assert result == ["default-skill"]
        upload_calls = sandbox.upload_file.call_args_list
        assert upload_calls[0][1]["destination_path"] == "/home/agent/skills/default-skill/skill.zip"
        assert "unzip" in sandbox.run_command_shell.call_args[0][0]

    def test_batch_unzip_once_for_all_skills(self):
        """Multiple skills get one batch unzip shell call (no separate mkdir)."""
        zip_bytes = _make_zip({"f": b"x"})
        sandbox = MagicMock()
        sandbox.base_path = "/home/user"
        registry = Dynamiq.model_construct(
            connection=MagicMock(),
            skills=[
                DynamiqSkillEntry(id="i1", version_id="v1", name="skill-a", description=None),
                DynamiqSkillEntry(id="i2", version_id="v2", name="skill-b", description=None),
            ],
        )
        with patch("dynamiq.skills.registries.dynamiq.Dynamiq.download_skill_archive", return_value=zip_bytes):
            ingest_skills_into_sandbox(sandbox, registry)

        shell_calls = sandbox.run_command_shell.call_args_list
        unzip_calls = [c[0][0] for c in shell_calls if "unzip" in c[0][0]]
        assert len(unzip_calls) == 1
        assert "skill-a" in unzip_calls[0] and "skill-b" in unzip_calls[0]

    def test_returns_list_of_ingested_skill_names(self):
        """Returns the list of skill names that were successfully ingested."""
        zip_bytes = _make_zip({"a.txt": b"a"})
        sandbox = MagicMock()
        sandbox.base_path = "/home/user"
        registry = Dynamiq.model_construct(
            connection=MagicMock(),
            skills=[
                DynamiqSkillEntry(id="i1", version_id="v1", name="skill-a", description=None),
                DynamiqSkillEntry(id="i2", version_id="v2", name="skill-b", description=None),
            ],
        )
        with patch(
            "dynamiq.skills.registries.dynamiq.Dynamiq.download_skill_archive", return_value=zip_bytes
        ) as mock_download:
            result = ingest_skills_into_sandbox(sandbox, registry)

        assert result == ["skill-a", "skill-b"]
        assert mock_download.call_count == 2

    def test_skill_names_filter_only_ingests_requested(self):
        """When skill_names is set, only those skills are ingested."""
        zip_bytes = _make_zip({"x": b"x"})
        sandbox = MagicMock()
        sandbox.base_path = "/home/user"
        registry = Dynamiq.model_construct(
            connection=MagicMock(),
            skills=[
                DynamiqSkillEntry(id="i1", version_id="v1", name="one", description=None),
                DynamiqSkillEntry(id="i2", version_id="v2", name="two", description=None),
                DynamiqSkillEntry(id="i3", version_id="v3", name="three", description=None),
            ],
        )
        with patch(
            "dynamiq.skills.registries.dynamiq.Dynamiq.download_skill_archive", return_value=zip_bytes
        ) as mock_download:
            result = ingest_skills_into_sandbox(sandbox, registry, skill_names=["one", "three"])

        assert result == ["one", "three"]
        mock_download.assert_any_call("i1", "v1")
        mock_download.assert_any_call("i3", "v3")
        assert mock_download.call_count == 2

    def test_download_failure_raises_skill_registry_error(self):
        """When download_skill_archive fails, raises SkillRegistryError."""
        sandbox = MagicMock()
        sandbox.base_path = "/home/user"
        registry = Dynamiq.model_construct(
            connection=MagicMock(),
            skills=[
                DynamiqSkillEntry(id="i1", version_id="v1", name="fail-skill", description=None),
            ],
        )
        with patch(
            "dynamiq.skills.registries.dynamiq.Dynamiq.download_skill_archive",
            side_effect=ConnectionError("network error"),
        ):
            with pytest.raises(SkillRegistryError, match="Failed to ingest skill 'fail-skill'"):
                ingest_skills_into_sandbox(sandbox, registry)

    def test_upload_failure_raises_skill_registry_error(self):
        """When sandbox upload fails, raises SkillRegistryError."""
        zip_bytes = _make_zip({"f": b"content"})
        sandbox = MagicMock()
        sandbox.base_path = "/home/user"
        sandbox.upload_file.side_effect = OSError("disk full")
        registry = Dynamiq.model_construct(
            connection=MagicMock(),
            skills=[
                DynamiqSkillEntry(id="i1", version_id="v1", name="upload-fail", description=None),
            ],
        )
        with patch("dynamiq.skills.registries.dynamiq.Dynamiq.download_skill_archive", return_value=zip_bytes):
            with pytest.raises(SkillRegistryError, match="Failed to ingest skill 'upload-fail'"):
                ingest_skills_into_sandbox(sandbox, registry)

    def test_batch_unzip_failure_raises_skill_registry_error(self):
        """When batch unzip returns non-zero exit_code, raises SkillRegistryError."""
        zip_bytes = _make_zip({"f": b"x"})
        sandbox = MagicMock()
        sandbox.base_path = "/home/user"
        sandbox.run_command_shell.return_value = MagicMock(stdout="", stderr="unzip: command not found", exit_code=1)
        registry = Dynamiq.model_construct(
            connection=MagicMock(),
            skills=[
                DynamiqSkillEntry(id="i1", version_id="v1", name="skill-a", description=None),
            ],
        )
        with patch("dynamiq.skills.registries.dynamiq.Dynamiq.download_skill_archive", return_value=zip_bytes):
            with pytest.raises(SkillRegistryError, match="Failed to unzip skills in sandbox"):
                ingest_skills_into_sandbox(sandbox, registry)


class TestUploadZipToSandbox:
    """Tests for _upload_zip_to_sandbox (path handling and zip extraction)."""

    def test_uploads_zip_and_runs_unzip_in_sandbox(self):
        """One zip is uploaded, then unzip is run in sandbox (fast path)."""
        zip_bytes = _make_zip(
            {
                "scripts/run.py": b"#!/usr/bin/env python",
                "scripts/utils/helper.py": b"def help(): pass",
            }
        )
        sandbox = MagicMock()
        count = _upload_zip_to_sandbox(sandbox, zip_bytes, "/home/user/skills", "my-skill")

        assert count == 2
        assert sandbox.upload_file.call_count == 1
        assert sandbox.upload_file.call_args[1]["destination_path"] == "/home/user/skills/my-skill/skill.zip"
        unzip_cmd = sandbox.run_command_shell.call_args[0][0]
        assert "unzip" in unzip_cmd
        assert "/home/user/skills/my-skill/skill.zip" in unzip_cmd
        assert "/home/user/skills/my-skill" in unzip_cmd

    def test_skips_directories_in_zip(self):
        """Zip directory entries are excluded from sanitized zip; one zip uploaded."""
        buf = BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("scripts/", b"")  # directory
            zf.writestr("scripts/run.py", b"code")
        zip_bytes = buf.getvalue()
        sandbox = MagicMock()
        count = _upload_zip_to_sandbox(sandbox, zip_bytes, "/base", "s1")
        assert count == 1
        assert sandbox.upload_file.call_count == 1
        assert sandbox.upload_file.call_args[1]["destination_path"] == "/base/s1/skill.zip"
        assert "unzip" in sandbox.run_command_shell.call_args[0][0]

    def test_skips_path_traversal_and_absolute_names(self):
        """Sanitized zip contains only safe entries (no .. or leading /)."""
        zip_bytes = _make_zip(
            {
                "normal.txt": b"ok",
                "../escape.txt": b"bad",
                "/absolute.txt": b"bad",
                "sub/../other.txt": b"bad",
            }
        )
        sandbox = MagicMock()
        count = _upload_zip_to_sandbox(sandbox, zip_bytes, "/base", "s1")
        assert count == 1
        assert sandbox.upload_file.call_count == 1
        # Uploaded zip must contain only normal.txt (sanitized).
        uploaded_content = sandbox.upload_file.call_args[0][1]
        with zipfile.ZipFile(BytesIO(uploaded_content), "r") as zf:
            names = zf.namelist()
        assert names == ["normal.txt"]
        assert "unzip" in sandbox.run_command_shell.call_args[0][0]

    def test_returns_file_count(self):
        """_upload_zip_to_sandbox returns the number of files in the zip; uploads one zip and runs unzip."""
        zip_bytes = _make_zip({"a": b"1", "b/c": b"2"})
        sandbox = MagicMock()
        count = _upload_zip_to_sandbox(sandbox, zip_bytes, "/home/skills", "my-skill")
        assert count == 2
        assert sandbox.upload_file.call_count == 1
        assert "unzip" in sandbox.run_command_shell.call_args[0][0]

    def test_fallback_to_per_file_when_unzip_fails(self):
        """When unzip returns non-zero exit_code, fall back to per-file upload."""
        zip_bytes = _make_zip({"scripts/run.py": b"code", "SKILL.md": b"# Skill"})
        sandbox = MagicMock()
        # First call: unzip (exit_code=1); second: rm cleanup. Use simple objects to avoid pulling in e2b_desktop.
        fail_result = MagicMock(stdout="", stderr="unzip: command not found", exit_code=1)
        ok_result = MagicMock(stdout="", stderr="", exit_code=0)
        sandbox.run_command_shell.side_effect = [fail_result, ok_result]
        count = _upload_zip_to_sandbox(sandbox, zip_bytes, "/base", "s1")
        assert count == 2
        # One zip upload, then per-file fallback: 1 + 2 = 3 upload_file calls.
        assert sandbox.upload_file.call_count == 3
        paths = [c[1]["destination_path"] for c in sandbox.upload_file.call_args_list]
        assert "/base/s1/skill.zip" in paths
        assert "/base/s1/scripts/run.py" in paths
        assert "/base/s1/SKILL.md" in paths
