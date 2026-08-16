"""The CLI credentials file must not be readable by other users on the machine.

``Path.write_text`` leaves the mode to the umask, which is 0022 almost everywhere and
lands an API key at 0644.
"""

import importlib
import os
import stat

import pytest

pytestmark = pytest.mark.skipif(os.name == "nt", reason="POSIX file modes")


@pytest.fixture
def permissive_umask():
    """Run under the standard permissive umask, which is what makes this bug bite.

    The process umask has to be set for real — rebinding ``os.umask`` would not change
    what the kernel applies to ``open()``.
    """
    previous = os.umask(0o022)
    try:
        yield
    finally:
        os.umask(previous)


@pytest.fixture
def cli_config(tmp_path, monkeypatch, permissive_umask):
    """Reload the module so its module-level paths point inside tmp_path."""
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))

    import dynamiq.cli.config as config

    return importlib.reload(config)


def _mode(path) -> int:
    return stat.S_IMODE(os.stat(path).st_mode)


@pytest.fixture
def saved(cli_config):
    settings = cli_config.Settings(
        org_id="org", project_id="proj", api_key="dq-secret-value", api_host="https://api.example.invalid"
    )
    settings.save_settings()
    return cli_config


class TestCredentialsFileMode:
    def test_credentials_file_is_owner_only(self, saved):
        assert _mode(saved._CREDS_FILE_PATH) == 0o600

    def test_credentials_file_is_not_group_or_world_readable(self, saved):
        mode = _mode(saved._CREDS_FILE_PATH)
        assert not mode & stat.S_IRGRP
        assert not mode & stat.S_IROTH

    def test_parent_directory_is_owner_only(self, saved):
        assert _mode(saved._CREDS_FILE_PATH.parent) == 0o700

    def test_contents_are_still_correct(self, saved):
        import json

        data = json.loads(saved._CREDS_FILE_PATH.read_text())
        assert data["api_key"] == "dq-secret-value"
        assert data["api_host"] == "https://api.example.invalid"

    def test_settings_round_trip(self, saved):
        loaded = saved.Settings.load_settings()
        assert loaded.api_key == "dq-secret-value"
        assert loaded.org_id == "org"

    def test_rewriting_an_existing_wide_open_file_tightens_it(self, saved):
        """An upgrade must repair a file an older version already wrote at 0644."""
        os.chmod(saved._CREDS_FILE_PATH, 0o644)
        assert _mode(saved._CREDS_FILE_PATH) == 0o644

        saved.Settings(api_key="rotated", api_host="https://api.example.invalid").save_settings()

        assert _mode(saved._CREDS_FILE_PATH) == 0o600

    def test_rewrite_truncates_rather_than_appends(self, saved):
        import json

        saved.Settings(api_key="short", api_host="https://h").save_settings()
        data = json.loads(saved._CREDS_FILE_PATH.read_text())
        assert data["api_key"] == "short"

    def test_reading_repairs_a_file_an_older_version_left_exposed(self, saved):
        """Existing installs must be tightened on the next load, not only on the next save."""
        os.chmod(saved._CREDS_FILE_PATH, 0o644)
        os.chmod(saved._CREDS_FILE_PATH.parent, 0o755)

        loaded = saved.Settings.load_settings()

        assert loaded.api_key == "dq-secret-value", "repair must not disturb the contents"
        assert _mode(saved._CREDS_FILE_PATH) == 0o600
        assert _mode(saved._CREDS_FILE_PATH.parent) == 0o700

    def test_non_secret_config_file_is_untouched(self, saved):
        """config.json holds no credentials; its handling is deliberately unchanged."""
        assert saved._CONFIG_FILE_PATH.exists()
        import json

        assert json.loads(saved._CONFIG_FILE_PATH.read_text())["org_id"] == "org"
