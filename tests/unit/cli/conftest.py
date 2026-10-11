import json

import pytest

from dynamiq.cli import config as cli_config

DYNAMIQ_ENV_VARS = (
    "DYNAMIQ_API_BASE_URL",
    "DYNAMIQ_API_HOST",
    "DYNAMIQ_API_TOKEN",
    "DYNAMIQ_API_KEY",
    "DYNAMIQ_ORG_ID",
    "DYNAMIQ_PROJECT_ID",
)


class ConfigDir:
    """The CLI's config and credentials files, redirected into a temp dir."""

    def __init__(self, root):
        self.config_path = root / "dynamiq" / "config.json"
        self.creds_path = root / "dynamiq" / "credentials.json"

    def write(self, config: dict | None = None, creds: dict | None = None) -> None:
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        if config is not None:
            self.config_path.write_text(json.dumps(config))
        if creds is not None:
            self.creds_path.write_text(json.dumps(creds))

    def config(self) -> dict:
        return json.loads(self.config_path.read_text())

    def creds(self) -> dict:
        return json.loads(self.creds_path.read_text())


@pytest.fixture
def config_dir(tmp_path, monkeypatch) -> ConfigDir:
    """Point the CLI at empty config files and an environment without Dynamiq vars."""
    for name in DYNAMIQ_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    files = ConfigDir(tmp_path)
    monkeypatch.setattr(cli_config, "_CONFIG_FILE_PATH", files.config_path)
    monkeypatch.setattr(cli_config, "_CREDS_FILE_PATH", files.creds_path)
    return files
