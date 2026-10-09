from click.testing import CliRunner

from dynamiq.cli.commands.config import config
from dynamiq.cli.commands.context import DynamiqCtx

HOST = "https://api.example.test"


def run(*args, input=None):
    return CliRunner(mix_stderr=False).invoke(config, list(args), input=input, obj=DynamiqCtx())


def test_config_prompt_enter_keeps_stored_values_and_skips_env_token(config_dir, monkeypatch):
    config_dir.write(config={"org_id": "org-1"}, creds={"api_host": HOST})
    monkeypatch.setenv("DYNAMIQ_API_TOKEN", "secret-from-env")
    monkeypatch.setenv("DYNAMIQ_API_BASE_URL", "https://env.example.test")

    result = run(input="\n\n")

    assert result.exit_code == 0, result.output
    assert config_dir.creds() == {"api_host": HOST}
    assert config_dir.config() == {"org_id": "org-1"}


def test_config_prompt_saves_typed_key(config_dir, monkeypatch):
    monkeypatch.setenv("DYNAMIQ_API_TOKEN", "secret-from-env")

    result = run(input=f"{HOST}\ntyped-key\n")

    assert result.exit_code == 0, result.output
    assert config_dir.creds() == {"api_host": HOST, "api_key": "typed-key"}


def test_config_show_masks_token_and_names_sources(config_dir, monkeypatch):
    config_dir.write(config={"org_id": "org-1"}, creds={"api_key": "tok-file-0123456789"})
    monkeypatch.setenv("DYNAMIQ_PROJECT_ID", "proj-env")

    result = run("show")

    assert result.exit_code == 0, result.stderr
    assert "tok-file-0123456789" not in result.stdout
    assert "DYNAMIQ API KEY: tok-... (credentials file)" in result.stdout
    assert "DYNAMIQ API HOST: https://api.getdynamiq.ai (default)" in result.stdout
    assert "DYNAMIQ ORG ID: org-1 (config file)" in result.stdout
    assert "DYNAMIQ PROJECT ID: proj-env (env DYNAMIQ_PROJECT_ID)" in result.stdout
