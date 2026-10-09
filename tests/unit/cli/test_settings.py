import pytest

from dynamiq.cli.config import DYNAMIQ_BASE_URL, Settings

HOST = "https://api.example.test"


def test_sources_from_files(config_dir):
    config_dir.write(config={"org_id": "org-1", "project_id": "proj-1"}, creds={"api_key": "tok-file"})

    settings = Settings.load_settings()

    assert settings.org_id == "org-1"
    assert settings.api_key == "tok-file"
    assert settings.sources == {
        "org_id": "config file",
        "project_id": "config file",
        "api_host": "default",
        "api_key": "credentials file",
    }
    assert settings.api_host == DYNAMIQ_BASE_URL


def test_nothing_configured(config_dir):
    settings = Settings.load_settings()

    assert settings.source_of("api_key") == "not set"
    assert settings.source_of("org_id") == "not set"
    assert settings.source_of("api_host") == "default"


def test_null_in_file_is_not_set(config_dir):
    config_dir.write(config={"org_id": None, "project_id": None})

    assert Settings.load_settings().source_of("org_id") == "not set"


@pytest.mark.parametrize(
    ("env", "field", "source"),
    [
        ({"DYNAMIQ_API_TOKEN": "tok"}, "api_key", "env DYNAMIQ_API_TOKEN"),
        ({"DYNAMIQ_API_KEY": "tok"}, "api_key", "env DYNAMIQ_API_KEY"),
        ({"DYNAMIQ_API_TOKEN": "tok", "DYNAMIQ_API_KEY": "other"}, "api_key", "env DYNAMIQ_API_TOKEN"),
        ({"DYNAMIQ_API_BASE_URL": HOST}, "api_host", "env DYNAMIQ_API_BASE_URL"),
        ({"DYNAMIQ_API_HOST": HOST}, "api_host", "env DYNAMIQ_API_HOST"),
        ({"DYNAMIQ_ORG_ID": "o"}, "org_id", "env DYNAMIQ_ORG_ID"),
        ({"DYNAMIQ_PROJECT_ID": "p"}, "project_id", "env DYNAMIQ_PROJECT_ID"),
    ],
)
def test_env_source_names_the_variable(config_dir, monkeypatch, env, field, source):
    for name, value in env.items():
        monkeypatch.setenv(name, value)

    assert Settings.load_settings().source_of(field) == source


def test_env_beats_files(config_dir, monkeypatch):
    config_dir.write(config={"project_id": "proj-file"}, creds={"api_key": "tok-file"})
    monkeypatch.setenv("DYNAMIQ_PROJECT_ID", "proj-env")

    settings = Settings.load_settings()

    assert settings.project_id == "proj-env"
    assert settings.source_of("project_id") == "env DYNAMIQ_PROJECT_ID"
    assert settings.source_of("api_key") == "credentials file"


def test_constructed_settings_keep_working():
    settings = Settings(api_host=HOST, api_key="tok")

    assert settings.base_url == HOST
    assert settings.source_of("api_key") == "set explicitly"
    assert settings.source_of("org_id") == "not set"
    with pytest.raises(KeyError):
        settings.source_of("nope")


def test_warns_when_env_overrides_a_different_stored_value(config_dir, monkeypatch, capsys):
    config_dir.write(config={"project_id": "proj-file", "org_id": "org-1"}, creds={"api_key": "tok-file"})
    monkeypatch.setenv("DYNAMIQ_PROJECT_ID", "proj-env")
    monkeypatch.setenv("DYNAMIQ_API_TOKEN", "tok-env")

    Settings.load_settings()

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "DYNAMIQ_PROJECT_ID overrides project_id from config file" in captured.err
    assert "DYNAMIQ_API_TOKEN overrides api_key from credentials file" in captured.err
    assert "org_id" not in captured.err
    # Never echo the values themselves.
    assert "tok-env" not in captured.err and "tok-file" not in captured.err
    assert captured.err.count("overrides") == 2


def test_no_warning_when_env_matches_disk(config_dir, monkeypatch, capsys):
    config_dir.write(config={"project_id": "same"}, creds={"api_key": "tok"})
    monkeypatch.setenv("DYNAMIQ_PROJECT_ID", "same")
    monkeypatch.setenv("DYNAMIQ_API_KEY", "tok")

    Settings.load_settings()

    assert capsys.readouterr().err == ""


def test_no_warning_when_nothing_stored(config_dir, monkeypatch, capsys):
    monkeypatch.setenv("DYNAMIQ_API_TOKEN", "tok-env")

    Settings.load_settings()

    assert capsys.readouterr().err == ""


def test_warning_can_be_silenced(config_dir, monkeypatch, capsys):
    config_dir.write(config={"project_id": "proj-file"})
    monkeypatch.setenv("DYNAMIQ_PROJECT_ID", "proj-env")

    Settings.load_settings(warn=False)

    assert capsys.readouterr().err == ""
