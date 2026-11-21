import pytest

from dynamiq.utils.local_files import (
    LOCAL_FILE_PATHS_ALLOWED_ENV_VAR,
    LocalFilePathsNotAllowedError,
    ensure_local_file_paths_allowed,
    local_file_paths_allowed,
)


def test_local_file_paths_are_allowed_when_the_variable_is_unset(monkeypatch):
    monkeypatch.delenv(LOCAL_FILE_PATHS_ALLOWED_ENV_VAR, raising=False)

    assert local_file_paths_allowed()
    ensure_local_file_paths_allowed(["notes.txt"])


@pytest.mark.parametrize("value", ["true", "TRUE", " true ", "1", "yes", "on"])
def test_local_file_paths_are_allowed_when_the_variable_turns_them_on(monkeypatch, value):
    monkeypatch.setenv(LOCAL_FILE_PATHS_ALLOWED_ENV_VAR, value)

    assert local_file_paths_allowed()
    ensure_local_file_paths_allowed(["notes.txt"])


@pytest.mark.parametrize("value", ["false", "0", "no", "off", "", "ture"])
def test_local_file_paths_are_refused_for_any_other_value(monkeypatch, value):
    monkeypatch.setenv(LOCAL_FILE_PATHS_ALLOWED_ENV_VAR, value)

    assert not local_file_paths_allowed()
    with pytest.raises(LocalFilePathsNotAllowedError, match=LOCAL_FILE_PATHS_ALLOWED_ENV_VAR):
        ensure_local_file_paths_allowed(["notes.txt"])


def test_no_paths_are_not_refused(monkeypatch):
    monkeypatch.setenv(LOCAL_FILE_PATHS_ALLOWED_ENV_VAR, "false")

    ensure_local_file_paths_allowed([])
