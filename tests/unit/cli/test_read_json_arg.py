import click
import pytest

import dynamiq.cli.commands.workflow as module
from dynamiq.cli.commands.workflow import read_json_arg


def _cp1252_default_open(monkeypatch):
    real_open = open

    def fake_open(file, mode="r", *args, encoding=None, **kwargs):
        return real_open(file, mode, *args, encoding=encoding or "cp1252", **kwargs)

    monkeypatch.setattr(module, "open", fake_open, raising=False)


def test_read_json_arg_reads_file_as_utf8(tmp_path, monkeypatch):
    _cp1252_default_open(monkeypatch)
    path = tmp_path / "items.json"
    path.write_text('[{"q": "日本語 – résumé"}]', encoding="utf-8")

    assert read_json_arg(f"@{path}") == [{"q": "日本語 – résumé"}]


def test_read_json_arg_non_utf8_file_is_click_exception(tmp_path):
    path = tmp_path / "items.json"
    path.write_bytes('["résumé"]'.encode("cp1252"))

    with pytest.raises(click.ClickException, match="not valid UTF-8"):
        read_json_arg(f"@{path}")
