from click.testing import CliRunner

import dynamiq.cli.commands.dataset as module
from dynamiq.cli.commands.dataset import dataset


def test_dataset_download_writes_utf8_file(cli_ctx, recording_api, tmp_path, monkeypatch):
    real_open = open

    def cp1252_default_open(file, mode="r", *args, encoding=None, **kwargs):
        return real_open(file, mode, *args, encoding=encoding or "cp1252", **kwargs)

    monkeypatch.setattr(module, "open", cp1252_default_open, raising=False)
    text = '{"question": "日本語 – résumé"}'
    recording_api.get = lambda path, **kw: type("R", (), {"status_code": 200, "text": text})()
    out = tmp_path / "version.json"

    result = CliRunner().invoke(dataset, ["download", "v1", "--out", str(out)], obj=cli_ctx)

    assert result.exit_code == 0, result.output
    assert out.read_text(encoding="utf-8") == text
