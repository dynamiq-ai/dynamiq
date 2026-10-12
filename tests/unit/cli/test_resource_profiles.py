from types import SimpleNamespace

from click.testing import CliRunner

from dynamiq.cli.commands.context import DynamiqCtx
from dynamiq.cli.commands.resource_profiles import profile
from dynamiq.cli.config import Settings


class StaticApi:
    def __init__(self, profiles: list[dict]):
        self.profiles = profiles

    def get(self, path, **kwargs):
        return SimpleNamespace(status_code=200, json=lambda: {"data": self.profiles})


def test_list_prints_profiles_without_a_description():
    dctx = DynamiqCtx()
    dctx.settings = Settings()
    # The API leaves an unset description out of the response; older versions sent null.
    dctx.api = StaticApi(
        [
            {"id": "p1", "name": "small", "description": "2 CPU, 4 GiB"},
            {"id": "p2", "name": "medium", "description": None},
            {"id": "p3", "name": "large"},
        ]
    )

    result = CliRunner().invoke(profile, ["list"], obj=dctx)

    assert result.exit_code == 0, result.output
    assert "3 resource(s) found." in result.output
    assert "2 CPU, 4 GiB" in result.output
    assert "medium" in result.output
    assert "large" in result.output


def test_list_accepts_database_purpose(cli_ctx, recording_api):
    recording_api.responses[("GET", "/v1/resource-profiles?purpose=database&page_size=100&sort=sort_order")] = {
        "data": [{"id": "p1", "name": "db-small"}]
    }

    result = CliRunner().invoke(profile, ["list", "--purpose", "database"], obj=cli_ctx)

    assert result.exit_code == 0, result.output
    assert recording_api.calls[0][1] == "/v1/resource-profiles?purpose=database&page_size=100&sort=sort_order"
    assert "db-small" in result.output
