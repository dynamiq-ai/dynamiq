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
