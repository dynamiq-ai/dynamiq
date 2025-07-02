import click

from dynamiq.cli.client import ApiClient
from dynamiq.cli.config import Settings


class DynamiqCtx:
    def __init__(self) -> None:
        self.settings: Settings | None = None
        self.api: ApiClient | None = None


pass_dctx = click.make_pass_decorator(DynamiqCtx, ensure=True)


def with_api_and_settings(fn):
    """Decorator to inject `api` kwarg after verifying settings."""

    @pass_dctx
    def _wrapper(dctx: DynamiqCtx, *args, **kwargs):
        if dctx.settings is None:
            dctx.settings = Settings.load_settings()
            dctx.api = ApiClient(dctx.settings)
        return fn(*args, api=dctx.api, settings=dctx.settings, **kwargs)

    return _wrapper
