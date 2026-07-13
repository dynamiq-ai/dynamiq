"""Compatibility guards between the `e2b` base SDK and `e2b-code-interpreter`.

`e2b-code-interpreter` reaches into private attributes of the base `e2b` sandbox.
When the base SDK reshuffles those internals, the pair silently resolves to an
incompatible combination and every `run_code` call fails at runtime.

These tests pin that contract without needing E2B_API_KEY or network access, so
they run on every PR rather than only in the credentialed workflows.
"""

import pytest
from e2b.connection_config import ConnectionConfig
from e2b_code_interpreter import Sandbox
from packaging.version import Version


@pytest.fixture
def offline_sandbox() -> Sandbox:
    """A sync Sandbox built without touching the network.

    The sync constructor performs no API calls; it only wires up the filesystem,
    commands and pty subsystems, which is exactly the state `run_code` relies on.
    """
    return Sandbox(
        sandbox_id="test-sandbox",
        envd_version=Version("0.2.0"),
        envd_access_token=None,
        sandbox_domain=None,
        connection_config=ConnectionConfig(api_key="sk-test-not-a-real-key"),
    )


@pytest.mark.unit
def test_code_interpreter_can_build_its_http_client(offline_sandbox: Sandbox):
    """`_client` must be constructible from a sync sandbox.

    Regression: e2b-code-interpreter 2.0.0 built this from `self._transport`, an
    attribute e2b 2.29.0 stopped setting on the sync sandbox (it survives only on
    the async one). Resolving that pair raised, on every run_code call:

        AttributeError: 'Sandbox' object has no attribute '_transport'
    """
    assert offline_sandbox._client is not None


@pytest.mark.unit
def test_code_interpreter_can_build_its_jupyter_url(offline_sandbox: Sandbox):
    """`_jupyter_url` is the other private-attribute bridge into the base SDK."""
    assert offline_sandbox._jupyter_url.endswith(".e2b.app")


@pytest.mark.unit
def test_dynamiq_uses_the_e2b_v2_sandbox_api():
    """Every base-SDK symbol `E2BInterpreterTool` calls must exist on Sandbox.

    Guards against a v1/v2 style break, and against the base SDK renaming the
    surface the tool drives.
    """
    for attribute in (
        "create",
        "connect",
        "kill",
        "run_code",
        "get_host",
        "set_timeout",
        "commands",
        "files",
        "connection_config",
    ):
        assert hasattr(Sandbox, attribute), f"e2b Sandbox is missing {attribute!r}"
