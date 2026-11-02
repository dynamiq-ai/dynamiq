from importlib.metadata import version

from dynamiq.callbacks import TracingCallbackHandler


def test_host_lists_the_installed_packages():
    handler = TracingCallbackHandler(installed_pkgs=["pydantic", "Pydantic", "a-package-that-is-not-installed"])

    assert handler.host == {"installed_pkgs": [{"name": "pydantic", "version": version("pydantic")}]}
