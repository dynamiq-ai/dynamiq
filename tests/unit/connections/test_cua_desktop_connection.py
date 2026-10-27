import sys
import types

import pytest

from dynamiq.connections import CuaDesktop
from dynamiq.connections.connections import CuaDockerImage, CuaProviderType


class RecordingComputer:
    """Stands in for cua-computer's Computer and keeps the arguments it was built with."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs


@pytest.fixture
def computer_module(monkeypatch):
    """cua-computer is an optional extra; the connection imports `computer.Computer` when it connects."""
    module = types.ModuleType("computer")
    module.Computer = RecordingComputer
    monkeypatch.setitem(sys.modules, "computer", module)


def test_cloud_computer_authenticates_with_the_connection_key(computer_module, monkeypatch):
    monkeypatch.setenv("CUA_API_KEY", "key-of-another-account")

    computer = CuaDesktop(api_key="connection-key", computer_name="sandbox-1").connect()

    assert computer.kwargs == {
        "os_type": "linux",
        "provider_type": "cloud",
        "name": "sandbox-1",
        "api_key": "connection-key",
    }


def test_docker_computer_gets_no_api_key(computer_module):
    computer = CuaDesktop(provider_type=CuaProviderType.DOCKER, image=CuaDockerImage.XFCE).connect()

    assert computer.kwargs == {
        "os_type": "linux",
        "provider_type": "docker",
        "image": "trycua/cua-xfce:latest",
    }
