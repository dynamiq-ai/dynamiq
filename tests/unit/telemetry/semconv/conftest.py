import pytest

from dynamiq.telemetry.types import CaptureSettings


@pytest.fixture
def settings() -> CaptureSettings:
    return CaptureSettings()


@pytest.fixture
def no_content() -> CaptureSettings:
    return CaptureSettings(content_enabled=False)
