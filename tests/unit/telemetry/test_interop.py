"""The warning about LiteLLM's own OpenTelemetry callback."""

import sys
import types

import pytest

from dynamiq.telemetry import interop
from dynamiq.utils.logger import logger


class OpenTelemetry:
    """Stands in for litellm.integrations.opentelemetry.OpenTelemetry."""


class SubclassedOpenTelemetry(OpenTelemetry):
    pass


class OtherLogger:
    pass


@pytest.fixture
def fake_litellm(monkeypatch):
    module = types.ModuleType("litellm")
    module.callbacks = []
    module.success_callback = []
    module.failure_callback = []
    module._async_success_callback = []
    monkeypatch.setitem(sys.modules, "litellm", module)
    monkeypatch.setattr(interop, "_warned", False)
    monkeypatch.setattr(interop, "_searched_lengths", None)
    return module


@pytest.fixture
def warnings_logged(monkeypatch):
    messages = []
    monkeypatch.setattr(logger, "warning", lambda message, *a, **k: messages.append(message))
    return messages


@pytest.mark.parametrize("list_name", ["callbacks", "success_callback", "failure_callback", "_async_success_callback"])
@pytest.mark.parametrize("entry", ["otel", OpenTelemetry(), SubclassedOpenTelemetry()])
def test_warns_for_an_otel_callback(fake_litellm, warnings_logged, list_name, entry):
    getattr(fake_litellm, list_name).append(entry)

    interop.warn_if_litellm_otel_callback()

    assert len(warnings_logged) == 1
    assert "USE_OTEL_LITELLM_REQUEST_SPAN" in warnings_logged[0]
    assert "Remove 'otel'" in warnings_logged[0]


def test_warns_once_per_process(fake_litellm, warnings_logged):
    fake_litellm.callbacks.append("otel")

    for _ in range(5):
        interop.warn_if_litellm_otel_callback()

    assert len(warnings_logged) == 1


def test_other_callbacks_do_not_warn(fake_litellm, warnings_logged):
    fake_litellm.callbacks.extend(["langfuse", OtherLogger(), print])
    fake_litellm.success_callback.append("langfuse")

    interop.warn_if_litellm_otel_callback()

    assert warnings_logged == []


def test_a_callback_registered_later_is_noticed(fake_litellm, warnings_logged):
    interop.warn_if_litellm_otel_callback()
    interop.warn_if_litellm_otel_callback()
    assert warnings_logged == []

    fake_litellm.success_callback.append("otel")
    interop.warn_if_litellm_otel_callback()

    assert len(warnings_logged) == 1


def test_unchanged_lists_are_not_searched_again(fake_litellm, warnings_logged, monkeypatch):
    fake_litellm.callbacks.append("langfuse")
    interop.warn_if_litellm_otel_callback()
    monkeypatch.setattr(interop, "_is_otel_callback", lambda callback: pytest.fail("lists were searched again"))

    interop.warn_if_litellm_otel_callback()


def test_nothing_happens_when_litellm_was_not_imported(monkeypatch, warnings_logged):
    monkeypatch.setattr(interop, "_warned", False)
    monkeypatch.setattr(interop, "_searched_lengths", None)
    monkeypatch.delitem(sys.modules, "litellm", raising=False)

    interop.warn_if_litellm_otel_callback()

    assert "litellm" not in sys.modules
    assert warnings_logged == []


def test_never_raises_on_unexpected_litellm_state(fake_litellm, warnings_logged):
    fake_litellm.callbacks = None
    fake_litellm.success_callback = [object()]

    interop.warn_if_litellm_otel_callback()

    class Hostile:
        def __eq__(self, other):
            raise RuntimeError("boom")

    fake_litellm.failure_callback.append(Hostile())
    interop.warn_if_litellm_otel_callback()
    assert warnings_logged == []
