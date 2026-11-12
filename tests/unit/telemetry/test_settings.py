"""Resolving the tracer and the capture settings of a run tree from its config and the environment."""

import logging

import pytest
from opentelemetry import trace

from dynamiq.runnables import RunnableConfig
from dynamiq.telemetry import semconv, settings
from dynamiq.telemetry.types import CaptureSettings
from dynamiq.types.telemetry import ContentCaptureMode, TracingConfig
from dynamiq.utils.user_agent import VERSION

sdk_trace = pytest.importorskip("opentelemetry.sdk.trace")

NEW_VARIABLE = "OTEL_INSTRUMENTATION_GENAI_CAPTURE_CONTENT"
OLD_VARIABLE = "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT"
SETTINGS_VARIABLES = (
    NEW_VARIABLE,
    OLD_VARIABLE,
    "DYNAMIQ_TRACING_ENABLED",
    "DYNAMIQ_TRACING_BINARY_CONTENT_ENABLED",
    "DYNAMIQ_TRACING_NODE_CONFIG_ENABLED",
    "DYNAMIQ_TRACING_CONTENT_MAX_LENGTH",
    "OTEL_SDK_DISABLED",
)


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch):
    """Keeps the developer's shell out of the tests, and lets each test see the once-only warnings."""
    for variable in SETTINGS_VARIABLES:
        monkeypatch.delenv(variable, raising=False)
    monkeypatch.setattr(settings, "_warned_variables", set())


def config_with(**tracing_fields) -> RunnableConfig:
    return RunnableConfig(tracing=TracingConfig(**tracing_fields))


def test_defaults_capture_content_but_no_binary_and_no_node_config(caplog):
    with caplog.at_level(logging.WARNING):
        resolved = settings.resolve_settings(RunnableConfig())

    assert resolved == CaptureSettings(
        content_enabled=True, binary_content_enabled=False, node_config_enabled=False, content_max_length=None
    )
    assert settings.resolve_settings(None) == resolved
    # Reading variables that are not set is not worth a warning on every run.
    assert caplog.records == []


@pytest.mark.parametrize(
    ("value", "content_enabled"),
    [
        ("SPAN_ONLY", True),
        ("SPAN_AND_EVENT", True),
        ("true", True),
        ("1", True),
        ("span_only", True),
        ("NO_CONTENT", False),
        ("EVENT_ONLY", False),
        ("false", False),
        ("0", False),
        ("No_Content", False),
        ("  NO_CONTENT  ", False),
        ("", True),
    ],
)
@pytest.mark.parametrize("variable", [NEW_VARIABLE, OLD_VARIABLE])
def test_content_capture_follows_the_environment(monkeypatch, variable, value, content_enabled):
    monkeypatch.setenv(variable, value)

    assert settings.resolve_settings(RunnableConfig()).content_enabled is content_enabled


def test_older_content_variable_is_read_only_when_the_newer_one_is_unset(monkeypatch):
    monkeypatch.setenv(OLD_VARIABLE, "false")
    assert settings.resolve_settings(None).content_enabled is False

    monkeypatch.setenv(NEW_VARIABLE, "SPAN_ONLY")
    assert settings.resolve_settings(None).content_enabled is True


def test_unrecognised_content_mode_captures_content_and_warns_once(monkeypatch, caplog):
    monkeypatch.setenv(NEW_VARIABLE, "everything")
    monkeypatch.setenv(OLD_VARIABLE, "false")

    with caplog.at_level(logging.WARNING):
        first = settings.resolve_settings(None)
        second = settings.resolve_settings(None)

    assert first.content_enabled is True
    assert second.content_enabled is True
    assert [record.getMessage() for record in caplog.records if NEW_VARIABLE in record.getMessage()] == [
        f"{NEW_VARIABLE}='everything' is not a content capture mode; content is captured."
    ]


@pytest.mark.parametrize(
    ("mode", "content_enabled"), [(ContentCaptureMode.NO_CONTENT, False), (ContentCaptureMode.SPAN_ONLY, True)]
)
def test_content_mode_of_the_config_wins_over_the_environment(monkeypatch, mode, content_enabled):
    monkeypatch.setenv(NEW_VARIABLE, "NO_CONTENT" if content_enabled else "SPAN_ONLY")

    assert settings.resolve_settings(config_with(content_capture=mode)).content_enabled is content_enabled


def test_optional_capture_is_read_from_the_environment(monkeypatch):
    monkeypatch.setenv("DYNAMIQ_TRACING_BINARY_CONTENT_ENABLED", "true")
    monkeypatch.setenv("DYNAMIQ_TRACING_NODE_CONFIG_ENABLED", "1")
    monkeypatch.setenv("DYNAMIQ_TRACING_CONTENT_MAX_LENGTH", "2000")

    resolved = settings.resolve_settings(RunnableConfig())

    assert resolved.binary_content_enabled is True
    assert resolved.node_config_enabled is True
    assert resolved.content_max_length == 2000


def test_config_fields_win_over_the_environment(monkeypatch):
    monkeypatch.setenv("DYNAMIQ_TRACING_BINARY_CONTENT_ENABLED", "true")
    monkeypatch.setenv("DYNAMIQ_TRACING_NODE_CONFIG_ENABLED", "true")
    monkeypatch.setenv("DYNAMIQ_TRACING_CONTENT_MAX_LENGTH", "2000")

    resolved = settings.resolve_settings(
        config_with(binary_content_enabled=False, node_config_enabled=False, content_max_length=50)
    )

    assert resolved.binary_content_enabled is False
    assert resolved.node_config_enabled is False
    assert resolved.content_max_length == 50


def test_node_config_is_not_captured_when_content_is_off():
    resolved = settings.resolve_settings(
        config_with(node_config_enabled=True, content_capture=ContentCaptureMode.NO_CONTENT)
    )

    assert resolved.node_config_enabled is False


@pytest.mark.parametrize("value", ["many", "0", "-5", "1.5"])
def test_invalid_max_length_truncates_nothing_and_warns_once(monkeypatch, caplog, value):
    monkeypatch.setenv("DYNAMIQ_TRACING_CONTENT_MAX_LENGTH", value)

    with caplog.at_level(logging.WARNING):
        assert settings.resolve_settings(None).content_max_length is None
        assert settings.resolve_settings(None).content_max_length is None

    assert len([record for record in caplog.records if "CONTENT_MAX_LENGTH" in record.getMessage()]) == 1


def test_unrecognised_flag_keeps_its_default_and_warns_once(monkeypatch, caplog):
    monkeypatch.setenv("DYNAMIQ_TRACING_BINARY_CONTENT_ENABLED", "yes please")

    with caplog.at_level(logging.WARNING):
        assert settings.resolve_settings(None).binary_content_enabled is False
        assert settings.resolve_settings(None).binary_content_enabled is False

    assert len([record for record in caplog.records if "BINARY_CONTENT" in record.getMessage()]) == 1


def test_run_level_values_come_from_the_config():
    resolved = settings.resolve_settings(
        config_with(
            session_id="session-1",
            user_id="user-1",
            source_id="source-1",
            tags=["a", "b"],
            metadata={"tier": "pro", "attempt": 2},
        )
    )

    assert (resolved.session_id, resolved.user_id, resolved.source_id) == ("session-1", "user-1", "source-1")
    assert resolved.tags == ("a", "b")
    assert resolved.metadata == {"tier": "pro", "attempt": 2}


def test_no_tracer_without_an_sdk_provider():
    assert isinstance(trace.get_tracer_provider(), trace.ProxyTracerProvider)

    assert settings.resolve_tracer(None) is None
    assert settings.resolve_tracer(RunnableConfig()) is None
    assert settings.resolve_tracer(config_with(session_id="session-1")) is None
    assert settings.resolve_tracer(config_with(tracer_provider=trace.NoOpTracerProvider())) is None


def test_tracer_of_the_config_provider_names_the_dynamiq_scope():
    provider = sdk_trace.TracerProvider()

    tracer = settings.resolve_tracer(config_with(tracer_provider=provider))

    scope = tracer.start_span("probe").instrumentation_scope
    assert scope.name == "dynamiq"
    assert scope.version == VERSION
    assert scope.schema_url == semconv.SEMCONV_SCHEMA_URL
    assert dict(scope.attributes) == semconv.scope_attributes()


def test_tracer_is_built_once_per_provider(mocker):
    provider = sdk_trace.TracerProvider()
    get_tracer = mocker.spy(provider, "get_tracer")
    config = config_with(tracer_provider=provider)

    first = settings.resolve_tracer(config)
    second = settings.resolve_tracer(config_with(tracer_provider=provider))

    assert first is second
    assert get_tracer.call_count == 1
    assert settings.resolve_tracer(config_with(tracer_provider=sdk_trace.TracerProvider())) is not first


def test_no_tracer_when_tracing_is_disabled_on_the_config():
    assert settings.resolve_tracer(config_with(enabled=False, tracer_provider=sdk_trace.TracerProvider())) is None


@pytest.mark.parametrize(("value", "traced"), [("false", False), ("0", False), ("true", True), ("", True)])
def test_environment_switch_disables_tracing(monkeypatch, value, traced):
    monkeypatch.setenv("DYNAMIQ_TRACING_ENABLED", value)

    tracer = settings.resolve_tracer(config_with(tracer_provider=sdk_trace.TracerProvider()))

    assert (tracer is not None) is traced


def test_no_tracer_from_a_disabled_sdk(monkeypatch):
    monkeypatch.setenv("OTEL_SDK_DISABLED", "true")

    assert settings.resolve_tracer(config_with(tracer_provider=sdk_trace.TracerProvider())) is None


def test_provider_that_cannot_be_cached_still_gives_a_tracer():
    class UnhashableProvider:
        # The cache is keyed by provider, so it cannot hold this one.
        __hash__ = None

        def __init__(self):
            self.inner = sdk_trace.TracerProvider()

        def get_tracer(self, *args, **kwargs):
            return self.inner.get_tracer(*args, **kwargs)

    provider = UnhashableProvider()

    assert settings.resolve_tracer(config_with(tracer_provider=provider)) is not None
    assert settings.resolve_tracer(config_with(tracer_provider=provider)) is not None
