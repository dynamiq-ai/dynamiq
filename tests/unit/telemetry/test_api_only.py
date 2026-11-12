"""Tracing with only the OpenTelemetry API installed, which is what the default install has.

This file also runs in the CI job of the default install, so it imports nothing from the SDK.
"""

import importlib.util
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
from opentelemetry import context as context_api
from opentelemetry import trace

import dynamiq
from dynamiq import Workflow
from dynamiq.flows import Flow
from dynamiq.runnables import RunnableConfig, RunnableStatus
from dynamiq.telemetry import propagation, registry
from dynamiq.types.telemetry import TracingConfig
from tests.unit.telemetry.conftest import Echo, Failing

REPO_ROOT = Path(dynamiq.__file__).parent.parent


def two_step_workflow() -> Workflow:
    first = Echo(id="first", name="first")
    second = Echo(id="second", name="second")
    second.depends_on(first)
    return Workflow(flow=Flow(nodes=[first, second]))


def test_default_install_has_the_api_without_the_sdk():
    assert importlib.util.find_spec("opentelemetry.trace") is not None
    if importlib.util.find_spec("opentelemetry.sdk") is not None:
        pytest.skip("the otel extra is installed")
    assert importlib.util.find_spec("opentelemetry.exporter") is None


def test_run_code_imports_the_api_but_never_the_sdk():
    script = textwrap.dedent(
        """
        import json
        import sys

        import dynamiq

        loaded_by_dynamiq = sorted(name for name in sys.modules if name.startswith("opentelemetry"))

        import dynamiq.telemetry.emitter
        import dynamiq.workflow

        print(json.dumps({
            "loaded_by_dynamiq": loaded_by_dynamiq,
            "api": "opentelemetry.trace" in sys.modules,
            "sdk": sorted(name for name in sys.modules if name.startswith("opentelemetry.sdk")),
        }))
        """
    )

    completed = subprocess.run(  # nosec B603
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        timeout=120,
        # With eager imports, importing dynamiq imports the run code, and with it the API.
        env={**os.environ, "DYNAMIQ_EAGER_IMPORTS": ""},
    )

    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout.strip().splitlines()[-1]) == {"loaded_by_dynamiq": [], "api": True, "sdk": []}


def test_workflow_runs_untraced_and_leaves_nothing_behind():
    assert isinstance(trace.get_tracer_provider(), trace.ProxyTracerProvider)
    context_before = context_api.get_current()

    result = two_step_workflow().run(input_data={"value": 1})

    assert result.status == RunnableStatus.SUCCESS
    assert result.output["second"]["output"] == {"value": 1}
    assert registry.size() == 0
    assert context_api.get_current() is context_before


def test_failing_workflow_runs_untraced_and_leaves_nothing_behind():
    failing = Failing(id="failing")
    skipped = Echo(id="skipped")
    skipped.depends_on(failing)

    result = Workflow(flow=Flow(nodes=[failing, skipped])).run(input_data={})

    assert result.status == RunnableStatus.FAILURE
    assert registry.size() == 0


def test_tracing_settings_without_a_provider_change_nothing():
    config = RunnableConfig(tracing=TracingConfig(session_id="session-1", tags=["a"]))

    result = two_step_workflow().run(input_data={"value": 1}, config=config)

    assert result.status == RunnableStatus.SUCCESS
    assert registry.size() == 0


def test_run_inside_a_span_of_the_api_alone_is_untraced():
    # Without an SDK the API hands out spans that record nothing and have no ids to continue.
    with trace.get_tracer("application").start_as_current_span("request"):
        result = two_step_workflow().run(input_data={"value": 1})
        assert propagation.mcp_request_meta() is None

    assert result.status == RunnableStatus.SUCCESS
    assert registry.size() == 0
