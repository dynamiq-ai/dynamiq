"""The hook examples run offline (scripted LLM) and show what they claim."""

import os
import subprocess
import sys

import pytest

EXAMPLES = os.path.join(os.path.dirname(__file__), "..", "..", "..", "examples", "components", "agents", "hooks")


def run_example(name: str, *args: str) -> str:
    env = {k: v for k, v in os.environ.items() if k not in ("DYNAMIQ_TRACE_ACCESS_KEY", "OPENAI_API_KEY")}
    done = subprocess.run(
        [sys.executable, os.path.join(EXAMPLES, name, "main.py"), "--offline", *args],
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
        cwd=os.path.join(EXAMPLES, name),
    )
    assert done.returncode == 0, done.stderr[-2000:]
    return done.stdout


def test_pii_example_the_model_never_sees_the_values_and_the_tool_gets_the_real_address():
    out = run_example("pii_mask_restore")
    assert "status: success" in out
    for secret in ("ann.lee@example.com", "+1 415 555 0100", "boss@example.com"):
        assert f"LLM was sent {secret!r}: False" in out
    assert "answer: Done: I emailed ann.lee@example.com, will call +1 415 555 0100" in out  # user's own values back
    assert "(<EMAIL_2>)" in out  # the account manager's address (from a tool result) stays masked
    assert "[pii] before_tool tool=send-email -> modify" in out


def test_prompt_injection_example():
    out = run_example("prompt_injection")
    assert out.count("status: success") == 3
    assert "blocked: True  blocked_by: input-guard" in out
    assert "[page-guard] after_tool tool=fetch-page -> block" in out
    for secret in ("attacker@example.com", "IGNORE ALL PREVIOUS", "print your system prompt"):
        assert f"LLM was sent {secret!r}: False" in out


def test_tool_policy_example():
    out = run_example("tool_policy_subagent")
    assert "[admin-only-delete] before_tool tool=delete-customer -> block" in out
    assert "[researcher-access] before_tool tool=Researcher -> block" in out
    assert "[search-budget] before_tool tool=web-search -> block" in out
    assert out.count("status: success") == 8
    assert out.count("[refund-approval] before_tool tool=refund-customer -> ask") == 2
    assert "'outcome': 'approved'" in out and "'outcome': 'declined'" in out
    assert out.count("[approval asked]") == 2  # none for the admin
    assert out.count("[researcher-access] before_tool tool=Researcher -> block") == 1  # the viewer, not analyst/admin
    assert out.count("[admin-only-delete] before_tool tool=delete-customer -> block") == 2  # viewer and analyst


def test_transform_and_python_hook_example():
    out = run_example("transform_and_python_hook")
    assert "[numeric-id] before_tool tool=lookup-customer -> block" in out
    # Unconditionally denied tools are hidden; attempted calls fail before before_tool hooks run.
    assert "Delete tool available: False" in out
    assert "Denied tool call rejected: True" in out
    assert out.count("status: success") == 4
    assert "blocked: True  blocked_by: input-guard" in out
    assert "LLM was sent 'SECRET-INTERNAL-NOTE': False" in out
    assert "contact her at [EMAIL]" in out


def test_inherit_scope_context_example():
    out = run_example("inherit_scope_context")
    assert out.count("status: success") == 2
    assert "[tenant-from-caller] before_tool tool=crm-lookup -> modify" in out
    assert "[crm-budget] before_tool tool=crm-lookup -> block" in out  # parent and sub-agent share the budget
    assert "[pii] on_output -> modify" in out
    for secret in ("ann@leak.example", "zq9boss@leak.example"):
        assert f"LLM was sent {secret!r}: False" in out


@pytest.mark.parametrize(
    "name",
    [
        "pii_mask_restore",
        "prompt_injection",
        "tool_policy_subagent",
        "transform_and_python_hook",
        "inherit_scope_context",
    ],
)
def test_every_example_ships_its_dag(name):
    assert os.path.isfile(os.path.join(EXAMPLES, name, "dag.yaml"))
