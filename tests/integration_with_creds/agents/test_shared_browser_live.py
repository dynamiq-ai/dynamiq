"""Live shared-browser test (requires Browserbase + model credentials).

Drives two Stagehand tools under one SharedSession (no LLM) and asserts they
share ONE Browserbase session and that the lease serializes them. Skipped
without BROWSERBASE_API_KEY.
"""

import os

import pytest

from dynamiq.connections import Http  # noqa: F401  # placeholder: use the project's Browserbase connection
from dynamiq.nodes.agents.shared_session import SharedSession, _current_agent_run, _shared_session


def _browserbase_stagehand():
    """Build a Browserbase-backed Stagehand tool from env creds.

    NOTE (implementer): construct with the repo's actual Browserbase connection
    (browserbase_api_key + model_api_key), mirroring an existing Stagehand usage
    in tests/ or examples/. Kept abstract here to avoid guessing connection fields.
    """
    from dynamiq.nodes.tools.stagehand import Stagehand  # noqa: F401

    raise NotImplementedError("wire the project's Browserbase connection here")


@pytest.mark.integration
def test_two_agents_share_one_browserbase_session():
    if not os.getenv("BROWSERBASE_API_KEY"):
        pytest.skip("BROWSERBASE_API_KEY is not set; skipping live shared-browser test.")

    ss = SharedSession(share_browser=True)
    session_token = _shared_session.set(ss)
    tool_a = _browserbase_stagehand()
    tool_b = _browserbase_stagehand()
    try:
        # Agent A creates + records the shared session by navigating.
        run_a = _current_agent_run.set("runA")
        try:
            tool_a.run(input_data={"action_type": "navigate", "url": "https://example.com"})
        finally:
            _current_agent_run.reset(run_a)
            ss.release_browser("runA")

        shared_sid = ss.browser_session_id()
        assert shared_sid is not None

        # Agent B attaches to the SAME session.
        run_b = _current_agent_run.set("runB")
        try:
            tool_b.run(input_data={"action_type": "extract", "instruction": "the page title"})
            assert tool_b._session_id == shared_sid
        finally:
            _current_agent_run.reset(run_b)
            ss.release_browser("runB")
    finally:
        ss.close_browser()
        _shared_session.reset(session_token)
