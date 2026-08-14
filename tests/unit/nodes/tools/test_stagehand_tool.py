"""Unit tests for the Stagehand tool's v3 call mapping, clone safety and error taxonomy.

These cover pure logic only — no network, no browser. The shared-browser-session protocol is
covered separately in tests/unit/nodes/agents/test_browser_sharing_wiring.py.
"""

import asyncio
import io
import logging
from unittest.mock import AsyncMock, MagicMock

import pytest
from stagehand import AuthenticationError, InternalServerError, NotFoundError

from dynamiq.connections import Browserbase as BrowserbaseConnection
from dynamiq.nodes.agents.exceptions import ToolExecutionException
from dynamiq.nodes.tools.stagehand import Stagehand, StagehandActionType, StagehandInputSchema


def bare_stagehand(**attrs) -> Stagehand:
    """A Stagehand instance without connection/pydantic init, for pure-logic tests."""
    tool = Stagehand.__new__(Stagehand)
    object.__setattr__(tool, "__pydantic_fields_set__", set())
    object.__setattr__(tool, "__pydantic_extra__", None)
    defaults = {
        "name": "T",
        "id": "id-1",
        "model_name": "anthropic/claude-sonnet-5",
        "fallback_model_name": None,
        "timeout": 3600,
        "client": None,
        "_session_id": None,
        "_stagehand_session": None,
        "_cdp_url": None,
        "_live_view_url": None,
        "_shares_browser_session": False,
        "_browserbase_client": None,
        "_steel_client": None,
        "_steel_browser_session": None,
        "_playwright": None,
        "_pw_browser": None,
        "_pw_page": None,
        "_loop": None,
        "_loop_thread": None,
        "browser_context_id": None,
        "shared_browser_session_timeout": 3600,
    }
    merged = {**defaults, **attrs}
    object.__setattr__(tool, "__pydantic_private__", {k: v for k, v in merged.items() if k.startswith("_")})
    for key, value in merged.items():
        if not key.startswith("_"):
            object.__setattr__(tool, key, value)
    return tool


def _stub_execute_scaffolding(tool: Stagehand) -> None:
    """Neutralize everything execute_async does around the action itself."""

    async def _noop(*args, **kwargs):
        return None

    object.__setattr__(tool, "_acquire_shared_browser", _noop)
    object.__setattr__(tool, "_init_client", _noop)
    object.__setattr__(tool, "run_on_node_execute_run", lambda *a, **k: None)
    object.__setattr__(tool, "is_return_screenshot_bytes_enabled", False)
    object.__setattr__(tool, "is_return_live_view_url_enabled", False)


class _FakeChooserWait:
    """Stand-in for page.expect_file_chooser(...): an async CM exposing an awaitable `value`."""

    def __init__(self, chooser):
        self._chooser = chooser

    async def __aenter__(self):
        return MagicMock(value=asyncio.sleep(0, result=self._chooser))

    async def __aexit__(self, *exc_info):
        return False


class TestCallKwargsMapping:
    """Extra input fields must map onto the parameters each v3 action actually accepts."""

    def test_variables_go_to_options_for_act(self):
        tool = bare_stagehand()
        kwargs = tool._build_call_kwargs({"variables": {"user": "jdoe"}}, StagehandActionType.ACT)
        assert kwargs["options"]["variables"] == {"user": "jdoe"}

    def test_variables_dropped_for_extract(self):
        """The extract API has no options.variables — sending it would be silently ignored."""
        tool = bare_stagehand()
        kwargs = tool._build_call_kwargs({"variables": {"user": "jdoe"}}, StagehandActionType.EXTRACT)
        assert "options" not in kwargs

    def test_model_and_variables_dropped_for_goto(self):
        tool = bare_stagehand()
        kwargs = tool._build_call_kwargs(
            {"model": "openai/gpt-5", "variables": {"a": "b"}, "timeout": 5000}, StagehandActionType.GOTO
        )
        assert kwargs["options"] == {"timeout": 5000}

    def test_schema_passthrough_only_for_extract(self):
        tool = bare_stagehand()
        schema = {"type": "object", "properties": {"title": {"type": "string"}}}
        assert tool._build_call_kwargs({"schema": schema}, StagehandActionType.EXTRACT)["schema"] == schema
        assert "schema" not in tool._build_call_kwargs({"schema": schema}, StagehandActionType.ACT)

    def test_frame_id_is_a_top_level_param(self):
        tool = bare_stagehand()
        kwargs = tool._build_call_kwargs({"frame_id": "frame-1"}, StagehandActionType.OBSERVE)
        assert kwargs["frame_id"] == "frame-1"

    def test_unknown_fields_are_dropped_not_raised(self):
        """A hallucinated field must not abort the agent's whole step."""
        tool = bare_stagehand()
        kwargs = tool._build_call_kwargs({"nonsense": 1, "iframes": True}, StagehandActionType.ACT)
        assert kwargs == {}

    def test_variables_are_kept_for_observe(self):
        """observe takes variables to return %placeholders% instead of literal secrets."""
        tool = bare_stagehand()
        kwargs = tool._build_call_kwargs({"variables": {"password": "hunter2"}}, StagehandActionType.OBSERVE)
        assert kwargs["options"]["variables"] == {"password": "hunter2"}

    def test_selectors_scope_extract_and_observe(self):
        tool = bare_stagehand()
        payload = {"selector": "#main", "ignore_selectors": [".cookie-banner"]}
        for action in (StagehandActionType.EXTRACT, StagehandActionType.OBSERVE):
            options = tool._build_call_kwargs(dict(payload), action)["options"]
            assert options["selector"] == "#main"
            assert options["ignore_selectors"] == [".cookie-banner"]

    def test_selectors_dropped_for_act(self):
        """act has no selector options; forwarding them would be rejected server-side."""
        tool = bare_stagehand()
        assert tool._build_call_kwargs({"selector": "#main"}, StagehandActionType.ACT) == {}

    def test_screenshot_option_is_extract_only(self):
        tool = bare_stagehand()
        assert tool._build_call_kwargs({"screenshot": True}, StagehandActionType.EXTRACT)["options"] == {
            "screenshot": True
        }
        assert tool._build_call_kwargs({"screenshot": True}, StagehandActionType.OBSERVE) == {}

    def test_navigation_options_pass_through_for_goto(self):
        tool = bare_stagehand()
        kwargs = tool._build_call_kwargs(
            {"wait_until": "networkidle", "referer": "https://example.com"}, StagehandActionType.GOTO
        )
        assert kwargs["options"] == {"wait_until": "networkidle", "referer": "https://example.com"}


class TestSessionStartOptions:
    """extra_config feeds sessions.start, whose signature is a fixed keyword list."""

    def test_supported_keys_pass_through(self):
        tool = bare_stagehand(connection=MagicMock(extra_config={"self_heal": False, "verbose": 2}))
        assert tool._session_start_options() == {"self_heal": False, "verbose": 2}

    def test_legacy_keys_are_filtered_out(self):
        """0.4-era keys would raise TypeError inside the SDK, outside our error handling."""
        tool = bare_stagehand(
            connection=MagicMock(extra_config={"enable_caching": True, "api_url": "x", "self_heal": True})
        )
        assert tool._session_start_options() == {"self_heal": True}


class TestCloneSafety:
    """clone() copies private attrs shallowly; a clone must not touch the original's resources."""

    def test_reset_clears_every_live_handle(self):
        tool = bare_stagehand(
            _loop=MagicMock(),
            _loop_thread=MagicMock(),
            client=MagicMock(),
            _stagehand_session=MagicMock(),
            _session_id="sess-1",
            _cdp_url="ws://x",
            _live_view_url="https://lv",
            _browserbase_client=MagicMock(),
            _steel_client=MagicMock(),
            _steel_browser_session=MagicMock(),
            _playwright=MagicMock(),
            _pw_browser=MagicMock(),
            _pw_page=MagicMock(),
            _shares_browser_session=True,
        )

        tool.reset_clone_resources()

        assert tool._loop is None  # init_loop must not stop the ORIGINAL's loop
        assert tool.client is None
        assert tool._stagehand_session is None
        assert tool._session_id is None
        assert tool._browserbase_client is None
        assert tool._steel_client is None
        assert tool._steel_browser_session is None
        assert tool._pw_browser is None
        assert tool._playwright is None
        assert tool._shares_browser_session is False

    def test_reset_runs_before_loop_init(self):
        """Order matters: resetting after init_loop would discard the clone's own loop."""
        names = Stagehand._clone_init_methods_names
        assert names.index("reset_clone_resources") < names.index("init_loop")


def _schema_error() -> InternalServerError:
    return InternalServerError(
        "Error code: 500 - {'success': False, 'message': 'No object generated: response did not match schema.'}",
        response=MagicMock(),
        body=None,
    )


class TestModelFallback:
    """A model whose answer the server cannot coerce fails identically on same-model retry."""

    def test_failed_call_is_retried_on_the_fallback_model(self):
        tool = bare_stagehand(fallback_model_name="anthropic/claude-sonnet-4-6")
        seen = []

        async def call(payload):
            seen.append(payload)
            if len(seen) == 1:
                raise _schema_error()
            return "recovered"

        result = asyncio.run(tool._call_with_model_fallback(call, "observe", {"frame_id": "f1"}))

        assert result == "recovered"
        assert len(seen) == 2
        assert "options" not in seen[0]  # first attempt uses the session's own model
        assert seen[1]["options"]["model"] == "anthropic/claude-sonnet-4-6"
        assert seen[1]["frame_id"] == "f1"  # other params survive the retry

    def test_no_retry_without_a_fallback_configured(self):
        tool = bare_stagehand(fallback_model_name=None)
        calls = []

        async def call(payload):
            calls.append(payload)
            raise _schema_error()

        with pytest.raises(InternalServerError):
            asyncio.run(tool._call_with_model_fallback(call, "observe", {}))
        assert len(calls) == 1

    def test_no_retry_when_fallback_equals_primary(self):
        tool = bare_stagehand(fallback_model_name="anthropic/claude-sonnet-5")
        calls = []

        async def call(payload):
            calls.append(payload)
            raise _schema_error()

        with pytest.raises(InternalServerError):
            asyncio.run(tool._call_with_model_fallback(call, "observe", {}))
        assert len(calls) == 1  # the same model would fail identically

    def test_other_server_errors_are_not_retried(self):
        tool = bare_stagehand(fallback_model_name="anthropic/claude-sonnet-4-6")
        calls = []

        async def call(payload):
            calls.append(payload)
            raise InternalServerError("boom", response=MagicMock(), body=None)

        with pytest.raises(InternalServerError):
            asyncio.run(tool._call_with_model_fallback(call, "act", {}))
        assert len(calls) == 1

    def test_caller_payload_is_not_mutated(self):
        tool = bare_stagehand(fallback_model_name="anthropic/claude-sonnet-4-6")
        payload = {"options": {"timeout": 5000}}

        async def call(p):
            if "model" not in (p.get("options") or {}):
                raise _schema_error()
            return "ok"

        asyncio.run(tool._call_with_model_fallback(call, "observe", payload))
        assert payload == {"options": {"timeout": 5000}}

    def test_upload_retries_inside_the_file_chooser_wait(self):
        """upload acts under the hood, so it needs the fallback too — and the retry's click is
        the one that must be caught by the still-open chooser waiter."""
        tool = bare_stagehand(fallback_model_name="anthropic/claude-sonnet-4-6")
        chooser = MagicMock(set_files=AsyncMock())
        page = MagicMock(expect_file_chooser=lambda **kw: _FakeChooserWait(chooser))
        attempts = []

        async def act(*, input, **params):
            attempts.append(params)
            if len(attempts) == 1:
                raise _schema_error()
            return MagicMock(data=MagicMock(result=MagicMock(model_dump=lambda: {"success": True})))

        session = MagicMock(act=act)
        tool._stagehand_session = session
        _stub_execute_scaffolding(tool)
        object.__setattr__(tool, "_ensure_page", AsyncMock(return_value=page))

        upload = io.BytesIO(b"col_a,col_b\n1,2\n")
        upload.name = "report.csv"
        result = asyncio.run(
            tool.execute_async(
                StagehandInputSchema(action_type=StagehandActionType.UPLOAD, instruction="click Upload", files=upload)
            )
        )

        assert len(attempts) == 2
        assert attempts[1]["options"]["model"] == "anthropic/claude-sonnet-4-6"
        assert result["content"] == {"success": True}
        chooser.set_files.assert_awaited_once()


class TestCrossProviderFallbackWarning:
    """One model_api_key serves both models, so a fallback from another provider cannot authenticate."""

    def _tool(self, **kwargs) -> Stagehand:
        connection = BrowserbaseConnection(
            browserbase_api_key="bb-key",
            browserbase_project_id="proj-1",
            model_api_key=kwargs.pop("model_api_key", "openai-key"),
        )
        return Stagehand(connection=connection, is_postponed_component_init=True, **kwargs)

    def test_warns_when_providers_differ(self, caplog):
        with caplog.at_level(logging.WARNING):
            self._tool(model_name="openai/gpt-4o", fallback_model_name="anthropic/claude-sonnet-4-6")
        assert "different provider" in caplog.text

    def test_silent_when_providers_match(self, caplog):
        with caplog.at_level(logging.WARNING):
            self._tool(model_name="anthropic/claude-sonnet-5", fallback_model_name="anthropic/claude-sonnet-4-6")
        assert "different provider" not in caplog.text

    def test_silent_without_a_provider_key(self, caplog):
        """No key means the Browserbase Model Gateway, which can serve both providers."""
        with caplog.at_level(logging.WARNING):
            self._tool(
                model_api_key=None,
                model_name="openai/gpt-4o",
                fallback_model_name="anthropic/claude-sonnet-4-6",
            )
        assert "different provider" not in caplog.text


class TestPageResolution:
    """Stagehand runs server-side and may have moved to a tab we never saw."""

    def _connected_browser(self, pages):
        context = MagicMock(pages=pages)
        return MagicMock(contexts=[context], is_connected=MagicMock(return_value=True))

    def test_newest_open_page_wins_over_the_previous_handle(self):
        """A popup opened by act must be the one we screenshot / go back on / upload through."""
        first, popup = MagicMock(), MagicMock()
        first.is_closed.return_value = False
        popup.is_closed.return_value = False
        tool = bare_stagehand(
            _playwright=MagicMock(),
            _pw_browser=self._connected_browser([first, popup]),
            _pw_page=first,
        )

        assert asyncio.run(tool._ensure_page()) is popup

    def test_closed_pages_are_skipped(self):
        live, closed = MagicMock(), MagicMock()
        live.is_closed.return_value = False
        closed.is_closed.return_value = True
        tool = bare_stagehand(_playwright=MagicMock(), _pw_browser=self._connected_browser([live, closed]))

        assert asyncio.run(tool._ensure_page()) is live


class TestErrorTaxonomy:
    """Agents retry recoverable errors; hopeless ones must not be marked recoverable."""

    def _run_failing_action(self, tool, exc):
        session = MagicMock()

        async def _raise(**kwargs):
            raise exc

        session.extract = _raise
        tool._stagehand_session = session
        _stub_execute_scaffolding(tool)
        data = StagehandInputSchema(action_type=StagehandActionType.EXTRACT, instruction="get the title")
        with pytest.raises(ToolExecutionException) as info:
            asyncio.run(tool.execute_async(data))
        return info.value

    def test_auth_error_is_not_recoverable(self):
        tool = bare_stagehand(_session_id="sess-1")
        err = self._run_failing_action(tool, AuthenticationError("bad key", response=MagicMock(), body=None))
        assert err.recoverable is False

    def test_missing_session_is_recoverable_and_clears_state(self):
        """A dead session must be forgotten so the retry starts a fresh one."""
        tool = bare_stagehand(_session_id="sess-1", _live_view_url="https://lv", _cdp_url="ws://x")
        err = self._run_failing_action(tool, NotFoundError("gone", response=MagicMock(), body=None))
        assert err.recoverable is True
        assert tool._session_id is None
        assert tool._stagehand_session is None
        assert tool._live_view_url is None

    def test_missing_session_releases_the_steel_browser(self):
        """Steel owns the browser separately: leaving it running would strand a paid session."""
        steel_client = MagicMock()
        steel_client.sessions.release = AsyncMock()
        tool = bare_stagehand(
            _session_id="steel-1",
            _steel_client=steel_client,
            _steel_browser_session=MagicMock(id="steel-1"),
        )

        self._run_failing_action(tool, NotFoundError("gone", response=MagicMock(), body=None))

        steel_client.sessions.release.assert_awaited_once_with("steel-1")
        assert tool._steel_browser_session is None
