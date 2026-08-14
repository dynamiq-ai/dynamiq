import asyncio
import io
import os
import threading
import time
import zipfile
from enum import Enum
from typing import Any, ClassVar, Coroutine, Literal
from urllib.parse import urlparse

from browserbase import AsyncBrowserbase
from playwright.async_api import async_playwright
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, field_validator, model_validator
from stagehand import AsyncStagehand, AuthenticationError, InternalServerError, NotFoundError, PermissionDeniedError
from steel import AsyncSteel

from dynamiq.connections import Browserbase, SteelBrowser, SteelBrowserEnvironment
from dynamiq.connections.managers import ConnectionManager
from dynamiq.nodes import NodeGroup
from dynamiq.nodes.agents.exceptions import ToolExecutionException
from dynamiq.nodes.agents.shared_session import SharedSession, _current_agent_run, active_browser_session
from dynamiq.nodes.node import ConnectionNode, ensure_config
from dynamiq.nodes.tools.utils import guess_mime_type_from_bytes, sanitize_filename
from dynamiq.runnables import RunnableConfig
from dynamiq.types.cancellation import CanceledException, check_cancellation
from dynamiq.utils.logger import logger

DESCRIPTION_STAGEHAND = """## Stagehand Tool
### Description
A headless browser automation and observation tool designed to navigate, interact with,
and extract structured data from web pages using natural language instructions.

### Parameters
- `action_type`: Must be one of the following:
  - `goto`: Navigate to the specified URL.
  - `observe`: Return a list of candidate DOM elements based on the instruction.
  - `extract`: Extract structured data as described in the instruction.
  - `act`: Perform a user-specified action (e.g., click a button, select an option).
  - `upload`: Perform an action that opens a file chooser and upload the provided file.
  - `go_back`: Go to the previous page.
- `instruction`: A natural language prompt specifying the action to perform. Required for all actions except `goto`.
- `brief`: A short summary of what this tool call is doing. Required for all actions.
- `url`: The web address to navigate to. Required only when `action_type` is `goto`.

Optional fields passed in the input (beyond the ones above) are forwarded to Stagehand:
- `schema`: (extract only) JSON Schema describing the structure of the data to extract.
- `variables`: (act, upload, observe) A mapping of variables to substitute in the instruction, e.g.
  {"username": "jdoe"} with instruction "type %username% into the login field".
- `selector`: (extract, observe) CSS selector scoping the call to one part of the page.
- `ignore_selectors`: (extract, observe) List of CSS selectors to exclude, e.g. cookie banners or
  navigation chrome that would otherwise crowd out the content you asked for.
- `frame_id`: Target a specific frame by id. Iframes are otherwise handled automatically.

### Usage Examples
1. Navigate to a web page:
   ```json
   {
     "action_type": "goto",
     "brief": "Open the target website homepage",
     "url": "https://example.com"
   }
    ````

2. Observe candidate elements:

   ```json
   {
     "action_type": "observe",
     "instruction": "Find all clickable links on the homepage",
     "brief": "Identify clickable links on the homepage"
   }
   ```

3. Extract data:

   ```json
   {
     "action_type": "extract",
     "instruction": "Get the list of product names and prices",
     "brief": "Extract product names and prices"
   }
   ```

4. Perform an action:

   ```json
   {
     "action_type": "act",
     "instruction": "Click the 'Add to Cart' button for the first product",
     "brief": "Add the first product to cart"
   }
   ```

5. Perform an action whey you expect to open file chooser:

   ```json
   {
     "action_type": "upload",
     "instruction": "Click the 'Upload Report' button for the first product",
     "brief": "Upload the report file via file chooser",
     "files": "test.csv"
   }
   ```

### Tips

- !!! Always divide complex tasks into clear, isolated actions. For example:
a. Wait for the page to load
b. Enter text into the input field
c. Click the search button
d. Extract data from the results
- Example:
    - **Incorrect**: Enter `'text'` in the search bar and press Enter.
    - **Correct**:
      - **Call 1**: Enter `'text'` into the search bar.
      - **Call 2**: Press `Enter` on the search bar.
- Entering the text into search field and clicking the search button always should be two different steps.
- Use clear, specific instructions.
- `observe` is useful for debugging or when planning a follow-up action.
- Make sure you use 'upload' action instead of 'act' when you expect that file chooser will be opened.
"""

# Input fields that map to per-call Stagehand options rather than top-level params. The option set
# differs per action (stagehand 3.22.x session_{act,observe,extract,navigate}_params.Options), so a
# field is accepted only for the actions whose API actually carries it:
#   * variables — act/upload substitute values into the instruction; observe only exposes the NAMES
#     so suggested actions come back as %placeholders% instead of literal secrets. extract has none.
#   * selector / ignore_selectors — scope or prune the accessibility tree observe and extract read.
#   * screenshot — extract only: include a viewport image in the extraction call.
#   * referer / wait_until — navigate only.
_OPTION_FIELDS_BY_ACTION = {
    "act": ("model", "timeout", "variables"),
    "upload": ("model", "timeout", "variables"),
    "observe": ("ignore_selectors", "model", "selector", "timeout", "variables"),
    "extract": ("ignore_selectors", "model", "screenshot", "selector", "timeout"),
    "goto": ("referer", "timeout", "wait_until"),
    "go_back": (),
}

# Every field that is an option for at least one action. A field valid elsewhere but not for the
# action at hand is dropped with a warning naming the action, rather than silently forwarded.
_OPTION_FIELDS = tuple(sorted({field for fields in _OPTION_FIELDS_BY_ACTION.values() for field in fields}))

# Seconds any single teardown step may take. Cleanup runs from close()/__del__, where an unbounded
# wait can hang interpreter shutdown on a loop thread that is already frozen.
_CLEANUP_TIMEOUT = 30.0

# Milliseconds to wait for a history navigation to settle.
_HISTORY_NAV_TIMEOUT_MS = 30_000

# Marker of a server-side failure to turn the model's answer into the shape act/observe/extract
# require. Retrying the same model is pointless (the server decodes deterministically: the retry
# reproduces the same invalid answer) — only a different model recovers it.
_SCHEMA_MISMATCH_MARKER = "did not match schema"

# Keyword params accepted by stagehand 3.22.x sessions.start; anything else in extra_config would
# raise a TypeError outside the tool's error handling.
_SESSION_START_OPTIONS = (
    "browser",
    "browserbase_session_create_params",
    "browserbase_session_id",
    "dom_settle_timeout_ms",
    "experimental",
    "self_heal",
    "system_prompt",
    "verbose",
)


def end_browserbase_session(api_key: str, project_id: str, session_id: str) -> None:
    """End a Browserbase session by id, independent of any Stagehand client.

    Ends the run's shared session: the creating tool may be gone. Ending also persists the
    session's Context for the next run.
    """
    from browserbase import Browserbase as BrowserbaseSDK

    BrowserbaseSDK(api_key=api_key).sessions.update(session_id, status="REQUEST_RELEASE", project_id=project_id)
    logger.info(f"Ended shared Browserbase session {session_id}")


class StagehandActionType(str, Enum):
    ACT = "act"
    EXTRACT = "extract"
    OBSERVE = "observe"
    GOTO = "goto"
    UPLOAD = "upload"
    GOBACK = "go_back"


class StagehandInputSchema(BaseModel):
    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)

    action_type: StagehandActionType = Field(
        ...,
        description="""
    Specifies the type of interaction with the web page:

    - `act`: Perform an action on the current page (e.g., click, fill, select).
    - `extract`: Retrieve structured data from the page based on the given instruction.
    - `observe`: Return a list of candidate DOM elements for potential interaction.
    - `goto`: Navigate to the specified URL.
    """,
    )

    instruction: str | None = Field(
        None,
        description="""
    A natural language instruction describing the action, data to extract, or elements to observe.
    Required for all action types except `goto`.
    """,
    )
    brief: str = Field(
        default="Using browser to perform the action.",
        description="""
        A brief description of the action to perform. Example: "Click the 'Add to Cart' button for the first product"
        """,
    )

    url: str | None = Field(
        None,
        description="""
    The target URL to navigate to. Required only when `action_type` is `goto`.
    """,
    )

    files: str | io.BytesIO | None = Field(
        default=None,
        description="Name of a file to upload in opened dialog window.",
        json_schema_extra={"map_from_storage": True},
    )

    @field_validator("files", mode="before")
    @classmethod
    def files_validator(cls, input_data: str | io.BytesIO) -> io.BytesIO:
        """Validate and process files."""
        if isinstance(input_data, str):
            raise ToolExecutionException(
                "File path provided for upload action. Please provide a file to upload.", recoverable=True
            )
        return input_data


class Stagehand(ConnectionNode):
    group: Literal[NodeGroup.TOOLS] = NodeGroup.TOOLS
    name: str = "stagehand-browser"
    description: str = DESCRIPTION_STAGEHAND
    connection: Browserbase | SteelBrowser
    model_name: str
    fallback_model_name: str | None = Field(
        default=None,
        description=(
            "Model used to retry a single act/observe/extract call when the primary model's answer "
            "is rejected by Stagehand's structured-output validation. Recommended whenever the "
            "primary model is newer than the Stagehand server: retrying the same model cannot "
            "recover (the server decodes deterministically), a different one does. Must come from "
            "the same provider as model_name unless the connection routes through the Browserbase "
            "Model Gateway (model_api_key unset), because one provider key serves both models."
        ),
    )
    is_return_screenshot_bytes_enabled: bool = Field(
        default=False, description="Whether to capture and return a screenshot as bytes after act/goto actions."
    )
    is_return_live_view_url_enabled: bool = Field(
        default=False, description="Whether to return the live view URL in every response."
    )
    browser_context_id: str | None = Field(
        default=None,
        description=(
            "Existing Browserbase Context to load instead of creating a fresh one. Carries state to "
            "a LATER run: pass a stable id (e.g. one per end user) so the user stays logged in across "
            "runs. Leave unset for per-run throwaway state."
        ),
    )
    shared_browser_session_timeout: int = Field(
        default=3600,
        description=(
            "Seconds before the shared browser session auto-ends. Overrides the Browserbase project "
            "default because the session must outlast the whole run."
        ),
    )

    _browserbase_client: AsyncBrowserbase | None = PrivateAttr(default=None)
    _steel_client: AsyncSteel | None = PrivateAttr(default=None)
    _steel_browser_session: Any | None = PrivateAttr(default=None)
    input_schema: ClassVar[type[StagehandInputSchema]] = StagehandInputSchema
    timeout: int = 3600
    is_files_allowed: bool = True
    _session_id: str | None = PrivateAttr(default=None)
    _live_view_url: str | None = PrivateAttr(default=None)
    # Stagehand v3 is a pure API client: AI operations run server-side against the browser
    # session, while direct page control (screenshots, history navigation, file choosers) goes
    # through our own Playwright connection to the same browser over CDP.
    _stagehand_session: Any | None = PrivateAttr(default=None)
    _cdp_url: str | None = PrivateAttr(default=None)
    _playwright: Any | None = PrivateAttr(default=None)
    _pw_browser: Any | None = PrivateAttr(default=None)
    _pw_page: Any | None = PrivateAttr(default=None)
    _loop = PrivateAttr(default=None)
    _loop_thread = PrivateAttr(default=None)
    _call_lock = PrivateAttr(default_factory=threading.Lock)
    # True while driving the run's shared session: close() must then detach, not end it.
    _shares_browser_session: bool = PrivateAttr(default=False)

    # reset_clone_resources MUST run first: it clears the shallow-copied loop reference so the
    # clone's init_loop does not stop the ORIGINAL instance's running loop.
    _clone_init_methods_names: ClassVar[list[str]] = ["reset_clone_resources", "init_loop", "init_call_lock"]

    @model_validator(mode="after")
    def warn_on_cross_provider_fallback(self):
        """Warn when the fallback model cannot authenticate with the configured provider key.

        ``model_api_key`` becomes a single ``x-model-api-key`` header covering every call on the
        session, so a fallback from another provider authenticates with the wrong key and turns a
        recoverable schema mismatch into a hard auth failure. Only the Browserbase Model Gateway
        (``model_api_key`` unset) can serve two providers.
        """
        fallback = self.fallback_model_name
        if not fallback or not getattr(self.connection, "model_api_key", None):
            return self
        primary_provider, _, _ = self.model_name.partition("/")
        fallback_provider, _, _ = fallback.partition("/")
        if primary_provider and fallback_provider and primary_provider != fallback_provider:
            logger.warning(
                f"Tool {self.name} - {self.id}: fallback_model_name '{fallback}' is from a different "
                f"provider than model_name '{self.model_name}', but the connection supplies a single "
                f"model_api_key. The fallback will fail to authenticate — use a '{primary_provider}/' "
                f"model, or leave model_api_key unset to route through the Browserbase Model Gateway."
            )
        return self

    def reset_clone_resources(self):
        """Detach a clone from the original's live resources.

        ``clone()`` copies private attrs shallowly, so without this a clone would share the
        original's event loop (and stop it via ``init_loop``), drive the original's browser
        session, and end it from ``close()``/``__del__``. A clone starts with no live handles
        and builds its own on first use.
        """
        self._loop = None
        self._loop_thread = None
        self.client = None
        self._stagehand_session = None
        self._session_id = None
        self._cdp_url = None
        self._live_view_url = None
        self._browserbase_client = None
        self._steel_client = None
        self._steel_browser_session = None
        self._playwright = None
        self._pw_browser = None
        self._pw_page = None
        self._shares_browser_session = False

    def _is_steel_browser_connection(self) -> bool:
        """Check if the current connection is an instance of Steel Browser connection."""
        return isinstance(self.connection, SteelBrowser)

    def is_clone_safe_for_parallel(self) -> bool:
        """Don't clone under browser sharing: a clone would open a SECOND session on the shared
        Context, diverging and clobbering on persist. One instance keeps one session per turn;
        ``execute`` serializes the calls."""
        if active_browser_session() is None or self._is_steel_browser_connection():
            return True
        return False

    async def _acquire_shared_browser(self) -> "SharedSession | None":
        """Take control of the shared page for this agent's turn; return the session, else None.

        Control is held from the first Stagehand call until the turn ends (or it delegates).
        Browserbase only in P3 (Steel is a fast-follow).
        """
        ss = active_browser_session()
        if ss is None:
            return None
        if self._is_steel_browser_connection():
            return None  # Steel sharing is a fast-follow
        run_key = _current_agent_run.get()
        if run_key is None:
            # The agent's finally releases under the real run key; a substituted key would never
            # match, locking the page for the whole run. Running unshared is the safe degradation.
            logger.warning(
                "Tool %s - %s: no _current_agent_run set (tool is running outside Agent.execute); "
                "skipping browser sharing and using an isolated session.",
                self.name,
                self.id,
            )
            return None
        # Off the event loop: the blocking wait must not stall the loop serving the current holder.
        try:
            await asyncio.to_thread(ss.acquire_page_control, run_key)
        except TimeoutError as exc:
            # Surface as a tool observation, not a crash. Not recoverable: a retry just blocks again
            # — the topology must change (don't browse and delegate in one parallel batch).
            raise ToolExecutionException(str(exc), recoverable=False) from exc
        return ss

    async def _join_shared_browser(self, ss: "SharedSession") -> None:
        """Attach to the run's shared session, or prepare to create it if we are first.

        Sets ``_session_id`` so ``_init_client`` resumes it; marks sharing so ``close()`` detaches.
        """
        self._shares_browser_session = True
        shared_sid = ss.browser_session_id()
        if shared_sid:
            self._session_id = shared_sid
            return

        # Fresh shared run with no published session: anything we still hold belongs to a PREVIOUS
        # run and was ended at its teardown — drop it, or needs_start would resume a dead session.
        if self._session_id is not None:
            self._session_id = None
            self._stagehand_session = None
            self._live_view_url = None
            self._cdp_url = None

        # First to browse: settle the persistent Context (cross-run state) so the session we create
        # loads it and writes it back on end.
        if ss.browser_context_id() is None:
            if self.browser_context_id:
                ss.adopt_browser_context_id(self.browser_context_id)
                logger.info(f"Tool {self.name} - {self.id}: reusing browser context {self.browser_context_id}")
            else:
                self._browserbase_client = self._browserbase_client or AsyncBrowserbase(
                    api_key=self.connection.browserbase_api_key
                )
                response = await self._browserbase_client.contexts.create(
                    project_id=self.connection.browserbase_project_id
                )
                ss.adopt_browser_context_id(response.id)
                logger.info(f"Tool {self.name} - {self.id}: created shared browser context {response.id}")

    def _record_shared_browser_session(self, ss: "SharedSession") -> None:
        """After init, publish the session we created so every later agent attaches to it."""
        if not self._session_id:
            return
        adopted = ss.adopt_browser_session_id(self._session_id)
        if adopted != self._session_id:
            # Lost the race: another session is the shared one, so end ours and fall in behind it.
            logger.warning(
                "Tool %s - %s: created session %s but %s is already shared; ending ours.",
                self.name,
                self.id,
                self._session_id,
                adopted,
            )
            end_browserbase_session(
                self.connection.browserbase_api_key,
                self.connection.browserbase_project_id,
                self._session_id,
            )
            self._session_id = adopted
            # Our Stagehand session wraps the session we just ended — a later call re-attaches.
            self._stagehand_session = None
            # The live view URL points at the ended session; clear it so re-attach fetches the
            # winner's URL instead of publishing a dead debugger link.
            self._live_view_url = None
            return
        # Ours is the run's session: register an end that does not depend on this tool surviving.
        api_key = self.connection.browserbase_api_key
        project_id = self.connection.browserbase_project_id
        session_id = self._session_id
        ss.register_browser_end(lambda: end_browserbase_session(api_key, project_id, session_id))

    def _apply_shared_browser_config(self, params: dict | None, context_id: str | None) -> dict:
        """Configure the Browserbase session we create as the run's shared session.

        ``persist`` writes the Context back on end (cross-run state). ``keep_alive`` stops it dying
        when the creating agent disconnects, and an explicit timeout stops the project default
        cutting a run short — both because one session now spans the whole run.
        """
        params = dict(params or {})
        browser_settings = dict(params.get("browser_settings") or {})
        if context_id:
            browser_settings["context"] = {"id": context_id, "persist": True}
        params["browser_settings"] = browser_settings
        params.setdefault("project_id", self.connection.browserbase_project_id)
        params.setdefault("keep_alive", True)
        params.setdefault("timeout", self.shared_browser_session_timeout)
        return params

    def _loop_alive(self) -> bool:
        return self._loop is not None and not self._loop.is_closed()

    def _detach_shared_browser(self) -> None:
        """Drop our connection to the shared session WITHOUT ending it.

        The Stagehand v3 client holds no browser resources of its own — ``client.close()`` only
        closes its HTTP client and never ends browser sessions — so a full detach is closing our
        Playwright CDP connection plus the API client, leaving the Browserbase session running.
        The owner's teardown ends it once.
        """
        if not self._loop_alive():
            return
        if self._pw_browser is not None or self._playwright is not None:
            try:
                self._run_in_loop(self._close_playwright(), timeout=_CLEANUP_TIMEOUT)
            except Exception as exc:
                logger.warning(f"Stagehand detach from shared session failed: {exc}")
        if self.client is not None and hasattr(self.client, "close"):
            try:
                self._run_in_loop(self.client.close(), timeout=_CLEANUP_TIMEOUT)
            except Exception as exc:
                logger.warning(f"Stagehand client close on detach failed: {exc}")

    def _get_steel_browser_cdp_url(self) -> str:
        """CDP URL of the Steel session, with the API key appended for cloud sessions."""
        cdp_url = self._steel_browser_session.websocket_url
        if self.connection.environment == SteelBrowserEnvironment.CLOUD and self.connection.api_key:
            separator = "&" if "?" in cdp_url else "?"
            cdp_url = f"{cdp_url}{separator}apiKey={self.connection.api_key}"
        return cdp_url

    async def _ensure_steel_browser_session(self):
        """Check if the Steel Browser session exists and create it if it doesn't."""
        if self._steel_browser_session is None:
            self._steel_browser_session = await self._steel_client.sessions.create(**self.connection.session_config)
            if hasattr(self._steel_browser_session, "id"):
                logger.debug(f"Steel session created: {self._steel_browser_session.id}")
            if hasattr(self._steel_browser_session, "session_viewer_url"):
                logger.debug(f"Steel session viewer: {self._steel_browser_session.session_viewer_url}")
        return self._steel_browser_session

    async def _release_steel_browser_session(self) -> None:
        """Best-effort release of the Steel session, then drop the handle.

        No-op off the Steel path. Releasing is best-effort because the caller is already handling
        a failure: an unreachable Steel API must not mask the original error.
        """
        session = self._steel_browser_session
        if session is None:
            return
        self._steel_browser_session = None
        session_id = getattr(session, "id", None)
        if not session_id or self._steel_client is None:
            return
        try:
            await self._steel_client.sessions.release(session_id)
        except Exception as exc:
            logger.warning(f"Failed to release Steel session {session_id}: {exc}")

    async def _list_steel_browser_session_files(self, session_id: str) -> list:
        """List all files in the Steel session's filesystem."""
        listing = await self._steel_client.sessions.files.list(session_id)
        files = getattr(listing, "data", None) or []
        logger.debug(f"Steel session files: {files}")
        return list(files)

    async def _download_steel_browser_session_files_archive(self, session_id: str) -> bytes:
        """Download the archive of the files in the Steel session."""
        response = await self._steel_client.sessions.files.download_archive(session_id)
        return await response.read()

    def init_components(self, connection_manager: ConnectionManager | None = None):
        super().init_components(connection_manager)
        if self._is_steel_browser_connection():
            if self.connection.environment == SteelBrowserEnvironment.CLOUD:
                self._steel_client = AsyncSteel(steel_api_key=self.connection.api_key)
            elif self.connection.environment == SteelBrowserEnvironment.SELF_HOSTED:
                self._steel_client = AsyncSteel(base_url=self.connection.base_url)
        else:
            self._browserbase_client = AsyncBrowserbase(api_key=self.connection.browserbase_api_key)
        self.init_loop()

    def init_call_lock(self):
        """Give this instance its own call lock.

        ``clone()`` copies private attrs shallowly, so without this a clone shares the original's
        lock and independent parallel calls needlessly serialize.
        """
        self._call_lock = threading.Lock()

    def close_loop(self):
        if self._loop and not self._loop.is_closed():
            self._loop.call_soon_threadsafe(self._loop.stop)
        self._loop = None
        self._loop_thread = None

    def init_loop(self):
        """Initialize a persistent event loop running in a background thread."""
        self.close_loop()

        self._loop = asyncio.new_event_loop()
        self._loop_thread = threading.Thread(target=self._run_loop, daemon=True)
        self._loop_thread.start()
        # Wait for loop to be ready
        future = asyncio.run_coroutine_threadsafe(asyncio.sleep(0), self._loop)
        future.result()

    def _run_loop(self):
        """Run the event loop in the background thread."""
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def _run_in_loop(self, coro: Coroutine, timeout: float | None = None):
        """Run a coroutine in the persistent event loop.

        ``timeout`` (seconds) bounds the wait — cleanup paths pass one so ``close()``/``__del__``
        cannot hang forever on a frozen loop (e.g. during interpreter shutdown).
        """
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        try:
            return future.result(timeout)
        except TimeoutError:
            future.cancel()
            raise

    async def upload_files(self, files: list[io.BytesIO]):
        if self._is_steel_browser_connection():
            session = await self._ensure_steel_browser_session()
            if not hasattr(session, "id"):
                raise ToolExecutionException("Steel session missing required 'id' attribute", recoverable=False)
            for file_obj in files:
                if not hasattr(file_obj, "name"):
                    file_obj.name = "uploaded_file"
                result = await self._steel_client.sessions.files.upload(
                    session.id, file=(file_obj.name, file_obj.getvalue())
                )
                logger.debug(f"Uploaded {file_obj.name} to Steel session: {result}")
            return

        self._browserbase_client = self._browserbase_client or AsyncBrowserbase(
            api_key=self.connection.browserbase_api_key
        )
        for file_obj in files:
            if not hasattr(file_obj, "name"):
                file_obj.name = "uploaded_file"

            result = await self._browserbase_client.sessions.uploads.create(self._session_id, file=file_obj)
            logger.debug(f"Uploaded {file_obj.name}: {result}")

    def _session_start_options(self) -> dict[str, Any]:
        """Session start parameters supplied via the connection (self_heal, system_prompt, ...).

        Filtered to the keywords sessions.start actually accepts: a stray or 0.4-era key would
        otherwise raise a TypeError outside the tool's error handling.
        """
        supplied = dict(self.connection.extra_config or {})
        options = {k: v for k, v in supplied.items() if k in _SESSION_START_OPTIONS}
        dropped = sorted(set(supplied) - set(options))
        if dropped:
            logger.warning(
                f"Tool {self.name} - {self.id}: ignoring extra_config keys not supported by "
                f"Stagehand session start: {dropped}"
            )
        return options

    async def _init_client(
        self,
        shared_context_id: str | None = None,
        create_shared_session: bool = False,
    ):
        """Initialize the Stagehand API client and start (or resume) its browser session.

        ``create_shared_session``: we are first to browse in a shared-browser run, so the session
        we create is configured to outlive our turn and to load ``shared_context_id`` (prior-run state).
        """
        if self._is_steel_browser_connection():
            if self._steel_client is None:
                # Recreate Steel client if it was cleaned up during close()
                if self.connection.environment == SteelBrowserEnvironment.CLOUD:
                    self._steel_client = AsyncSteel(steel_api_key=self.connection.api_key)
                elif self.connection.environment == SteelBrowserEnvironment.SELF_HOSTED:
                    self._steel_client = AsyncSteel(base_url=self.connection.base_url)
            await self._ensure_steel_browser_session()
            if (
                not hasattr(self._steel_browser_session, "websocket_url")
                or not self._steel_browser_session.websocket_url
            ):
                raise ToolExecutionException(
                    "Steel session missing required 'websocket_url' attribute or has invalid value", recoverable=False
                )
            if self.client is None:
                anthropic_base_url = os.environ.get("ANTHROPIC_BASE_URL", "")
                if (
                    self.model_name.startswith("anthropic/")
                    and anthropic_base_url
                    and not anthropic_base_url.rstrip("/").endswith("/v1")
                ):
                    # The bundled server inherits this env var and hands it verbatim to the AI SDK,
                    # whose Anthropic provider expects the base URL to INCLUDE /v1 (unset works —
                    # the provider then defaults to https://api.anthropic.com/v1).
                    logger.warning(
                        "Tool %s - %s: ANTHROPIC_BASE_URL=%s lacks the /v1 suffix; the local Stagehand "
                        "server will call the Anthropic API at the wrong path and model calls will fail "
                        "with 404. Set ANTHROPIC_BASE_URL=https://api.anthropic.com/v1 or unset it.",
                        self.name,
                        self.id,
                        anthropic_base_url,
                    )
                # The bundled local Stagehand server drives the Steel browser over CDP.
                self.client = AsyncStagehand(
                    server="local",
                    model_api_key=self.connection.model_api_key,
                    timeout=self.timeout,
                    local_ready_timeout_s=30.0,
                )
            if self._stagehand_session is None:
                self._cdp_url = self._get_steel_browser_cdp_url()
                self._stagehand_session = await self.client.sessions.start(
                    model_name=self.model_name,
                    browser={"type": "local", "cdp_url": self._cdp_url},
                    **self._session_start_options(),
                )
                self._session_id = getattr(self._steel_browser_session, "id", None)
        else:
            if self.client is None:
                self.client = AsyncStagehand(
                    browserbase_api_key=self.connection.browserbase_api_key,
                    # None routes model calls through the Browserbase Model Gateway.
                    model_api_key=self.connection.model_api_key or None,
                    timeout=self.timeout,
                )
                self._browserbase_client = self._browserbase_client or AsyncBrowserbase(
                    api_key=self.connection.browserbase_api_key
                )
            # (Re)start when there is no live session, or when the run's shared session was
            # published by another agent and ours must attach to it instead.
            needs_start = self._stagehand_session is None or (
                self._session_id is not None and self._stagehand_session.data.session_id != self._session_id
            )
            if needs_start:
                start_kwargs = self._session_start_options()
                if create_shared_session:
                    start_kwargs["browserbase_session_create_params"] = self._apply_shared_browser_config(
                        start_kwargs.get("browserbase_session_create_params"), shared_context_id
                    )
                if self._session_id:
                    start_kwargs["browserbase_session_id"] = self._session_id
                else:
                    # Creating a fresh session: honor the connection's explicit project id (v3 keys
                    # are project-scoped, but an explicitly configured project must win).
                    create_params = dict(start_kwargs.get("browserbase_session_create_params") or {})
                    create_params.setdefault("project_id", self.connection.browserbase_project_id)
                    start_kwargs["browserbase_session_create_params"] = create_params
                self._stagehand_session = await self.client.sessions.start(
                    model_name=self.model_name,
                    **start_kwargs,
                )
                self._session_id = self._stagehand_session.data.session_id
                self._cdp_url = self._stagehand_session.data.cdp_url
                # A new session means any previous CDP page handle is stale.
                await self._close_playwright()

        # Fetch live view URL only when enabled (once per session)
        if self.is_return_live_view_url_enabled and not self._live_view_url:
            await self._fetch_live_view_url()

    async def _get_connect_cdp_url(self) -> str:
        """CDP endpoint of the live browser session, for our own Playwright connection."""
        if self._is_steel_browser_connection():
            return self._get_steel_browser_cdp_url()
        if self._cdp_url:
            return self._cdp_url
        # Resumed sessions may not carry a cdp_url in the start response; ask Browserbase.
        session = await self._browserbase_client.sessions.retrieve(self._session_id)
        return session.connect_url

    async def _ensure_page(self):
        """Playwright page attached over CDP to the same browser Stagehand drives.

        Needed only for operations the Stagehand API does not cover: file choosers, history
        navigation and screenshots.

        The CONNECTION is cached but the page is re-resolved on every call. Stagehand runs
        server-side and may have moved to a tab we never saw (a link opening in a new window, a
        popup): a cached handle would then screenshot, go back on, or wait for a file chooser on
        the tab the user has already left. Re-resolving costs nothing — the driver keeps
        ``context.pages`` locally.
        """
        if self._playwright is None:
            self._playwright = await async_playwright().start()
        if self._pw_browser is None or not self._pw_browser.is_connected():
            self._pw_browser = await self._playwright.chromium.connect_over_cdp(await self._get_connect_cdp_url())
        context = self._pw_browser.contexts[0] if self._pw_browser.contexts else await self._pw_browser.new_context()
        open_pages = [page for page in context.pages if not page.is_closed()]
        self._pw_page = open_pages[-1] if open_pages else await context.new_page()
        return self._pw_page

    async def _close_playwright(self) -> None:
        """Tear down our CDP connection without touching the remote browser session."""
        for attr in ("_pw_browser", "_playwright"):
            resource = getattr(self, attr)
            if resource is None:
                continue
            try:
                if attr == "_pw_browser":
                    await resource.close()
                else:
                    await resource.stop()
            except Exception as exc:
                logger.debug(f"Stagehand playwright cleanup ({attr}) failed: {exc}")
        self._pw_page = None
        self._pw_browser = None
        self._playwright = None

    async def _fetch_live_view_url(self) -> None:
        """Fetch and store the live view URL for the current session."""
        try:
            if self._is_steel_browser_connection():
                # Steel Browser: debug_url is the embeddable live view (session_viewer_url is the
                # auth-gated dashboard page, kept only as a fallback).
                if self._steel_browser_session:
                    self._live_view_url = getattr(self._steel_browser_session, "debug_url", None) or getattr(
                        self._steel_browser_session, "session_viewer_url", None
                    )
            else:
                # Browserbase: fetch from debug API
                if self._session_id and self._browserbase_client:
                    debug_info = await self._browserbase_client.sessions.debug(self._session_id)
                    if hasattr(debug_info, "debugger_fullscreen_url"):
                        self._live_view_url = debug_info.debugger_fullscreen_url
                    elif hasattr(debug_info, "debuggerFullscreenUrl"):
                        self._live_view_url = debug_info.debuggerFullscreenUrl

            if self._live_view_url:
                logger.debug(f"Live view URL: {self._live_view_url}")
        except Exception as e:
            logger.warning(f"Could not fetch live view URL: {e}")

    def _run_async(self, coro) -> Any:
        """Run coroutine in the persistent event loop."""
        return self._run_in_loop(coro)

    async def _take_screenshot(self) -> io.BytesIO | None:
        """Capture a screenshot of the current page.

        Returns:
            BytesIO object containing the screenshot PNG, or None if capture fails.
        """
        try:
            page = await self._ensure_page()
            screenshot_bytes = await page.screenshot()

            page_host = ""
            try:
                page_url = page.url
                page_host = urlparse(page_url).hostname or "unknown"
                page_host = page_host.replace(".", "_")
            except Exception:
                page_host = "unknown"

            timestamp = int(time.time())
            fobj = io.BytesIO(screenshot_bytes)
            fobj.name = f"screenshot_{page_host}_{timestamp}.png"
            fobj.content_type = "image/png"
            return fobj
        except Exception as e:
            logger.warning(f"Stagehand screenshot capture failed: {e}")
            return None

    def execute(
        self, input_data: StagehandInputSchema, config: RunnableConfig | None = None, **kwargs
    ) -> dict[str, Any]:
        """
        Executes the tool synchronously with the provided input.

        Always dispatches onto a dedicated background event loop to avoid cross-loop issues.

        Args:
            input_data (StagehandInputSchema): Input data for the tool execution.
            config (RunnableConfig, optional): Configuration for the runnable, including callbacks.
            **kwargs: Additional arguments passed to the execution context.

        Returns:
            dict[str, Any]: A dictionary containing the tool's output.
        """
        # Serialize calls: under browser sharing parallel calls are not cloned, so they would race
        # on one client/page.
        with self._call_lock:
            # close() tears the loop down at each turn's end under sharing; a later turn rebuilds it.
            if self._loop is None or self._loop.is_closed():
                self.init_loop()
            return self._run_async(self.execute_async(input_data, config, **kwargs))

    async def get_downloads_via_sdk(
        self,
        session_id: str,
        max_retry_sec: int = 20,
        config: RunnableConfig | None = None,
    ) -> bytes:
        """
        Retrieve the downloads archive for the given session.
        Retries until content is non-empty or timeout.
        """
        if self._is_steel_browser_connection():
            files = await self._list_steel_browser_session_files(session_id)
            if files:
                return await self._download_steel_browser_session_files_archive(session_id)
            else:
                return b""

        elapsed = 0
        interval = 2

        while elapsed < max_retry_sec:
            check_cancellation(config)
            response = await self._browserbase_client.sessions.downloads.list(session_id)
            data = await response.read()

            if data and len(data) > 0:
                return data

            await asyncio.sleep(interval)
            elapsed += interval

        raise RuntimeError("Failed to retrieve downloads via SDK within time limit")

    def extract_files_from_zip(self, zip_bytes: bytes) -> list[io.BytesIO]:
        """Extracts files from ZIP bytes with sanitized filenames to prevent Zip Slip attacks."""
        files = []
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            for idx, name in enumerate(zf.namelist()):
                # Skip directory entries
                if name.endswith("/"):
                    continue
                safe_name = sanitize_filename(filename=name, default=f"extracted_file_{idx}")
                with zf.open(name) as file:
                    bio = io.BytesIO(file.read())
                    bio.name = safe_name
                    files.append(bio)

        return files

    def _build_call_kwargs(self, payload: dict[str, Any], action_type: StagehandActionType) -> dict[str, Any]:
        """Map extra input fields onto Stagehand v3 call parameters.

        Top-level option fields (`model`, `variables`, `selector`, ...) fold into `options`;
        `schema` is passed through for extract; unknown fields are dropped with a warning instead
        of failing the whole call.
        """
        payload = dict(payload)
        call_kwargs: dict[str, Any] = {}
        options = dict(payload.pop("options", None) or {})
        allowed_options = _OPTION_FIELDS_BY_ACTION.get(action_type.value, ())
        for key in _OPTION_FIELDS:
            if key in payload:
                if key in allowed_options:
                    options[key] = payload.pop(key)
                else:
                    payload.pop(key)
                    logger.warning(
                        f"Tool {self.name} - {self.id}: '{key}' is not supported by the "
                        f"'{action_type.value}' action; ignoring."
                    )
        if "frame_id" in payload:
            call_kwargs["frame_id"] = payload.pop("frame_id")
        if action_type == StagehandActionType.EXTRACT and "schema" in payload:
            call_kwargs["schema"] = payload.pop("schema")
        if payload.pop("iframes", None):
            logger.info(f"Tool {self.name} - {self.id}: 'iframes' is handled automatically by Stagehand; ignoring.")
        if payload:
            logger.warning(
                f"Tool {self.name} - {self.id}: ignoring input fields not supported by Stagehand: {sorted(payload)}"
            )
        if options:
            call_kwargs["options"] = options
        return call_kwargs

    async def _call_with_model_fallback(self, call, action_label: str, payload: dict[str, Any]):
        """Run a Stagehand call, retrying on ``fallback_model_name`` if the model's answer is
        rejected by the server's structured-output validation.

        Some models produce answers the server cannot coerce into the shape act/observe/extract
        require. It is deterministic per request — measured on claude-sonnet-5, whose observe fails
        this way for a sizeable share of pages and fails identically on every same-model retry —
        so the only recovery is a different model for that one call.
        """
        try:
            return await call(payload)
        except InternalServerError as exc:
            fallback = self.fallback_model_name
            if _SCHEMA_MISMATCH_MARKER not in str(exc) or not fallback or fallback == self.model_name:
                raise
            logger.warning(
                f"Tool {self.name} - {self.id}: {action_label} produced schema-invalid output with "
                f"{self.model_name}; retrying this call with {fallback}."
            )
            retry_payload = dict(payload)
            options = dict(retry_payload.get("options") or {})
            options["model"] = fallback
            retry_payload["options"] = options
            return await call(retry_payload)

    async def execute_async(
        self, input_data: StagehandInputSchema, config: RunnableConfig | None = None, **kwargs
    ) -> dict[str, Any]:
        """
        Executes a web automation task using the Stagehand session based on the input action type.

        Args:
            input_data (StagehandInputSchema): The schema describing the action to perform.
            config (RunnableConfig, optional): Execution config if applicable.
            **kwargs: Additional arguments.
        Returns:
            dict[str, Any]: The result of the action.
        Raises:
            ToolExecutionException: If the input is invalid or execution fails.
        """
        config = ensure_config(config)
        check_cancellation(config)
        shared_browser = await self._acquire_shared_browser()
        create_shared_session = False
        shared_context_id = None
        if shared_browser:
            await self._join_shared_browser(shared_browser)
            create_shared_session = self._session_id is None  # nobody has created it yet: we do
            shared_context_id = shared_browser.browser_context_id()
        await self._init_client(shared_context_id, create_shared_session)
        if shared_browser:
            if create_shared_session:
                self._record_shared_browser_session(shared_browser)
                if self._stagehand_session is None:
                    # Lost the creation race: our session was ended, attach to the winner's.
                    await self._init_client(shared_context_id, create_shared_session=False)
            shared_browser.set_browser_live_view_url(self._live_view_url)

        tool_data = {"tool_session_id": self._session_id}
        self.run_on_node_execute_run(
            config.callbacks,
            tool_data=tool_data,
            **kwargs,
        )

        session = self._stagehand_session
        files = []
        try:
            payload = self._build_call_kwargs(getattr(input_data, "model_extra", {}) or {}, input_data.action_type)
            if input_data.action_type == StagehandActionType.EXTRACT:
                response_obj = await self._call_with_model_fallback(
                    lambda p: session.extract(instruction=input_data.instruction, **p), "extract", payload
                )
                result = response_obj.data.result
            elif input_data.action_type == StagehandActionType.OBSERVE:
                response_obj = await self._call_with_model_fallback(
                    lambda p: session.observe(instruction=input_data.instruction, **p), "observe", payload
                )
                result = [el.model_dump() for el in response_obj.data.result]
            elif input_data.action_type == StagehandActionType.ACT:
                response_obj = await self._call_with_model_fallback(
                    lambda p: session.act(input=input_data.instruction, **p), "act", payload
                )
                result = response_obj.data.result.model_dump()
                zip_data = await self.get_downloads_via_sdk(self._session_id, config=config)
                files = self.extract_files_from_zip(zip_data) if zip_data else []
            elif input_data.action_type == StagehandActionType.GOTO:
                if input_data.url is None:
                    raise ToolExecutionException(
                        "Missing required URL for 'navigate' action. Please provide a valid URL.", recoverable=True
                    )
                await session.navigate(url=input_data.url, **payload)
                result = "Navigated to " + input_data.url
            elif input_data.action_type == StagehandActionType.GOBACK:
                page = await self._ensure_page()
                url_before = page.url
                try:
                    await page.go_back(wait_until="domcontentloaded", timeout=_HISTORY_NAV_TIMEOUT_MS)
                except Exception:
                    # A back-navigation restored from the bfcache completes without firing the
                    # awaited lifecycle event; the URL moving is the real signal it worked.
                    if page.url == url_before:
                        raise
                    logger.info(f"Tool {self.name} - {self.id}: go_back completed without a load event.")
                result = "Navigated to previous page"
            elif input_data.action_type == StagehandActionType.UPLOAD:
                if input_data.files is None:
                    raise ToolExecutionException(
                        "No file provided for upload action. Please provide a file to upload.", recoverable=True
                    )

                page = await self._ensure_page()
                # The chooser only opens after act's server-side LLM round-trip and click, which
                # can outlast Playwright's stock 30s waiter. Cap the wait well below the tool
                # timeout: if act clicked something that never opens a chooser, an hour-long hold
                # of the call lock (and shared page control) would starve every other agent.
                chooser_timeout_ms = min(self.timeout, 300) * 1000
                # The fallback retry stays INSIDE the waiter: its click is the one that opens the
                # chooser when the primary model's answer was unusable.
                async with page.expect_file_chooser(timeout=chooser_timeout_ms) as fc_info:
                    response_obj = await self._call_with_model_fallback(
                        lambda p: session.act(input=input_data.instruction, **p), "upload", payload
                    )

                file_chooser = await fc_info.value
                if file_chooser:
                    logger.debug(f"{self.id} - {self.name} - Uploading file {input_data.files.name}")

                    # Read file content and automatically detect MIME type
                    file_content = input_data.files.getvalue()
                    mime_type = guess_mime_type_from_bytes(file_content, input_data.files.name)

                    logger.debug(f"Detected MIME type: {mime_type} for file: {input_data.files.name}")

                    await file_chooser.set_files(
                        [{"name": input_data.files.name, "mimeType": mime_type, "buffer": file_content}]
                    )

                result = response_obj.data.result.model_dump()
            else:
                raise ToolExecutionException(f"Invalid action type: {input_data.action_type}", recoverable=True)

        except ToolExecutionException:
            raise
        except CanceledException:
            # Cancellation must reach the runner, not become a retryable tool observation.
            raise
        except NotFoundError as e:
            # The session is gone server-side (timed out / released elsewhere): drop the local
            # handle so the retried call starts a fresh session instead of failing forever.
            # Steel owns the browser separately from the Stagehand session, so release it too —
            # otherwise the retry either reattaches to a dead browser or strands a live one.
            await self._release_steel_browser_session()
            if shared_browser is not None and self._session_id:
                # Clearing our own handles is not enough under sharing: the dead id is still the
                # run's published session, so _join_shared_browser would hand it straight back and
                # the retry would resume the same corpse. Retract it and the retry creates a new one.
                shared_browser.invalidate_browser_session(self._session_id)
            self._stagehand_session = None
            self._session_id = None
            self._live_view_url = None
            self._cdp_url = None
            # Our CDP connection points at the dead browser; a stale handle can still report
            # connected, so drop it rather than let _ensure_page reuse it.
            await self._close_playwright()
            raise ToolExecutionException(f"Browser session expired, retry will start a new one: {e}", recoverable=True)
        except (AuthenticationError, PermissionDeniedError) as e:
            # Bad credentials cannot be fixed by retrying the same call.
            raise ToolExecutionException(f"Error message: {e}", recoverable=False)
        except Exception as e:
            raise ToolExecutionException(f"Error message: {e}", recoverable=True)

        # Capture screenshot if enabled (only for act and goto actions)
        screenshot = None
        if self.is_return_screenshot_bytes_enabled and input_data.action_type in (
            StagehandActionType.ACT,
            StagehandActionType.GOTO,
            StagehandActionType.GOBACK,
            StagehandActionType.UPLOAD,
        ):
            screenshot = await self._take_screenshot()

        response: dict[str, Any] = {"content": result}
        if files:
            response["files"] = files
        if screenshot:
            response["screenshot"] = screenshot
        if self.is_return_live_view_url_enabled and self._live_view_url:
            response["live_view_url"] = self._live_view_url

        return response

    def close(self) -> None:
        """Best-effort cleanup when nodes are not explicitly shut down."""
        try:
            if self._shares_browser_session:
                # Detach, never end — other agents are still driving it (also guards __del__).
                self._detach_shared_browser()
            elif self._loop_alive():
                if self._stagehand_session is not None:
                    try:
                        self._run_in_loop(self._stagehand_session.end(), timeout=_CLEANUP_TIMEOUT)
                    except Exception as exc:
                        logger.warning(f"Stagehand session end failed: {exc}")
                if self._pw_browser is not None or self._playwright is not None:
                    try:
                        self._run_in_loop(self._close_playwright(), timeout=_CLEANUP_TIMEOUT)
                    except Exception as exc:
                        logger.warning(f"Stagehand playwright close failed: {exc}")
                if self.client is not None and hasattr(self.client, "close"):
                    try:
                        self._run_in_loop(self.client.close(), timeout=_CLEANUP_TIMEOUT)
                    except Exception as exc:
                        logger.warning(f"Stagehand client close failed: {exc}")

            if not self._loop_alive():
                pass
            elif self._is_steel_browser_connection():
                if self._steel_browser_session and self._steel_client:
                    try:
                        session_id = getattr(self._steel_browser_session, "id", self._session_id)
                        if session_id:
                            logger.info(f"Releasing Steel session: {session_id}")
                            self._run_in_loop(self._steel_client.sessions.release(session_id), timeout=_CLEANUP_TIMEOUT)
                        else:
                            logger.warning("Cannot release Steel session: no valid session ID found")
                    except Exception as exc:
                        logger.warning(f"Failed to release Steel session: {exc}")
                if self._steel_client and hasattr(self._steel_client, "close"):
                    try:
                        self._run_in_loop(self._steel_client.close(), timeout=_CLEANUP_TIMEOUT)
                    except Exception as exc:
                        logger.warning(f"Failed to close Steel client: {exc}")
            elif self._browserbase_client and hasattr(self._browserbase_client, "close"):
                self._run_in_loop(self._browserbase_client.close(), timeout=_CLEANUP_TIMEOUT)
        except Exception as e:
            logger.warning(f"Stagehand close() failed: {e}")
        finally:
            self.client = None
            self._stagehand_session = None
            self._cdp_url = None
            self._browserbase_client = None
            self._steel_browser_session = None
            self._steel_client = None
            self._live_view_url = None
            # Drop CDP handles even if teardown failed, so a later run reconnects instead of
            # reusing a stale page.
            self._pw_page = None
            self._pw_browser = None
            self._playwright = None
            # Clear the (now-terminated) session id so a later run starts fresh, not on a dead id.
            self._session_id = None
            # Sharing is re-established per turn by _join_shared_browser. Leaving this set would
            # make a later UNSHARED run detach instead of end, stranding its session.
            self._shares_browser_session = False
            self.close_loop()

    def __del__(self):
        try:
            self.close()
        except Exception:
            logger.warning("Stagehand __del__ failed")

    @property
    def to_dict_exclude_params(self):
        """
        Property to define which parameters should be excluded when converting the class instance to a dictionary.

        Returns:
            dict: A dictionary defining the parameters to exclude.
        """
        return super().to_dict_exclude_params | {
            "_browserbase_client": True,
            "_steel_browser_session": True,
            "_steel_client": True,
            "_stagehand_session": True,
            "_playwright": True,
            "_pw_browser": True,
            "_pw_page": True,
        }
