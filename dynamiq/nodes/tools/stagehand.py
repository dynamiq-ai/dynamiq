import asyncio
import io
import threading
import time
import zipfile
from enum import Enum
from typing import Any, ClassVar, Coroutine, Literal
from urllib.parse import urlparse

import requests
from browserbase import AsyncBrowserbase
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, field_validator
from stagehand import Stagehand as StagehandClient
from steel import AsyncSteel

from dynamiq.connections import Browserbase, StagehandEnvironment, SteelBrowser, SteelBrowserEnvironment
from dynamiq.connections.managers import ConnectionManager
from dynamiq.nodes import NodeGroup
from dynamiq.nodes.agents.exceptions import ToolExecutionException
from dynamiq.nodes.node import ConnectionNode, ensure_config
from dynamiq.nodes.tools.utils import guess_mime_type_from_bytes, sanitize_filename
from dynamiq.runnables import RunnableConfig
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

Additional fields passed in the input (beyond the ones above) are forwarded unchanged
to the underlying Stagehand call. Provide them in the exact format expected by Stagehand.
If you need to interact with iframe content, make sure to set 'iframes': True as one of the input fields. For best
performance, use this option only when necessary, such as when encountering an iframe-related error.

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
     "brief": "Identify clickable links on the homepage",
     "iframes": true
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
    is_return_screenshot_bytes_enabled: bool = Field(
        default=False, description="Whether to capture and return a screenshot as bytes after act/goto actions."
    )
    is_return_live_view_url_enabled: bool = Field(
        default=False, description="Whether to return the live view URL in every response."
    )

    _browserbase_client: AsyncBrowserbase | None = PrivateAttr(default=None)
    _steel_client: AsyncSteel | None = PrivateAttr(default=None)
    _steel_browser_session: Any | None = PrivateAttr(default=None)
    input_schema: ClassVar[type[StagehandInputSchema]] = StagehandInputSchema
    timeout: int = 3600
    is_files_allowed: bool = True
    _session_id: str | None = PrivateAttr(default=None)
    _live_view_url: str | None = PrivateAttr(default=None)
    _loop = PrivateAttr(default=None)
    _loop_thread = PrivateAttr(default=None)

    _clone_init_methods_names: ClassVar[list[str]] = ["init_loop"]

    def _is_steel_browser_connection(self) -> bool:
        """Check if the current connection is an instance of Steel Browser connection."""
        return isinstance(self.connection, SteelBrowser)

    def _get_steel_browser_headers(self) -> dict[str, str]:
        """Get the headers for the Steel Browser API requests."""
        if not self._is_steel_browser_connection():
            return {}
        if self.connection.environment == SteelBrowserEnvironment.CLOUD and self.connection.api_key:
            return {"steel-api-key": self.connection.api_key}
        return {}

    def _get_steel_browser_base_url(self) -> str:
        """Get the base URL for the Steel Browser API requests."""
        if self.connection.environment == SteelBrowserEnvironment.CLOUD:
            return "https://api.steel.dev/v1"
        return (self.connection.base_url or "http://localhost:3000").rstrip("/")

    def _create_steel_browser_stagehand_config(self, cdp_url: str):
        """Create a Stagehand config for the Steel Browser session."""
        from stagehand import StagehandConfig

        connection_cdp_url = cdp_url
        if self.connection.environment == SteelBrowserEnvironment.CLOUD and self.connection.api_key:
            separator = "&" if "?" in cdp_url else "?"
            connection_cdp_url = f"{cdp_url}{separator}apiKey={self.connection.api_key}"

        return StagehandConfig(
            env=StagehandEnvironment.LOCAL,
            model_name=self.model_name,
            model_api_key=self.connection.model_api_key,
            local_browser_launch_options={"cdp_url": connection_cdp_url},
            **self.connection.extra_config,
        )

    async def _ensure_steel_browser_session(self):
        """Check if the Steel Browser session exists and create it if it doesn't."""
        if self._steel_browser_session is None:
            self._steel_browser_session = await self._steel_client.sessions.create(**self.connection.session_config)
            if hasattr(self._steel_browser_session, "id"):
                logger.info(f"Steel session created: {self._steel_browser_session.id}")
            if hasattr(self._steel_browser_session, "session_viewer_url"):
                logger.info(f"Steel session viewer: {self._steel_browser_session.session_viewer_url}")
        return self._steel_browser_session

    def _upload_file_bytes_to_steel_browser_session(self, session_id: str, file_bytes: bytes, file_name: str) -> dict:
        """Upload a file to the Steel Browser session."""
        url = f"{self._get_steel_browser_base_url()}/sessions/{session_id}/files/upload"
        files = {"file": (file_name, io.BytesIO(file_bytes))}
        response = requests.post(url, headers=self._get_steel_browser_headers(), files=files, timeout=self.timeout)
        response.raise_for_status()
        try:
            return response.json()
        except ValueError as e:
            logger.error(f"Failed to parse JSON from Steel Browser upload response: {e}")
            raise ToolExecutionException(
                f"Steel Browser API returned invalid JSON after file upload: {e}",
                recoverable=True,
            )

    def _list_steel_browser_session_files(self, session_id: str) -> list[dict]:
        """List all files in the Steel Browser session's filesystem."""
        url = f"{self._get_steel_browser_base_url()}/sessions/{session_id}/files"
        response = requests.get(url, headers=self._get_steel_browser_headers(), timeout=self.timeout)
        response.raise_for_status()
        try:
            payload = response.json()
        except ValueError as e:
            logger.error(f"Failed to parse JSON from Steel Browser files response: {e}")
            return []
        if isinstance(payload, dict):
            logger.info(f"Steel session files: {payload.get('data') or []}")
            return payload.get("data") or []
        return []

    def _download_steel_browser_session_files_archive(self, session_id: str) -> bytes:
        """Download the archive of the files in the Steel session."""
        url = f"{self._get_steel_browser_base_url()}/sessions/{session_id}/files/archive"
        response = requests.get(url, headers=self._get_steel_browser_headers(), timeout=self.timeout)
        response.raise_for_status()
        return response.content

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

    def _run_in_loop(self, coro: Coroutine):
        """Run a coroutine in the persistent event loop."""
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result()

    async def upload_files(self, files: list[io.BytesIO]):
        if self._is_steel_browser_connection():
            session = await self._ensure_steel_browser_session()
            if not hasattr(session, "id"):
                raise ToolExecutionException("Steel session missing required 'id' attribute", recoverable=False)
            for file_obj in files:
                if not hasattr(file_obj, "name"):
                    file_obj.name = "uploaded_file"
                result = await asyncio.to_thread(
                    self._upload_file_bytes_to_steel_browser_session,
                    session.id,
                    file_obj.getvalue(),
                    file_obj.name,
                )
                logger.info(f"Uploaded {file_obj.name} to Steel session: {result}")
            return

        for file_obj in files:
            if not hasattr(file_obj, "name"):
                file_obj.name = "uploaded_file"

            result = await self._browserbase_client.sessions.uploads.create(self._session_id, file=file_obj)
            logger.info(f"Uploaded {file_obj.name}: {result}")

    async def _init_client(self, files: list[io.BytesIO]):
        """Initialize Stagehand client and reinitialize with session reuse."""

        class StagehandNoSignal(StagehandClient):
            # Suppress signal handlers to avoid "signal only works in main thread" error
            def _register_signal_handlers(self):
                pass

        if self._is_steel_browser_connection():
            await self._ensure_steel_browser_session()
            if (
                not hasattr(self._steel_browser_session, "websocket_url")
                or not self._steel_browser_session.websocket_url
            ):
                raise ToolExecutionException(
                    "Steel session missing required 'websocket_url' attribute or has invalid value", recoverable=False
                )
            config = self._create_steel_browser_stagehand_config(self._steel_browser_session.websocket_url)
        else:
            config = self.connection.config or {}

        if self.client is None:
            self.client = StagehandNoSignal(config=config, model_api_key=self.connection.model_api_key)
            self.client.model_name = self.model_name
            if self._is_steel_browser_connection():
                # Recreate Steel client if it was cleaned up during close()
                if self._steel_client is None:
                    if self.connection.environment == SteelBrowserEnvironment.CLOUD:
                        self._steel_client = AsyncSteel(steel_api_key=self.connection.api_key)
                    elif self.connection.environment == SteelBrowserEnvironment.SELF_HOSTED:
                        self._steel_client = AsyncSteel(base_url=self.connection.base_url)
            else:
                self._browserbase_client = self._browserbase_client or AsyncBrowserbase(
                    api_key=self.connection.browserbase_api_key
                )
        else:
            self.client.config = config
            self.client.model_name = self.model_name

        self.client.initialized = False

        if self._session_id:
            self.client.session_id = self._session_id

        await self.client.init()
        if self._is_steel_browser_connection():
            self._session_id = getattr(self._steel_browser_session, "id", self._session_id)
            if self._session_id:
                self.client.session_id = self._session_id
        else:
            self._session_id = self.client.session_id

        # Fetch live view URL only when enabled (once per session)
        if self.is_return_live_view_url_enabled and not self._live_view_url:
            await self._fetch_live_view_url()

    async def _fetch_live_view_url(self) -> None:
        """Fetch and store the live view URL for the current session."""
        try:
            if self._is_steel_browser_connection():
                # Steel Browser: get from session object
                if self._steel_browser_session and hasattr(self._steel_browser_session, "session_viewer_url"):
                    self._live_view_url = self._steel_browser_session.session_viewer_url
            else:
                # Browserbase: fetch from debug API
                if self._session_id and self._browserbase_client:
                    debug_info = await self._browserbase_client.sessions.debug(self._session_id)
                    if hasattr(debug_info, "debugger_fullscreen_url"):
                        self._live_view_url = debug_info.debugger_fullscreen_url
                    elif hasattr(debug_info, "debuggerFullscreenUrl"):
                        self._live_view_url = debug_info.debuggerFullscreenUrl

            if self._live_view_url:
                logger.info(f"Live view URL: {self._live_view_url}")
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
            screenshot_bytes = await self.client.page.screenshot()

            page_host = ""
            try:
                page_url = self.client.page.url
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
        return self._run_async(self.execute_async(input_data, config, **kwargs))

    async def get_downloads_via_sdk(self, session_id: str, max_retry_sec: int = 20) -> bytes:
        """
        Retrieve the downloads archive for the given session.
        Retries until content is non-empty or timeout.
        """
        if self._is_steel_browser_connection():
            files = await asyncio.to_thread(self._list_steel_browser_session_files, session_id)
            if files:
                return await asyncio.to_thread(self._download_steel_browser_session_files_archive, session_id)
            else:
                return b""

        elapsed = 0
        interval = 2

        while elapsed < max_retry_sec:
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
        logger.info(f"Tool {self.name} - {self.id}: started with input:\n{input_data.model_dump()}")
        config = ensure_config(config)
        await self._init_client(input_data.files)

        tool_data = {"tool_session_id": self.client.session_id}
        self.run_on_node_execute_run(
            config.callbacks,
            tool_data=tool_data,
            **kwargs,
        )

        files = []
        try:
            payload: dict[str, Any] = getattr(input_data, "model_extra", {}) or {}
            if input_data.action_type == StagehandActionType.EXTRACT:
                result = await self.client.page.extract(input_data.instruction, **payload)
                result = result.model_dump()
            elif input_data.action_type == StagehandActionType.OBSERVE:
                result = await self.client.page.observe(input_data.instruction, **payload)
                result = [el.model_dump() for el in result]
            elif input_data.action_type == StagehandActionType.ACT:
                result = await self.client.page.act(input_data.instruction, **payload)
                result = result.model_dump()
                zip_data = await self.get_downloads_via_sdk(self.client.session_id)
                files = self.extract_files_from_zip(zip_data) if zip_data else []
            elif input_data.action_type == StagehandActionType.GOTO:
                if input_data.url is None:
                    raise ToolExecutionException(
                        "Missing required URL for 'navigate' action. Please provide a valid URL.", recoverable=True
                    )
                await self.client.page.goto(input_data.url)
                result = "Navigated to " + input_data.url
            elif input_data.action_type == StagehandActionType.GOBACK:
                await self.client.page.go_back()
                result = "Navigated to previous page"
            elif input_data.action_type == StagehandActionType.UPLOAD:
                if input_data.files is None:
                    raise ToolExecutionException(
                        "No file provided for upload action. Please provide a file to upload.", recoverable=True
                    )

                async with self.client.page.expect_file_chooser() as fc_info:
                    result = await self.client.page.act(input_data.instruction)

                file_chooser = await fc_info.value
                if file_chooser:
                    logger.info(f"{self.id} - {self.name} - Uploading file {input_data.files.name}")

                    # Read file content and automatically detect MIME type
                    file_content = input_data.files.getvalue()
                    mime_type = guess_mime_type_from_bytes(file_content, input_data.files.name)

                    logger.info(f"Detected MIME type: {mime_type} for file: {input_data.files.name}")

                    await file_chooser.set_files(
                        [{"name": input_data.files.name, "mimeType": mime_type, "buffer": file_content}]
                    )

                result = result.model_dump()
            else:
                raise ToolExecutionException(f"Invalid action type: {input_data.action_type}", recoverable=True)

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

        logger.info(f"Tool {self.name} - {self.id}: finished with result:\n{str(result)[:200]}...")

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
            if self.client and hasattr(self.client, "close"):
                self._run_in_loop(self.client.close())

            if self._is_steel_browser_connection():
                if self._steel_browser_session and self._steel_client:
                    try:
                        session_id = getattr(self._steel_browser_session, "id", self._session_id)
                        if session_id:
                            logger.info(f"Releasing Steel session: {session_id}")
                            self._run_in_loop(self._steel_client.sessions.release(session_id))
                        else:
                            logger.warning("Cannot release Steel session: no valid session ID found")
                    except Exception as exc:
                        logger.warning(f"Failed to release Steel session: {exc}")
                if self._steel_client and hasattr(self._steel_client, "close"):
                    try:
                        self._run_in_loop(self._steel_client.close())
                    except Exception as exc:
                        logger.warning(f"Failed to close Steel client: {exc}")
            elif self._browserbase_client and hasattr(self._browserbase_client, "close"):
                self._run_in_loop(self._browserbase_client.close())
        except Exception as e:
            logger.warning(f"Stagehand close() failed: {e}")
        finally:
            self.client = None
            self._browserbase_client = None
            self._steel_browser_session = None
            self._steel_client = None
            self._live_view_url = None
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
        }
