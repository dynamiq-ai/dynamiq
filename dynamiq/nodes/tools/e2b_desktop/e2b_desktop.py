import asyncio
import io
import threading
from typing import Any, ClassVar, Coroutine, Literal

from e2b_desktop import Sandbox
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, field_validator

from dynamiq.connections import E2B
from dynamiq.connections.managers import ConnectionManager
from dynamiq.nodes import NodeGroup
from dynamiq.nodes.agents.exceptions import ToolExecutionException
from dynamiq.nodes.node import ConnectionNode
from dynamiq.nodes.tools.e2b_desktop.types import (
    E2BAction,
    E2BClickMouseSchema,
    E2BLaunchSchema,
    E2BMoveMouseSchema,
    E2BRunBackgroundCommandSchema,
    E2BRunCommandSchema,
    E2BSendKeySchema,
    E2BTakeScreenshotSchema,
    E2BToolType,
    E2BTypeTextSchema,
    E2BWaitSchema,
)
from dynamiq.nodes.tools.utils import sanitize_filename
from dynamiq.runnables import RunnableConfig
from dynamiq.utils.logger import logger

ACTION_SCHEMA_MAP: dict[E2BAction, type] = {
    E2BAction.MOVE_MOUSE: E2BMoveMouseSchema,
    E2BAction.CLICK_MOUSE: E2BClickMouseSchema,
    E2BAction.WAIT: E2BWaitSchema,
    E2BAction.RUN_COMMAND: E2BRunCommandSchema,
    E2BAction.RUN_BACKGROUND_COMMAND: E2BRunBackgroundCommandSchema,
    E2BAction.SEND_KEY: E2BSendKeySchema,
    E2BAction.TYPE_TEXT: E2BTypeTextSchema,
    E2BAction.TAKE_SCREENSHOT: E2BTakeScreenshotSchema,
    E2BAction.LAUNCH: E2BLaunchSchema,
}


DESCRIPTION_E2B_DESKTOP = """## Sandbox Desktop Tool
### Overview
Use this tool to control a virtual desktop by performing `computer`
actions such as moving/clicking the mouse, typing, sending keys, running commands, taking screenshots,
 and launching applications.

### Supported action_type
- `action_type`: must be `computer`.

### Supported actions (set via `action`)
- `move_mouse`: Move the mouse to specific screen coordinates.
- `click_mouse`: Click at the current position or at provided coordinates.
- `wait`: Pause execution for a specified number of seconds.
- `run_command`: Run a shell command synchronously and return stdout/stderr.
- `run_background_command`: Start a shell command asynchronously (no output).
- `send_key`: Send a key or key combination (e.g., "Enter", "Ctrl+C").
- `type_text`: Type text into the currently focused input.
- `take_screenshot`: Capture a screenshot and return it as a file (`screenshot.png`).
- `launch`: Start an application in the background.

### Parameters by action
- `move_mouse`
  - `coordinates`: [x, y] absolute screen coordinates (integers)

- `click_mouse`
  - `coordinates` (optional): [x, y] to move before clicking; omit to click current position
  - `button` (optional): `left` or `right` (default: `left`)
  - `num_clicks` (optional): number of clicks (default: 1; use 2 for double-click)

- `wait`
  - `duration`: seconds to wait (integer)

- `run_command`
  - `command`: full shell command to run synchronously
  - `timeout` (optional): timeout in seconds (default: 10)

- `run_background_command`
  - `command`: full shell command to run in background

- `send_key`
  - `name`: key or combination name (e.g., `Enter`, `Ctrl+A`, `Ctrl+C`)

- `type_text`
  - `text`: the text to type (string)
  - `chunk_size` (optional): size for grouped typing (integer)
  - `delay_in_ms` (optional): delay between chunks in milliseconds (integer)

- `take_screenshot`
  - no additional parameters

- `launch`
  - `application`: command name (e.g., `google-chrome`, `firefox`, `code`)

### Usage examples
1) Move mouse
```json
{
  "action_type": "computer",
  "action": "move_mouse",
  "coordinates": [500, 300]
}
```

2) Click left at given position
```json
{
  "action_type": "computer",
  "action": "click_mouse",
  "button": "left",
  "coordinates": [500, 300]
}
```

3) Wait for 3 seconds
```json
{
  "action_type": "computer",
  "action": "wait",
  "duration": 3
}
```

4) Run a command synchronously
```json
{
  "action_type": "computer",
  "action": "run_command",
  "command": "echo hello",
  "timeout": 10
}
```

5) Run a command in background
```json
{
  "action_type": "computer",
  "action": "run_background_command",
  "command": "google-chrome --incognito"
}
```

6) Send a key
```json
{
  "action_type": "computer",
  "action": "send_key",
  "name": "Enter"
}
```

7) Type text
```json
{
  "action_type": "computer",
  "action": "type_text",
  "text": "Hello, World!"
}
```

8) Take screenshot
```json
{
  "action_type": "computer",
  "action": "take_screenshot"
}
```

9) Launch application
```json
{
  "action_type": "computer",
  "action": "launch",
  "application": "google-chrome"
}
```

### Execution guidelines
- Verify each step with a `take_screenshot` before proceeding.
- After launching GUI apps with `launch`, insert a short `wait`, then confirm with `take_screenshot`.
- Interpret each screenshot and plan the next action based on what changed.
- For web interactions, zoom out or scroll so required elements are visible before clicking.
- If an interaction fails, move the mouse to a neutral area, then retry the click.
- If the first screenshot is blank/black, click the screen center and take another screenshot.
- Before typing: ensure the input is focused; if needed, click it, then clear with
`send_key` = `Ctrl+A` and `Backspace`; then use `type_text`.
- After typing in a search/input field, activate it by pressing `Enter` or clicking the submit/search button.
- Insert short `wait` steps between critical actions to improve reliability.

### Best practices
- For browser tasks prioritize google-chrome with action
- Always capture a `take_screenshot` after actions to verify the result.
- After navigation or app launches, verify state changes with `take_screenshot` before continuing.
- Explain each action and the reason, especially for multi‑step sequences.
- Before typing in input field, make sure it is focused. Use 'action':'type_text'
with 'text':'/' to focus on first input field.
- To focus on search of website in browser use CTRL+L to activate browser search bar.
"""


class E2BDesktopToolInputSchema(BaseModel):
    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)

    action_type: E2BToolType = Field(
        ...,
        description="""
    Specifies the type of interaction with the desktop:
    - `computer`: Perform a desktop action (move_mouse, click_mouse, wait, run_command, run_background_command,
      send_key, type_text).
    """,
    )
    action: E2BAction = Field(
        ...,
        description="""
    Specifies the action to perform:
    - `move_mouse`: Move the mouse to the specified coordinates.
    - `click_mouse`: Click the mouse at the specified coordinates.
    - `wait`: Wait for the specified duration.
    - `run_command`: Run the specified command.
    - `run_background_command`: Run the specified command in background.
    - `send_key`: Send the specified key.
    - `type_text`: Type the specified text.
    """,
    )
    # Action parameters are provided flat at the top-level, not nested.

    files: str | list[str] | io.BytesIO | list[io.BytesIO] | bytes | list[bytes] | None = Field(
        default=None,
        description="Files to upload to /home/user/. Accepts names (str or list[str]) mapped from storage,"
        "or bytes/BytesIO object(s). Make sure to specify what filenames of files need "
        "to be uploaded otherwise all files from file system will be uploaded.",
        json_schema_extra={"map_from_storage": True},
    )

    @field_validator("files", mode="before")
    @classmethod
    def files_validator(cls, input_data: Any):
        """Validate and process files."""
        if isinstance(input_data, str) or isinstance(input_data, list[str]):
            raise ToolExecutionException(
                "File path provided for upload action. Please provide a file to upload.", recoverable=True
            )
        return input_data


class E2BDesktopTool(ConnectionNode):
    group: Literal[NodeGroup.TOOLS] = NodeGroup.TOOLS
    name: str = "E2B Sandbox Tool"
    description: str = DESCRIPTION_E2B_DESKTOP
    connection: E2B
    input_schema: ClassVar[type[E2BDesktopToolInputSchema]] = E2BDesktopToolInputSchema
    timeout: int = 3600
    is_files_allowed: bool = True

    _desktop: Sandbox | None = PrivateAttr(default=None)
    _sandbox_id: str | None = PrivateAttr(default=None)
    _stream_url: str | None = PrivateAttr(default=None)
    _loop = PrivateAttr(default=None)
    _loop_thread = PrivateAttr(default=None)
    _uploads_report: list[dict[str, Any]] | None = PrivateAttr(default=None)

    _clone_init_methods_names: ClassVar[list[str]] = ["init_loop"]

    def init_components(self, connection_manager: ConnectionManager | None = None):
        super().init_components(connection_manager)
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

    async def _upload_files(self, files_field: Any) -> None:
        """Upload provided file object(s) to /home/user/ and build a status report."""
        self._uploads_report = None
        if not files_field:
            return

        def _to_list(v):
            return v if isinstance(v, list) else [v]

        reports: list[dict[str, Any]] = []
        for fobj in _to_list(files_field):
            raw_name = getattr(fobj, "name", None)
            name = sanitize_filename(filename=raw_name, default="file")
            dest_path = f"/home/user/{name}"
            try:
                if isinstance(fobj, (bytes, bytearray)):
                    data = fobj
                elif isinstance(fobj, io.BytesIO):
                    data = fobj.getvalue()
                else:
                    data = bytes(fobj)
                self._desktop.files.write(dest_path, data)
                reports.append({"name": name, "path": dest_path, "status": "uploaded"})
            except Exception as e:
                reports.append({"name": name, "path": dest_path, "status": "error", "error": str(e)})
        if reports:
            self._uploads_report = reports
            logger.info(f"E2BDesktopTool uploads: {reports}")

    async def _init_client(self):
        """Initialize E2B desktop sandbox and stream (idempotent)."""
        if self._desktop is None:
            self._desktop = Sandbox.create(
                api_key=self.connection.api_key, timeout=self.timeout, domain=self.connection.domain
            )
            self._sandbox_id = self._desktop.sandbox_id
            self._desktop.stream.start(require_auth=True)
            auth_key = self._desktop.stream.get_auth_key()
            self._stream_url = self._desktop.stream.get_url(auth_key=auth_key)
            logger.debug(f"E2B sandbox started: {self._sandbox_id}")
            logger.debug(f"Stream URL: {self._stream_url}")

    def _run_async(self, coro) -> Any:
        """Run coroutine in the persistent event loop."""
        return self._run_in_loop(coro)

    def execute(
        self, input_data: E2BDesktopToolInputSchema, config: RunnableConfig | None = None, **kwargs
    ) -> dict[str, Any]:
        """
        Executes the tool synchronously with the provided input.

        Always dispatches onto a dedicated background event loop to avoid cross-loop issues.

        Args:
            input_data (E2BDesktopToolInputSchema): Input data for the tool execution.
            config (RunnableConfig, optional): Configuration for the runnable, including callbacks.
            **kwargs: Additional arguments passed to the execution context.

        Returns:
            dict[str, Any]: A dictionary containing the tool's output.
        """
        return self._run_async(self.execute_async(input_data, config, **kwargs))

    async def execute_async(
        self, input_data: E2BDesktopToolInputSchema, config: RunnableConfig | None = None, **kwargs
    ) -> dict[str, Any]:
        """
        Executes a task on a VM instance using the e2b sandbox session based on the input action type.

        Args:
            input_data (E2BDesktopToolInputSchema): The schema describing the action to perform.
            config (RunnableConfig, optional): Execution config if applicable.
            **kwargs: Additional arguments.
        Returns:
            dict[str, Any]: The result of the action.
        Raises:
            ToolExecutionException: If the input is invalid or execution fails.
        """
        logger.info(f"Tool {self.name} - {self.id}: started with input:\n{input_data.model_dump()}")

        await self._init_client()

        if input_data.files:
            logger.info(f"E2BDesktopTool uploading files: {input_data.files}")
            await self._upload_files(input_data.files)
        # E2B sandbox is initialized in _init_client; no separate VM instance step

        if input_data.action_type == E2BToolType.COMPUTER:
            try:
                result = await self._execute_computer_tool(input_data)
            except Exception as e:
                raise ToolExecutionException(f"Error message: {e}", recoverable=True) from e
        else:
            raise ToolExecutionException(f"Invalid action_type: {input_data.action_type}", recoverable=True)

        logger.info(f"Tool {self.name} - {self.id}: finished with result:\n{str(result)[:200]}...")

        # If inner result contains files, lift them to top-level for agent processing
        files = None
        if isinstance(result, dict) and "files" in result:
            files = result.pop("files")

        response: dict[str, Any] = {"content": result}
        if files:
            response["files"] = files
        return response

    async def _init_vm_instance(self):
        """No-op for E2B Sandbox (handled in _init_client)."""
        return

    async def _execute_computer_tool(self, input_data: E2BDesktopToolInputSchema) -> dict[str, Any]:
        """Execute computer tool action."""
        if not hasattr(input_data, "action") or not input_data.action:
            raise ToolExecutionException("Computer tool requires 'action' parameter", recoverable=True)

        action_enum = E2BAction(input_data.action)
        schema_class = ACTION_SCHEMA_MAP.get(action_enum)

        if not schema_class:
            raise ToolExecutionException(
                f"Unsupported action: {input_data.action}",
                recoverable=True,
            )

        try:
            params_dict = input_data.model_dump(exclude={"action_type", "files", "action"})
            params = schema_class(**params_dict).model_dump(exclude_none=True)
        except Exception as e:
            raise ToolExecutionException(
                f"Invalid parameters for action '{input_data.action}': {str(e)}",
                recoverable=True,
            )

        # Map actions to E2B Sandbox API
        logger.info(f"E2BDesktopTool executing action={action_enum.value} params={params}")
        files: list[io.BytesIO] | None = None
        if action_enum == E2BAction.MOVE_MOUSE:
            x, y = params["coordinates"]
            self._desktop.move_mouse(x, y)
            result = f"Moved mouse to {x}, {y}. Verify the change in the screenshot."
        elif action_enum == E2BAction.CLICK_MOUSE:
            coordinates = params.get("coordinates")
            if coordinates:
                x, y = coordinates
                self._desktop.move_mouse(x, y)
            # Click behavior from parameters only
            button = params.get("button", "left")
            clicks = params.get("num_clicks") or 1
            for _ in range(clicks):
                if button == "left":
                    self._desktop.left_click()
                elif button == "right":
                    self._desktop.right_click()
                else:
                    raise ToolExecutionException("Only left/right buttons are supported", recoverable=True)
            result = f"Clicked {button} {clicks} times. Verify the effect in the screenshot."
            files = self._append_screenshot(files)
        elif action_enum == E2BAction.WAIT:
            duration = max(float(params.get("duration", 0)), 0.0)
            wait_ms = int(duration * 1000)
            self._desktop.wait(wait_ms)
            result = f"Waited for {duration} seconds. Proceed after verifying the screenshot."
            files = self._append_screenshot(files)
        elif action_enum == E2BAction.RUN_COMMAND:
            cmd = params["command"]
            timeout = params.get("timeout", 10)
            run_result = self._desktop.commands.run(cmd, timeout=timeout)
            stdout = run_result.stdout or ""
            stderr = run_result.stderr or ""
            combined = (
                (stdout + ("\n" if stdout and stderr else "") + stderr)
                if (stdout or stderr)
                else f"Command executed: {cmd}. Verify the outcome in the screenshot."
            )
            result = combined
            files = self._append_screenshot(files)
        elif action_enum == E2BAction.RUN_BACKGROUND_COMMAND:
            cmd = params["command"]
            self._desktop.commands.run(cmd, background=True)
            result = f"Started command: {cmd}. Verify the app/window appears in the screenshot."
        elif action_enum == E2BAction.SEND_KEY:
            key_name = params["name"]
            self._desktop.press(key_name)
            result = f"Pressed key: {key_name}. Verify the effect in the screenshot."
            files = self._append_screenshot(files)
        elif action_enum == E2BAction.TYPE_TEXT:
            text = params["text"]
            chunk_size = params.get("chunk_size")
            delay_in_ms = params.get("delay_in_ms")
            kwargs: dict[str, int] = {}
            if chunk_size is not None:
                kwargs["chunk_size"] = int(chunk_size)
            if delay_in_ms is not None:
                kwargs["delay_in_ms"] = int(delay_in_ms)
            self._desktop.write(text, **kwargs)
            result = f"Typed text: {text}. Verify the input field in the screenshot."
            files = self._append_screenshot(files)
        elif action_enum == E2BAction.LAUNCH:
            application = params["application"]
            self._desktop.launch(application)
            result = f"Launch executed for: {application}. Verify it opened in the screenshot."
        elif action_enum == E2BAction.TAKE_SCREENSHOT:
            files = self._append_screenshot(files, strict=True)
            result = "Screenshot was taken. Interpret the screenshot with FileReadTool."
        else:
            raise ToolExecutionException(
                f"Unsupported action: {input_data.action}. Supported: move_mouse, click_mouse, wait",
                recoverable=True,
            )

        # Include upload summary in result if available
        if self._uploads_report:
            summary = "; ".join(
                [
                    f"{r.get('name', '<unknown>')}: {r.get('status')}"
                    + (f" ({r.get('error')})" if r.get("status") == "error" and r.get("error") else "")
                    for r in self._uploads_report
                ]
            )
            result = f"{result}\nUploads: {summary}"

        response = {
            "action_type": "computer",
            "action": action_enum.value,
            "stream_url": self._stream_url,
            "result": result,
        }
        if files:
            response["files"] = files
        if self._uploads_report:
            response["uploads"] = self._uploads_report
            self._uploads_report = None
        return response

    def _append_screenshot(self, files: list[io.BytesIO] | None, strict: bool = False) -> list[io.BytesIO] | None:
        """Capture a screenshot and append it to files as BytesIO named screenshot.png.

        If strict is True, raises ToolExecutionException on failure; otherwise logs and returns files unchanged.
        """
        try:
            img_obj = self._desktop.screenshot()
            logger.info(f"E2BDesktopTool screenshot raw_type={type(img_obj).__name__}")
            raw = bytes(img_obj)
            fobj = io.BytesIO(raw)
            fobj.name = "screenshot.png"
            return (files or []) + [fobj]
        except Exception as e:
            if strict:
                raise ToolExecutionException(f"Screenshot capture failed: {e}", recoverable=True)
            logger.warning(f"E2BDesktopTool: automatic screenshot failed: {e}")
            return files

    def close(self) -> None:
        """Cleanup when nodes are not explicitly shut down."""
        try:
            if self._desktop:
                logger.info("Stopping E2B sandbox stream and killing sandbox")
                try:
                    self._desktop.stream.stop()
                except Exception:
                    logger.warning("Failed to stop E2B sandbox stream")
                self._desktop.kill()
        except Exception as e:
            logger.warning(f"E2B close() failed: {e}")
        finally:
            self.client = None
            self._desktop = None
            self._sandbox_id = None
            self._stream_url = None
            self.close_loop()

    def __del__(self):
        try:
            self.close()
        except Exception:
            logger.warning("E2B __del__ failed")

    @property
    def to_dict_exclude_params(self):
        """
        Property to define which parameters should be excluded when converting the class instance to a dictionary.

        Returns:
            dict: A dictionary defining the parameters to exclude.
        """
        return super().to_dict_exclude_params
