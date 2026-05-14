import asyncio
import io
import threading
from typing import Any, ClassVar, Coroutine, Literal

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, field_validator

from dynamiq.connections import CuaDesktop as CuaDesktopConnection
from dynamiq.connections.managers import ConnectionManager
from dynamiq.nodes import NodeGroup
from dynamiq.nodes.agents.exceptions import ToolExecutionException
from dynamiq.nodes.node import ConnectionNode
from dynamiq.nodes.tools.cua_desktop.types import (
    CuaAction,
    CuaActivateWindowSchema,
    CuaClickMouseSchema,
    CuaClipboardOperation,
    CuaClipboardSchema,
    CuaCloseWindowSchema,
    CuaDragMouseSchema,
    CuaGetApplicationWindowsSchema,
    CuaGetCurrentWindowIdSchema,
    CuaGetCursorPositionSchema,
    CuaGetScreenSizeSchema,
    CuaKeyAction,
    CuaLaunchSchema,
    CuaMaximizeWindowSchema,
    CuaMinimizeWindowSchema,
    CuaMouseAction,
    CuaMouseActionSchema,
    CuaMouseButton,
    CuaMoveMouseSchema,
    CuaOpenSchema,
    CuaRunCommandSchema,
    CuaScrollDirection,
    CuaScrollSchema,
    CuaSendKeySchema,
    CuaTakeScreenshotSchema,
    CuaToScreenCoordinatesSchema,
    CuaToScreenshotCoordinatesSchema,
    CuaTypeTextSchema,
    CuaWaitSchema,
)
from dynamiq.nodes.tools.utils import sanitize_filename
from dynamiq.runnables import RunnableConfig
from dynamiq.types.cancellation import check_cancellation
from dynamiq.utils.logger import logger

ACTION_SCHEMA_MAP: dict[CuaAction, type] = {
    # Mouse Actions
    CuaAction.MOVE_MOUSE: CuaMoveMouseSchema,
    CuaAction.CLICK_MOUSE: CuaClickMouseSchema,
    CuaAction.MOUSE_ACTION: CuaMouseActionSchema,
    CuaAction.DRAG_MOUSE: CuaDragMouseSchema,
    CuaAction.GET_CURSOR_POSITION: CuaGetCursorPositionSchema,
    # Keyboard Actions
    CuaAction.TYPE_TEXT: CuaTypeTextSchema,
    CuaAction.SEND_KEY: CuaSendKeySchema,
    # Scrolling Actions
    CuaAction.SCROLL: CuaScrollSchema,
    # Screen Actions
    CuaAction.TAKE_SCREENSHOT: CuaTakeScreenshotSchema,
    CuaAction.GET_SCREEN_SIZE: CuaGetScreenSizeSchema,
    # Window Management
    CuaAction.LAUNCH: CuaLaunchSchema,
    CuaAction.OPEN: CuaOpenSchema,
    CuaAction.GET_APPLICATION_WINDOWS: CuaGetApplicationWindowsSchema,
    CuaAction.ACTIVATE_WINDOW: CuaActivateWindowSchema,
    CuaAction.GET_CURRENT_WINDOW_ID: CuaGetCurrentWindowIdSchema,
    CuaAction.MAXIMIZE_WINDOW: CuaMaximizeWindowSchema,
    CuaAction.MINIMIZE_WINDOW: CuaMinimizeWindowSchema,
    CuaAction.CLOSE_WINDOW: CuaCloseWindowSchema,
    # Clipboard Actions
    CuaAction.CLIPBOARD: CuaClipboardSchema,
    # Accessibility
    CuaAction.TO_SCREEN_COORDINATES: CuaToScreenCoordinatesSchema,
    CuaAction.TO_SCREENSHOT_COORDINATES: CuaToScreenshotCoordinatesSchema,
    # Other Actions
    CuaAction.WAIT: CuaWaitSchema,
    CuaAction.RUN_COMMAND: CuaRunCommandSchema,
}

DESCRIPTION_CUA_DESKTOP = """## CUA Sandbox Desktop Tool
### Overview
Use this tool to control a virtual desktop by performing comprehensive computer
actions including mouse control, keyboard input, window management, file operations,
clipboard management, and more.

### Supported actions (set via `action`)

#### Mouse Actions
- `move_mouse`: Move the cursor to specified screen coordinates
    - `coordinates` (list[int]): [x, y] absolute screen coordinates
    - `delay` (float): Optional delay in seconds after the action
- `click_mouse`: Perform mouse clicks
    - `coordinates` (list[int]): [x, y] absolute screen coordinates
    - `button` (str): 'left'/'right'/'middle'
    - `num_clicks` (int): number of clicks (default: 1; use 2 for double-click)
    - `delay` (float): Optional delay in seconds after the action
- `mouse_action`: Press or release a mouse button
    - `action_type` (str): 'down'/'up'
    - `coordinates` (list[int]): [x, y] absolute screen coordinates
    - `button` (str): 'left'/'right'/'middle'
    - `delay` (float): Optional delay in seconds after the action
- `drag_mouse`: Drag along a path of coordinates
    - `path` (list[list[int]]): [[x1, y1], [x2, y2], ...]
    - `button` (str): 'left'/'right'/'middle'
    - `duration` (float): duration in seconds (default: 0.5)
    - `delay` (float): Optional delay in seconds after the action
- `get_cursor_position`: Get the current cursor position

#### Keyboard Actions
- `type_text`: Type text into the currently focused input
    - `text` (str): text to type
    - `delay` (float): Optional delay in seconds after the action
- `send_key`: Send a key or key combination
    - `selected_keys` (list[str] | str): 'enter' or ['ctrl', 'c']
    - `action_type` (str): 'press'/'down'/'up'
    - `delay` (float): Optional delay in seconds after the action

#### Scrolling Actions
- `scroll`: Scroll the mouse wheel
    - `x` (int): x scroll amount (default: 0)
    - `y` (int): y scroll amount (default: 0)
    - `clicks` (int): number of clicks (use with direction, alternative to x/y)
    - `direction` (str): 'up'/'down' (required when using clicks)
    - `delay` (float): Optional delay in seconds after the action

#### Screen Actions
- `take_screenshot`: Capture a screenshot
    - `boxes` (list[list[int]]): list of boxes to capture (default: None)
    - `box_color` (str): color of the box (default: None)
    - `box_thickness` (int): thickness of the box (default: None)
    - `scale_factor` (float): scale factor (default: None)
- `get_screen_size`: Get the screen dimensions

#### Window Management
- `launch`: Launch an application with optional arguments
    - `app` (str): application to launch
    - `args` (list[str]): arguments to pass to the application
- `open`: Open a target using the system's default handler
    - `target` (str): target to open
- `get_application_windows`: Get all window identifiers for an application
    - `app` (str): application to get windows for
- `activate_window`: Bring a window to the foreground and focus it
    - `window_id` (str | int): window id to activate
- `get_current_window_id`: Get the identifier of the currently active window
- `maximize_window`: Maximize a window
    - `window_id` (str | int): window id to maximize
- `minimize_window`: Minimize a window
    - `window_id` (str | int): window id to minimize
- `close_window`: Close a window
    - `window_id` (str | int): window id to close

#### Clipboard Actions
- `clipboard`: Get or set clipboard content
    - `operation` (str): 'get'/'set'
    - `text` (str): text to set

#### Accessibility
- `to_screen_coordinates`: Convert screenshot coordinates to screen coordinates
    - `x` (float): x coordinate
    - `y` (float): y coordinate
- `to_screenshot_coordinates`: Convert screen coordinates to screenshot coordinates
    - `x` (float): x coordinate
    - `y` (float): y coordinate

#### Other Actions
- `wait`: Pause execution for a specified number of seconds
    - `duration` (float): duration in seconds
- `run_command`: Run shell command and return structured result
    - `command` (str): command to run

### Usage Examples

1) Move cursor to coordinates
```json
{
  "action": "move_mouse",
  "coordinates": [500, 300]
}
```

2) Left click at specified position
```json
{
  "action": "click_mouse",
  "coordinates": [500, 300],
  "button": "left",
  "num_clicks": 1
}
```

3) Double click
```json
{
  "action": "click_mouse",
  "coordinates": [500, 300],
  "num_clicks": 2
}
```

4) Type text
```json
{
  "action": "type_text",
  "text": "Hello, World!"
}
```

5) Press a key combination
```json
{
  "action": "send_key",
  "selected_keys": ["ctrl", "c"]
}
```

6) Scroll down
```json
{
  "action": "scroll",
  "clicks": 3,
  "direction": "down"
}
```

7) Take a screenshot
```json
{
  "action": "take_screenshot"
}
```

8) Launch an application
```json
{
  "action": "launch",
  "app": "firefox",
  "args": ["--private-window"]
}
```

9) Wait 3 seconds
```json
{
  "action": "wait",
  "duration": 3
}
```

10) Run a shell command
```json
{
  "action": "run_command",
  "command": "ls -la"
}
```

### Execution Guidelines
- Verify each step with `take_screenshot` before proceeding
- After launching GUI apps with `launch`, insert a short `wait`, then confirm with `take_screenshot`
- Interpret each screenshot and plan the next action based on what changed
- For web interactions, zoom out or scroll so required elements are visible before clicking
- If an interaction fails, move the mouse to a neutral area, then retry the click
- If the first screenshot is blank/black, click the screen center and take another screenshot
- Before typing: ensure the input is focused; if needed, click it, then clear with `send_key` with
`selected_keys` `["ctrl", "a"]` followed by `send_key` with `selected_keys` `"backspace"`; then use `type_text`
- After typing in a search/input field, activate it by pressing `send_key` with `selected_keys` `"enter"`
or clicking the submit/search button
- Insert short `wait` steps between critical actions to improve reliability

### Best Practices
- For browser tasks, prefer Firefox with the `launch` action
- For file operations, prefer the `run_command` action with
the `cp`, `mv`, `rm`, `mkdir`, `ls`, `cat`, `echo`, `touch`, `chmod`, `chown`, `chgrp` commands
- Always capture a `take_screenshot` after actions to verify the result
- After navigation or app launches, verify state changes with `take_screenshot` before continuing
- Explain each action and the reason, especially for multi-step sequences
- Before typing in input field, make sure it is focused
- To focus on search bar in browser, use `send_key` with `selected_keys` `["ctrl", "l"]` to activate browser address bar
- When uploading files, make sure to remember that the default directory is `/home/ubuntu/`
"""


class CuaDesktopToolInputSchema(BaseModel):
    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)

    action: CuaAction = Field(
        ...,
        description="""
    Specifies the action to perform. Supports mouse, keyboard, window, file system,
    clipboard, and other desktop operations.
    """,
    )

    files: str | list[str] | io.BytesIO | list[io.BytesIO] | bytes | list[bytes] | None = Field(
        default=None,
        description="Files to upload to /home/ubuntu/. Accepts names (str or list[str]) mapped from storage, "
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


class CuaDesktopTool(ConnectionNode):
    group: Literal[NodeGroup.TOOLS] = NodeGroup.TOOLS
    name: str = "cua-sandbox"
    description: str = DESCRIPTION_CUA_DESKTOP
    connection: CuaDesktopConnection
    input_schema: ClassVar[type[CuaDesktopToolInputSchema]] = CuaDesktopToolInputSchema
    timeout: int = 3600
    is_files_allowed: bool = True
    _force_thread_executor: ClassVar[bool] = True

    _computer: Any | None = PrivateAttr(default=None)
    _loop = PrivateAttr(default=None)
    _loop_thread = PrivateAttr(default=None)
    _init_lock: threading.Lock = PrivateAttr(default_factory=threading.Lock)
    _upload_counter: int = PrivateAttr(default=0)

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

    async def _upload_files(self, files_field: Any) -> list[dict[str, Any]] | None:
        """Upload provided file object(s) to /home/ubuntu/ and build a status report."""
        if not files_field:
            return None

        def _to_list(v):
            return v if isinstance(v, list) else [v]

        reports: list[dict[str, Any]] = []
        for fobj in _to_list(files_field):
            raw_name = getattr(fobj, "name", None)
            name = sanitize_filename(filename=raw_name)
            if not name:
                with self._init_lock:
                    self._upload_counter += 1
                    name = f"file_{self._upload_counter}"
            dest_path = f"/home/ubuntu/{name}"
            try:
                if isinstance(fobj, (bytes, bytearray)):
                    data = fobj
                elif isinstance(fobj, io.BytesIO):
                    data = fobj.getvalue()
                elif isinstance(fobj, str):
                    # Handle string file inputs by treating them as file paths or content
                    data = fobj.encode("utf-8")
                else:
                    data = bytes(str(fobj), encoding="utf-8")
                if hasattr(self._computer.interface, "write_bytes"):
                    await self._computer.interface.write_bytes(dest_path, data)
                else:
                    text_content = data.decode("utf-8") if isinstance(data, bytes) else str(data)
                    await self._computer.interface.write_text(dest_path, text_content)
                reports.append({"name": name, "path": dest_path, "status": "uploaded"})
            except Exception as e:
                reports.append({"name": name, "path": dest_path, "status": "error", "error": str(e)})
        if reports:
            logger.info(f"CuaDesktopTool uploads: {reports}")
            return reports
        return None

    async def _init_client(self):
        """Initialize CUA computer (idempotent)."""
        # Check if already initialized
        if self._computer is not None:
            return

        # Use lock only for the check-and-set to avoid holding it during async operations
        with self._init_lock:
            if self._computer is None:
                # Create computer instance
                self._computer = self.connection.connect()
            else:
                # Another coroutine already set it
                return

        # Do async initialization outside the lock to avoid deadlock
        try:
            await self._computer.run()
            logger.debug(f"CUA computer initialized: {self.connection.computer_name}")
        except Exception:
            # If initialization fails, reset _computer so it can be retried
            with self._init_lock:
                self._computer = None
            raise

    def _run_async(self, coro) -> Any:
        """Run coroutine in the persistent event loop."""
        return self._run_in_loop(coro)

    def execute(
        self, input_data: CuaDesktopToolInputSchema, config: RunnableConfig | None = None, **kwargs
    ) -> dict[str, Any]:
        """
        Executes the tool synchronously with the provided input.

        Always dispatches onto a dedicated background event loop to avoid cross-loop issues.

        Args:
            input_data (CuaDesktopToolInputSchema): Input data for the tool execution.
            config (RunnableConfig, optional): Configuration for the runnable, including callbacks.
            **kwargs: Additional arguments passed to the execution context.

        Returns:
            dict[str, Any]: A dictionary containing the tool's output.
        """
        return self._run_async(self.execute_async(input_data, config, **kwargs))

    async def execute_async(
        self, input_data: CuaDesktopToolInputSchema, config: RunnableConfig | None = None, **kwargs
    ) -> dict[str, Any]:
        """
        Executes a task on a CUA computer instance using the computer interface based on the input action type.

        Args:
            input_data (CuaDesktopToolInputSchema): The schema describing the action to perform.
            config (RunnableConfig, optional): Execution config if applicable.
            **kwargs: Additional arguments.
        Returns:
            dict[str, Any]: The result of the action.
        Raises:
            ToolExecutionException: If the input is invalid or execution fails.
        """
        logger.info(f"Tool {self.name} - {self.id}: started with input:\n{input_data.model_dump()}")
        check_cancellation(config)

        await self._init_client()

        # Filter out screenshots from previous actions
        files_to_upload = self._filter_screenshots(input_data.files)
        uploads_report = None
        if files_to_upload:
            logger.info(f"CuaDesktopTool uploading files: {files_to_upload}")
            uploads_report = await self._upload_files(files_to_upload)

        try:
            result = await self.execute_command(input_data, uploads_report)
        except Exception as e:
            raise ToolExecutionException(f"Error message: {e}", recoverable=True) from e

        logger.info(f"Tool {self.name} - {self.id}: finished with result:\n{str(result)[:200]}...")

        files = None
        if isinstance(result, dict) and "files" in result:
            files = result.pop("files")

        response: dict[str, Any] = {"content": result}
        if files:
            response["files"] = files
        return response

    def _filter_screenshots(self, files: str | list[str] | io.BytesIO | list[io.BytesIO] | bytes | list[bytes]) -> list:
        """Filter out screenshot files that should not be uploaded back to the computer instance.

        Screenshots are outputs from previous actions and should not be re-uploaded.
        """
        if not files:
            return []

        # Normalize to list
        if not isinstance(files, list):
            files = [files]

        filtered = []
        for f in files:
            # Check if file has a name attribute (BytesIO objects)
            if hasattr(f, "name"):
                # Skip files named screenshot.png or containing "screenshot" in the name
                if isinstance(f.name, str) and "screenshot" in f.name.lower():
                    logger.info(f"Filtering out screenshot file: {f.name}")
                    continue
            filtered.append(f)

        return filtered

    async def execute_command(
        self, input_data: CuaDesktopToolInputSchema, uploads_report: list[dict[str, Any]] | None = None
    ) -> dict[str, Any]:
        """Execute computer command."""
        if not hasattr(input_data, "action") or not input_data.action:
            raise ToolExecutionException("Computer tool requires 'action' parameter", recoverable=True)

        action_enum = CuaAction(input_data.action)
        schema_class = ACTION_SCHEMA_MAP.get(action_enum)

        if not schema_class:
            raise ToolExecutionException(
                f"Unsupported action: {input_data.action}",
                recoverable=True,
            )

        try:
            params_dict = input_data.model_dump(exclude={"files", "action"})
            params = schema_class(**params_dict).model_dump(exclude_none=True)
        except Exception as e:
            raise ToolExecutionException(
                f"Invalid parameters for action '{input_data.action}': {str(e)}",
                recoverable=True,
            )

        # Map actions to CUA Computer API
        logger.info(f"CuaDesktopTool executing action={action_enum.value} params={params}")
        files: list[io.BytesIO] | None = None
        if action_enum == CuaAction.MOVE_MOUSE:
            x, y = params["coordinates"]
            delay = params.get("delay")
            await self._computer.interface.move_cursor(x, y, delay=delay)
            result = f"Moved cursor to ({x}, {y})."
        elif action_enum == CuaAction.CLICK_MOUSE:
            coordinates = params.get("coordinates")
            button = params.get("button", "left")
            num_clicks = params.get("num_clicks", 1)
            delay = params.get("delay")

            # Move to coordinates if provided
            if coordinates:
                x, y = coordinates
                await self._computer.interface.move_cursor(x, y, delay=None)

            # Perform clicks
            for _ in range(num_clicks):
                if button == CuaMouseButton.LEFT:
                    await self._computer.interface.left_click(delay=delay if _ == num_clicks - 1 else None)
                elif button == CuaMouseButton.RIGHT:
                    await self._computer.interface.right_click(delay=delay if _ == num_clicks - 1 else None)
                elif button == CuaMouseButton.MIDDLE:
                    # Middle click via mouse_down/mouse_up
                    await self._computer.interface.mouse_down(button=button, delay=None)
                    await self._computer.interface.mouse_up(button=button, delay=delay if _ == num_clicks - 1 else None)

            click_type = "single" if num_clicks == 1 else str(num_clicks)
            location = f" at ({coordinates[0]}, {coordinates[1]})" if coordinates else " at current position"
            result = f"Performed {click_type} {button} click(s){location}."
            files = await self._append_screenshot(files)
        elif action_enum == CuaAction.MOUSE_ACTION:
            action_type = params["action_type"]
            coordinates = params.get("coordinates")
            button = params.get("button", "left")
            delay = params.get("delay")

            # Move to coordinates if provided
            if coordinates:
                x, y = coordinates
                await self._computer.interface.move_cursor(x, y, delay=None)

            if action_type == CuaMouseAction.DOWN:
                await self._computer.interface.mouse_down(x=None, y=None, button=button, delay=delay)
                result = (
                    f"Pressed {button} button"
                    + (f" at ({coordinates[0]}, {coordinates[1]})" if coordinates else " at current position")
                    + "."
                )
            elif action_type == CuaMouseAction.UP:
                await self._computer.interface.mouse_up(x=None, y=None, button=button, delay=delay)
                result = (
                    f"Released {button} button"
                    + (f" at ({coordinates[0]}, {coordinates[1]})" if coordinates else " at current position")
                    + "."
                )
        elif action_enum == CuaAction.DRAG_MOUSE:
            path = params["path"]
            button = params.get("button", "left")
            duration = params.get("duration", 0.5)
            delay = params.get("delay")

            # If single coordinate, use drag_to; otherwise use drag
            if len(path) == 1:
                x, y = path[0]
                await self._computer.interface.drag_to(x, y, button=button, duration=duration, delay=delay)
                result = f"Dragged to ({x}, {y}) using {button} button."
            else:
                # Convert list of lists to list of tuples
                path_tuples = [(p[0], p[1]) for p in path]
                await self._computer.interface.drag(path_tuples, button=button, duration=duration, delay=delay)
                result = f"Dragged along path with {len(path)} points using {button} button."
            files = await self._append_screenshot(files)
        elif action_enum == CuaAction.GET_CURSOR_POSITION:
            position = await self._computer.interface.get_cursor_position()
            result = f"Cursor position: {position}"
        elif action_enum == CuaAction.TYPE_TEXT:
            text = params["text"]
            delay = params.get("delay")
            await self._computer.interface.type_text(text, delay=delay)
            result = f"Typed text: {text}."
            files = await self._append_screenshot(files)
        elif action_enum == CuaAction.SEND_KEY:
            selected_keys = params["selected_keys"]
            action_type = params.get("action_type", "press")
            delay = params.get("delay")

            if isinstance(selected_keys, str):
                # Single key
                if action_type == CuaKeyAction.PRESS:
                    await self._computer.interface.press(selected_keys, delay=delay)
                    result = f"Pressed key: {selected_keys}."
                elif action_type == CuaKeyAction.DOWN:
                    await self._computer.interface.key_down(selected_keys, delay=delay)
                    result = f"Key down: {selected_keys}."
                elif action_type == CuaKeyAction.UP:
                    await self._computer.interface.key_up(selected_keys, delay=delay)
                    result = f"Key up: {selected_keys}."
            else:
                # Key combination
                if action_type == CuaKeyAction.PRESS:
                    await self._computer.interface.hotkey(*selected_keys, delay=delay)
                    result = f"Pressed key combination: {'+'.join(selected_keys)}."
                elif action_type == CuaKeyAction.DOWN:
                    # Press multiple keys down in sequence
                    for key in selected_keys:
                        await self._computer.interface.key_down(key, delay=None)
                    if delay:
                        await asyncio.sleep(delay)
                    result = f"Keys down: {'+'.join(selected_keys)}."
                elif action_type == CuaKeyAction.UP:
                    # Release multiple keys in reverse sequence
                    for key in reversed(selected_keys):
                        await self._computer.interface.key_up(key, delay=None)
                    if delay:
                        await asyncio.sleep(delay)
                    result = f"Keys up: {'+'.join(selected_keys)}."
            files = await self._append_screenshot(files)
        elif action_enum == CuaAction.SCROLL:
            x = params.get("x", 0)
            y = params.get("y", 0)
            clicks = params.get("clicks")
            direction = params.get("direction")
            delay = params.get("delay")

            if clicks is not None and direction is not None:
                # Use scroll_up/scroll_down
                if direction == CuaScrollDirection.DOWN:
                    await self._computer.interface.scroll_down(clicks=clicks, delay=delay)
                    result = f"Scrolled down {clicks} clicks."
                elif direction == CuaScrollDirection.UP:
                    await self._computer.interface.scroll_up(clicks=clicks, delay=delay)
                    result = f"Scrolled up {clicks} clicks."
            else:
                # Use x/y scroll
                await self._computer.interface.scroll(x, y, delay=delay)
                result = f"Scrolled by ({x}, {y})."
            files = await self._append_screenshot(files)
        elif action_enum == CuaAction.TAKE_SCREENSHOT:
            boxes = params.get("boxes")
            box_color = params.get("box_color", "#FF0000")
            box_thickness = params.get("box_thickness", 2)
            scale_factor = params.get("scale_factor", 1.0)
            # Convert list of lists to list of tuples if provided
            boxes_tuples = [(b[0], b[1], b[2], b[3]) for b in boxes] if boxes else None
            img_bytes = await self._computer.interface.screenshot(
                boxes=boxes_tuples, box_color=box_color, box_thickness=box_thickness, scale_factor=scale_factor
            )
            fobj = io.BytesIO(img_bytes)
            fobj.name = "screenshot.png"
            files = [fobj]
            result = "Screenshot captured."
        elif action_enum == CuaAction.GET_SCREEN_SIZE:
            size = await self._computer.interface.get_screen_size()
            result = f"Screen size: {size}"
        elif action_enum == CuaAction.WAIT:
            duration = max(float(params.get("duration", 0)), 0.0)
            # CUA doesn't have a specific wait method, use asyncio.sleep
            await asyncio.sleep(duration)
            result = f"Waited for {duration} seconds. Proceed after verifying the screenshot."
            files = await self._append_screenshot(files)
        elif action_enum == CuaAction.RUN_COMMAND:
            cmd = params["command"]
            run_result = await self._computer.interface.run_command(cmd)
            stdout = run_result.stdout or ""
            stderr = run_result.stderr or ""
            combined = (
                (stdout + ("\n" if stdout and stderr else "") + stderr)
                if (stdout or stderr)
                else f"Command executed: {cmd}. Verify the outcome in the screenshot."
            )
            result = combined
            files = await self._append_screenshot(files)
        elif action_enum == CuaAction.LAUNCH:
            app = params["app"]
            args = params.get("args")
            pid = await self._computer.interface.launch(app, args=args)
            result = f"Launched application: {app}" + (f" with PID: {pid}" if pid else "")
        elif action_enum == CuaAction.OPEN:
            target = params["target"]
            await self._computer.interface.open(target)
            result = f"Opened: {target}"
            files = await self._append_screenshot(files)
        elif action_enum == CuaAction.GET_APPLICATION_WINDOWS:
            app = params["app"]
            windows = await self._computer.interface.get_application_windows(app)
            result = f"Windows for {app}: {windows}"
        elif action_enum == CuaAction.ACTIVATE_WINDOW:
            window_id = params["window_id"]
            await self._computer.interface.activate_window(window_id)
            result = f"Activated window: {window_id}"
            files = await self._append_screenshot(files)
        elif action_enum == CuaAction.GET_CURRENT_WINDOW_ID:
            window_id = await self._computer.interface.get_current_window_id()
            result = f"Current window ID: {window_id}"
        elif action_enum == CuaAction.MAXIMIZE_WINDOW:
            window_id = params["window_id"]
            await self._computer.interface.maximize_window(window_id)
            result = f"Maximized window: {window_id}"
            files = await self._append_screenshot(files)
        elif action_enum == CuaAction.MINIMIZE_WINDOW:
            window_id = params["window_id"]
            await self._computer.interface.minimize_window(window_id)
            result = f"Minimized window: {window_id}"
            files = await self._append_screenshot(files)
        elif action_enum == CuaAction.CLOSE_WINDOW:
            window_id = params["window_id"]
            await self._computer.interface.close_window(window_id)
            result = f"Closed window: {window_id}"
            files = await self._append_screenshot(files)
        elif action_enum == CuaAction.CLIPBOARD:
            operation = params["operation"]
            if operation == CuaClipboardOperation.SET:
                text = params["text"]
                await self._computer.interface.set_clipboard(text)
                result = f"Set clipboard to: {text}"
            elif operation == CuaClipboardOperation.GET:
                content = await self._computer.interface.copy_to_clipboard()
                result = f"Clipboard content: {content}"
        elif action_enum == CuaAction.TO_SCREEN_COORDINATES:
            x = params["x"]
            y = params["y"]
            screen_x, screen_y = await self._computer.interface.to_screen_coordinates(x, y)
            result = f"Screen coordinates: ({screen_x}, {screen_y})"
        elif action_enum == CuaAction.TO_SCREENSHOT_COORDINATES:
            x = params["x"]
            y = params["y"]
            screenshot_x, screenshot_y = await self._computer.interface.to_screenshot_coordinates(x, y)
            result = f"Screenshot coordinates: ({screenshot_x}, {screenshot_y})"
        else:
            raise ToolExecutionException(
                f"Unsupported action: {input_data.action}",
                recoverable=True,
            )

        # Include upload summary in result if available
        if uploads_report:
            summary = "; ".join(
                [
                    f"{r.get('name', '<unknown>')}: {r.get('status')}"
                    + (f" ({r.get('error')})" if r.get("status") == "error" and r.get("error") else "")
                    for r in uploads_report
                ]
            )
            result = f"{result}\nUploads: {summary}"

        response = {
            "action": action_enum.value,
            "result": result,
        }
        if files:
            response["files"] = files
        if uploads_report:
            response["uploads"] = uploads_report
        return response

    async def _append_screenshot(self, files: list[io.BytesIO] | None, strict: bool = False) -> list[io.BytesIO] | None:
        """Capture a screenshot and append it to files as BytesIO named screenshot.png.

        If strict is True, raises ToolExecutionException on failure; otherwise logs and returns files unchanged.
        """
        try:
            img_bytes = await self._computer.interface.screenshot()
            logger.info(f"CuaDesktopTool screenshot captured, size={len(img_bytes)} bytes")
            fobj = io.BytesIO(img_bytes)
            fobj.name = "screenshot.png"
            return (files or []) + [fobj]
        except Exception as e:
            if strict:
                raise ToolExecutionException(f"Screenshot capture failed: {e}", recoverable=True)
            logger.warning(f"CuaDesktopTool: automatic screenshot failed: {e}")
            return files

    def close(self) -> None:
        """Cleanup when nodes are not explicitly shut down."""
        try:
            if self._computer:
                logger.info("Cleaning up CUA computer connection")
                try:
                    if hasattr(self._computer, "stop"):
                        stop_method = getattr(self._computer, "stop")
                        if asyncio.iscoroutinefunction(stop_method):
                            # Method is async, run it in the event loop
                            if self._loop and not self._loop.is_closed():
                                future = asyncio.run_coroutine_threadsafe(stop_method(), self._loop)
                                future.result(timeout=5.0)  # Wait up to 5 seconds for cleanup
                            else:
                                logger.warning("Event loop not available for async cleanup")
                        else:
                            # Method is synchronous
                            stop_method()
                except Exception as e:
                    logger.warning(f"Failed to cleanup CUA computer: {e}")
        except Exception as e:
            logger.warning(f"CUA close() failed: {e}")
        finally:
            self._computer = None
            self.close_loop()

    def __del__(self):
        try:
            self.close()
        except Exception:
            logger.warning("CUA __del__ failed")

    @property
    def to_dict_exclude_params(self):
        """
        Property to define which parameters should be excluded when converting the class instance to a dictionary.

        Returns:
            dict: A dictionary defining the parameters to exclude.
        """
        return super().to_dict_exclude_params
