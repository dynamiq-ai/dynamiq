from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator


class CuaMouseButton(str, Enum):
    """Mouse buttons supported by Cua desktop tool."""

    LEFT = "left"
    RIGHT = "right"
    MIDDLE = "middle"


class CuaMouseAction(str, Enum):
    """Mouse button press/release actions."""

    DOWN = "down"
    UP = "up"


class CuaKeyAction(str, Enum):
    """Key press/release actions."""

    PRESS = "press"
    DOWN = "down"
    UP = "up"


class CuaScrollDirection(str, Enum):
    """Scroll direction."""

    UP = "up"
    DOWN = "down"


class CuaClipboardOperation(str, Enum):
    """Clipboard operation."""

    GET = "get"
    SET = "set"


class CuaAction(str, Enum):
    """Actions supported by Cua desktop tool."""

    # Mouse Actions
    MOVE_MOUSE = "move_mouse"
    CLICK_MOUSE = "click_mouse"
    MOUSE_ACTION = "mouse_action"
    DRAG_MOUSE = "drag_mouse"
    GET_CURSOR_POSITION = "get_cursor_position"

    # Keyboard Actions
    TYPE_TEXT = "type_text"
    SEND_KEY = "send_key"

    # Scroll Actions
    SCROLL = "scroll"

    # Screen Actions
    TAKE_SCREENSHOT = "take_screenshot"
    GET_SCREEN_SIZE = "get_screen_size"

    # Window Management
    LAUNCH = "launch"
    OPEN = "open"
    GET_APPLICATION_WINDOWS = "get_application_windows"
    ACTIVATE_WINDOW = "activate_window"
    GET_CURRENT_WINDOW_ID = "get_current_window_id"
    MAXIMIZE_WINDOW = "maximize_window"
    MINIMIZE_WINDOW = "minimize_window"
    CLOSE_WINDOW = "close_window"

    # Clipboard Actions
    CLIPBOARD = "clipboard"

    # Accessibility
    TO_SCREEN_COORDINATES = "to_screen_coordinates"
    TO_SCREENSHOT_COORDINATES = "to_screenshot_coordinates"

    # Other Actions
    WAIT = "wait"
    RUN_COMMAND = "run_command"


# Mouse Actions


class CuaMoveMouseSchema(BaseModel):
    """move_mouse: Move the cursor to the specified screen coordinates.

    - coordinates: [x, y] absolute screen coordinates
    - delay: Optional delay in seconds after the action.
    """

    action: Literal[CuaAction.MOVE_MOUSE] = CuaAction.MOVE_MOUSE
    coordinates: list[int] = Field(..., description="[x, y] absolute screen coordinates")
    delay: float | None = Field(default=None, description="Optional delay in seconds after the action")

    @field_validator("coordinates")
    @classmethod
    def validate_coordinates(cls, v: list[int]) -> list[int]:
        """Validate that coordinates has exactly 2 elements [x, y]."""
        if len(v) != 2:
            raise ValueError(f"coordinates must have exactly 2 elements [x, y], got {len(v)} elements: {v}")
        return v


class CuaClickMouseSchema(BaseModel):
    """click_mouse: Perform mouse clicks at specified coordinates.

    - coordinates: Optional [x, y] to move before clicking; omit to click current position
    - button: Mouse button to click ('left', 'right', 'middle', default: 'left')
    - num_clicks: Number of clicks (default: 1; use 2 for double-click)
    - delay: Optional delay in seconds after the action.
    """

    action: Literal[CuaAction.CLICK_MOUSE] = CuaAction.CLICK_MOUSE
    coordinates: list[int] | None = Field(
        default=None, description="Optional [x, y] to move before clicking; omit to click current position"
    )
    button: CuaMouseButton = Field(default=CuaMouseButton.LEFT, description="Mouse button to click")
    num_clicks: int = Field(default=1, description="Number of clicks (1 for single, 2 for double, etc.)")
    delay: float | None = Field(default=None, description="Optional delay in seconds after the action")

    @field_validator("coordinates")
    @classmethod
    def validate_coordinates(cls, v: list[int] | None) -> list[int] | None:
        """Validate that coordinates has exactly 2 elements [x, y] if provided."""
        if v is not None and len(v) != 2:
            raise ValueError(f"coordinates must have exactly 2 elements [x, y], got {len(v)} elements: {v}")
        return v


class CuaMouseActionSchema(BaseModel):
    """mouse_action: Press or release a mouse button at specified coordinates.

    - action_type: 'down' to press, 'up' to release
    - coordinates: Optional [x, y] to move before action; omit to use current position
    - button: Mouse button ('left', 'right', 'middle', default: 'left')
    - delay: Optional delay in seconds after the action.
    """

    action: Literal[CuaAction.MOUSE_ACTION] = CuaAction.MOUSE_ACTION
    action_type: CuaMouseAction = Field(..., description="'down' to press, 'up' to release")
    coordinates: list[int] | None = Field(
        default=None, description="Optional [x, y] to move before action; omit to use current position"
    )
    button: CuaMouseButton = Field(default=CuaMouseButton.LEFT, description="Mouse button")
    delay: float | None = Field(default=None, description="Optional delay in seconds after the action")

    @field_validator("coordinates")
    @classmethod
    def validate_coordinates(cls, v: list[int] | None) -> list[int] | None:
        """Validate that coordinates has exactly 2 elements [x, y] if provided."""
        if v is not None and len(v) != 2:
            raise ValueError(f"coordinates must have exactly 2 elements [x, y], got {len(v)} elements: {v}")
        return v


class CuaDragMouseSchema(BaseModel):
    """drag_mouse: Drag from current position or along a path.

    - path: List of [x, y] coordinate pairs defining the drag path.
    If single pair, drags from current position to that point.
    - button: The mouse button to use ('left', 'middle', 'right', default: 'left')
    - duration: Total time in seconds that the drag operation should take (default: 0.5)
    - delay: Optional delay in seconds after the action.
    """

    action: Literal[CuaAction.DRAG_MOUSE] = CuaAction.DRAG_MOUSE
    path: list[list[int]] = Field(..., description="List of [x, y] coordinate pairs defining the drag path")
    button: CuaMouseButton = Field(default=CuaMouseButton.LEFT, description="The mouse button to use")
    duration: float = Field(default=0.5, description="Total time in seconds that the drag operation should take")
    delay: float | None = Field(default=None, description="Optional delay in seconds after the action")

    @field_validator("path")
    @classmethod
    def validate_path_coordinates(cls, v: list[list[int]]) -> list[list[int]]:
        """Validate that each coordinate pair has exactly 2 elements [x, y]."""
        if not v:
            raise ValueError("path cannot be empty")
        for i, coord in enumerate(v):
            if len(coord) != 2:
                raise ValueError(f"path[{i}] must have exactly 2 elements [x, y], got {len(coord)} elements: {coord}")
        return v


class CuaGetCursorPositionSchema(BaseModel):
    """get_cursor_position: Get the current cursor position."""

    action: Literal[CuaAction.GET_CURSOR_POSITION] = CuaAction.GET_CURSOR_POSITION


# Keyboard Actions


class CuaTypeTextSchema(BaseModel):
    """type_text: Type text into the current focused input.

    - text: the text to type
    - delay: Optional delay in seconds after the action.
    """

    action: Literal[CuaAction.TYPE_TEXT] = CuaAction.TYPE_TEXT
    text: str = Field(..., description="Text to type")
    delay: float | None = Field(default=None, description="Optional delay in seconds after the action")


class CuaSendKeySchema(BaseModel):
    """send_key: Send a key or key combination.

    - selected_keys: Single key or list of keys to press together (e.g., "enter", ["ctrl", "c"])
    - action_type: 'press' for press+release, 'down' to hold, 'up' to release (default: 'press')
    - delay: Optional delay in seconds after the action.
    """

    action: Literal[CuaAction.SEND_KEY] = CuaAction.SEND_KEY
    selected_keys: str | list[str] = Field(..., description="Single key or list of keys to press together")
    action_type: CuaKeyAction = Field(default=CuaKeyAction.PRESS, description="'press', 'down', or 'up'")
    delay: float | None = Field(default=None, description="Optional delay in seconds after the action")

    @field_validator("selected_keys")
    @classmethod
    def validate_selected_keys(cls, v: str | list[str]) -> str | list[str]:
        """Ensure keys is not empty."""
        if isinstance(v, list) and not v:
            raise ValueError("selected_keys list cannot be empty")
        if isinstance(v, str) and not v:
            raise ValueError("selected_keys string cannot be empty")
        return v


# Scrolling Actions


class CuaScrollSchema(BaseModel):
    """scroll: Scroll the mouse wheel.

    - x: Horizontal scroll amount (positive = right, negative = left), default: 0
    - y: Vertical scroll amount (positive = up, negative = down), default: 0
    - clicks: Number of scroll clicks (use with direction, alternative to x/y), default: None
    - direction: Scroll direction when using clicks ('up' or 'down'), default: None
    - delay: Optional delay in seconds after the action.
    """

    action: Literal[CuaAction.SCROLL] = CuaAction.SCROLL
    x: int = Field(default=0, description="Horizontal scroll amount (positive = right, negative = left)")
    y: int = Field(default=0, description="Vertical scroll amount (positive = up, negative = down)")
    clicks: int | None = Field(default=None, description="Number of scroll clicks (alternative to x/y)")
    direction: CuaScrollDirection | None = Field(default=None, description="Scroll direction when using clicks")
    delay: float | None = Field(default=None, description="Optional delay in seconds after the action")

    @model_validator(mode="after")
    def validate_clicks_and_direction(self):
        """Ensure clicks and direction are used together."""
        clicks = self.clicks
        direction = self.direction

        # Both must be provided
        if (clicks is not None) != (direction is not None):
            if clicks is not None:
                raise ValueError("clicks can only be used when direction is specified")
            else:
                raise ValueError("direction can only be used when clicks is specified")

        return self


# Screen Actions


class CuaTakeScreenshotSchema(BaseModel):
    """take_screenshot: Capture a screenshot and return it as base64 PNG.

    - boxes: Optional list of [x, y, width, height] tuples defining boxes to draw in screen coordinates
    - box_color: Color of the boxes in hex format (default: "#FF0000" red)
    - box_thickness: Thickness of the box borders in pixels (default: 2)
    - scale_factor: Factor to scale the final image by (default: 1.0). Use > 1.0 to enlarge, < 1.0 to shrink.
    """

    action: Literal[CuaAction.TAKE_SCREENSHOT] = CuaAction.TAKE_SCREENSHOT
    boxes: list[list[int]] | None = Field(
        default=None, description="Optional list of [x, y, width, height] tuples defining boxes to draw"
    )
    box_color: str = Field(default="#FF0000", description="Color of the boxes in hex format")
    box_thickness: int = Field(default=2, description="Thickness of the box borders in pixels")
    scale_factor: float = Field(default=1.0, description="Factor to scale the final image by")

    @field_validator("boxes")
    @classmethod
    def validate_boxes_coordinates(cls, v: list[list[int]] | None) -> list[list[int]] | None:
        """Validate that each box has exactly 4 elements [x, y, width, height]."""
        if v is None:
            return v
        for i, box in enumerate(v):
            if len(box) != 4:
                raise ValueError(
                    f"boxes[{i}] must have exactly 4 elements [x, y, width, height], got {len(box)} elements: {box}"
                )
        return v


class CuaGetScreenSizeSchema(BaseModel):
    """get_screen_size: Get the screen dimensions."""

    action: Literal[CuaAction.GET_SCREEN_SIZE] = CuaAction.GET_SCREEN_SIZE


# Window Management


class CuaLaunchSchema(BaseModel):
    """launch: Launch an application with optional arguments.

    - app: The application executable or bundle identifier
    - args: Optional list of arguments to pass to the application
    """

    action: Literal[CuaAction.LAUNCH] = CuaAction.LAUNCH
    app: str = Field(..., description="The application executable or bundle identifier")
    args: list[str] | None = Field(default=None, description="Optional list of arguments to pass to the application")


class CuaOpenSchema(BaseModel):
    """open: Open a target using the system's default handler.
    This can be a file path, folder path, or URL.

    - target: URL to open
    """

    action: Literal[CuaAction.OPEN] = CuaAction.OPEN
    target: str = Field(..., description="The file path, folder path, or URL to open.")


class CuaGetApplicationWindowsSchema(BaseModel):
    """get_application_windows: Get windows for a specific application.

    - app: The application name, executable, or identifier to query.
    """

    action: Literal[CuaAction.GET_APPLICATION_WINDOWS] = CuaAction.GET_APPLICATION_WINDOWS
    app: str = Field(..., description="The application name, executable, or identifier to query")


class CuaActivateWindowSchema(BaseModel):
    """activate_window: Bring a window to the foreground and focus it.

    - window_id: ID of the window to activate
    """

    action: Literal[CuaAction.ACTIVATE_WINDOW] = CuaAction.ACTIVATE_WINDOW
    window_id: str | int = Field(..., description="ID of the window to activate")


class CuaGetCurrentWindowIdSchema(BaseModel):
    """get_current_window_id: Get the ID of the currently active window."""

    action: Literal[CuaAction.GET_CURRENT_WINDOW_ID] = CuaAction.GET_CURRENT_WINDOW_ID


class CuaMaximizeWindowSchema(BaseModel):
    """maximize_window: Maximize a window.

    - window_id: ID of the window to maximize
    """

    action: Literal[CuaAction.MAXIMIZE_WINDOW] = CuaAction.MAXIMIZE_WINDOW
    window_id: str | int = Field(..., description="ID of the window to maximize")


class CuaMinimizeWindowSchema(BaseModel):
    """minimize_window: Minimize a window.

    - window_id: ID of the window to minimize
    """

    action: Literal[CuaAction.MINIMIZE_WINDOW] = CuaAction.MINIMIZE_WINDOW
    window_id: str | int = Field(..., description="ID of the window to minimize")


class CuaCloseWindowSchema(BaseModel):
    """close_window: Close a window.

    - window_id: ID of the window to close
    """

    action: Literal[CuaAction.CLOSE_WINDOW] = CuaAction.CLOSE_WINDOW
    window_id: str | int = Field(..., description="ID of the window to close")


# Clipboard Actions


class CuaClipboardSchema(BaseModel):
    """clipboard: Get or set clipboard content.

    - operation: 'get' to read clipboard, 'set' to write to clipboard
    - text: Text to set (required only for 'set' operation)
    """

    action: Literal[CuaAction.CLIPBOARD] = CuaAction.CLIPBOARD
    operation: CuaClipboardOperation = Field(..., description="'get' to read clipboard, 'set' to write to clipboard")
    text: str | None = Field(default=None, description="Text to set (required only for 'set' operation)")

    @field_validator("text")
    @classmethod
    def validate_text_for_set(cls, v, info):
        """Ensure text is provided for set operation."""
        if info.data.get("operation") == "set" and not v:
            raise ValueError("text is required when operation is 'set'")
        return v


# Accessibility


class CuaToScreenCoordinatesSchema(BaseModel):
    """to_screen_coordinates: Convert screenshot coordinates to screen coordinates.

    - x: X coordinate in screenshot space
    - y: Y coordinate in screenshot space
    """

    action: Literal[CuaAction.TO_SCREEN_COORDINATES] = CuaAction.TO_SCREEN_COORDINATES
    x: float = Field(..., description="X coordinate in screenshot space")
    y: float = Field(..., description="Y coordinate in screenshot space")


class CuaToScreenshotCoordinatesSchema(BaseModel):
    """to_screenshot_coordinates: Convert screen coordinates to screenshot coordinates.

    - x: X coordinate in screen space
    - y: Y coordinate in screen space
    """

    action: Literal[CuaAction.TO_SCREENSHOT_COORDINATES] = CuaAction.TO_SCREENSHOT_COORDINATES
    x: float = Field(..., description="X coordinate in screen space")
    y: float = Field(..., description="Y coordinate in screen space")


# Other Actions


class CuaWaitSchema(BaseModel):
    """wait: Pause execution for a given number of seconds."""

    action: Literal[CuaAction.WAIT] = CuaAction.WAIT
    duration: float = Field(..., description="Duration in seconds to wait")


class CuaRunCommandSchema(BaseModel):
    """run_command: Run shell command and return structured result.

    - command: The shell command to execute
    """

    action: Literal[CuaAction.RUN_COMMAND] = CuaAction.RUN_COMMAND
    command: str = Field(..., description="The shell command to execute")
