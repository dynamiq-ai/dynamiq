from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class E2BToolType(str, Enum):
    """Tool type for E2B desktop tool (only computer actions supported)."""

    COMPUTER = "computer"


class E2BAction(str, Enum):
    """Actions supported by E2B desktop tool."""

    MOVE_MOUSE = "move_mouse"
    CLICK_MOUSE = "click_mouse"
    WAIT = "wait"
    RUN_COMMAND = "run_command"
    RUN_BACKGROUND_COMMAND = "run_background_command"
    SEND_KEY = "send_key"
    TYPE_TEXT = "type_text"
    TAKE_SCREENSHOT = "take_screenshot"
    LAUNCH = "launch"


class E2BMoveMouseSchema(BaseModel):
    """move_mouse: Move the mouse to specific coordinates.

    - coordinates: [x, y] absolute screen coordinates (integers)
    - hold_keys: optional modifier keys to hold (e.g., ["ctrl", "shift"]) before moving
    - screenshot: optional flag to capture a screenshot after the action
    """

    action: Literal[E2BAction.MOVE_MOUSE] = E2BAction.MOVE_MOUSE
    coordinates: list[int] = Field(..., description="[x, y] coordinates to move to", min_length=2, max_length=2)
    hold_keys: list[str] | None = Field(default=None, description="Modifier keys to hold (e.g., ['ctrl', 'shift'])")
    screenshot: bool | None = Field(default=False, description="Whether to take a screenshot after the action")


class E2BClickMouseSchema(BaseModel):
    """click_mouse: Click at the current or specified coordinates.

    - coordinates: optional [x, y] to move before clicking; omit to click current position
    - button: left/right (defaults to left)
    - click_type: down/up/click (defaults to click)
    - num_clicks: number of clicks (defaults to 1)
    - hold_keys: optional modifier keys to hold
    - screenshot: optional flag to capture a screenshot after the action
    """

    action: Literal[E2BAction.CLICK_MOUSE] = E2BAction.CLICK_MOUSE
    coordinates: list[int] | None = Field(
        default=None,
        description="[x, y] coordinates to click (optional, uses current position if not provided)",
        min_length=2,
        max_length=2,
    )
    button: Literal["left", "right"] = Field(default="left", description="Mouse button to click")
    click_type: Literal["down", "up", "click"] | None = Field(default="click", description="Type of click action")
    num_clicks: int | None = Field(default=1, description="Number of clicks", ge=1)
    hold_keys: list[str] | None = Field(default=None, description="Modifier keys to hold")
    screenshot: bool | None = Field(default=False, description="Whether to take a screenshot after the action")


class E2BWaitSchema(BaseModel):
    """wait: Pause execution for a given number of seconds."""

    action: Literal[E2BAction.WAIT] = E2BAction.WAIT
    duration: float = Field(..., description="Duration in seconds to wait")


class E2BRunCommandSchema(BaseModel):
    """run_command: Run a shell command and return stdout/stderr.

    - command: full shell command to run synchronously
    - timeout: optional timeout in seconds
    """

    action: Literal[E2BAction.RUN_COMMAND] = E2BAction.RUN_COMMAND
    command: str = Field(..., description="Shell command to run synchronously")
    timeout: int | None = Field(default=10, description="Timeout in seconds")


class E2BRunBackgroundCommandSchema(BaseModel):
    """run_background_command: Start a shell command asynchronously (no output)."""

    action: Literal[E2BAction.RUN_BACKGROUND_COMMAND] = E2BAction.RUN_BACKGROUND_COMMAND
    command: str = Field(..., description="Shell command to run asynchronously")


class E2BSendKeySchema(BaseModel):
    """send_key: Send a key or combination (e.g., "Return", "Ctl-C")."""

    action: Literal[E2BAction.SEND_KEY] = E2BAction.SEND_KEY
    name: str = Field(..., description="Key or combination (e.g. 'Return', 'Ctl-C')")


class E2BTypeTextSchema(BaseModel):
    """type_text: Type text into the current focused input.

    - text: the text to type
    - chunk_size: optional chunk size for grouped typing
    - delay_in_ms: optional delay between chunks in milliseconds
    """

    action: Literal[E2BAction.TYPE_TEXT] = E2BAction.TYPE_TEXT
    text: str = Field(..., description="Text to type")
    chunk_size: int | None = Field(default=None, description="Optional chunk size for typing")
    delay_in_ms: int | None = Field(default=None, description="Optional delay between chunks in milliseconds")


class E2BTakeScreenshotSchema(BaseModel):
    """take_screenshot: Capture a screenshot and return it as base64 PNG."""

    action: Literal[E2BAction.TAKE_SCREENSHOT] = E2BAction.TAKE_SCREENSHOT


class E2BLaunchSchema(BaseModel):
    """launch: Start an application using a shell command in the background.

    - application: command name to execute (e.g., 'google-chrome', 'firefox', 'code')
    - args: optional list of additional arguments (e.g., ['--incognito'])
    """

    action: Literal[E2BAction.LAUNCH] = E2BAction.LAUNCH
    application: str = Field(..., description="Application command to launch (e.g., 'google-chrome')")
    args: list[str] | None = Field(default=None, description="Optional list of arguments")
