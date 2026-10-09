"""Hook primitives for agents: points, decisions, run context, the ``Hook`` base class and the runner.

A hook implements the points it needs; every other point allows. Hooks are evaluated one by one: ``before_*`` and
``on_input`` in list order, ``after_*`` and ``on_output`` in reverse order. ``Modify`` feeds the next hook, ``Block``
and ``Skip`` end the evaluation, and an ``Ask`` (``before_tool``) is answered once every hook has run.

==============  ==========================================  ===============================================
Point           When                                        Value (what ``Modify`` replaces)
==============  ==========================================  ===============================================
on_input        once per run                                ``str``, the user's message text
before_model    every LLM call                              ``list[Message]``, what is sent
after_model     every LLM call that succeeded               ``dict``, ``{"content", "tool_calls"}``
before_tool     every tool call (sub-agents included)       the tool input, after ``tool_params`` merging
after_tool      every tool call, failures included          ``ToolResult``
on_output       once per run, on the parsed final answer    the answer
==============  ==========================================  ===============================================

Hooks are pydantic configs (dumped to and loaded from YAML) shared by parallel tool calls and Map items, so per-run
data goes in ``ctx.hook_state``, never on the hook.
"""

import contextvars
import dataclasses
import fnmatch
import importlib
import re
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict, Field, model_serializer

from dynamiq.nodes.agents.exceptions import (
    HookAnswerException,
    HookBlockedException,
    HookStopException,
    RecoverableAgentException,
    ToolBlockedException,
)
from dynamiq.runnables import RunnableConfig
from dynamiq.types.feedback import FeedbackMethod
from dynamiq.utils.logger import logger


class HookPoint(str, Enum):
    ON_INPUT = "on_input"
    BEFORE_MODEL = "before_model"
    AFTER_MODEL = "after_model"
    BEFORE_TOOL = "before_tool"
    AFTER_TOOL = "after_tool"
    ON_OUTPUT = "on_output"


TOOL_POINTS = (HookPoint.BEFORE_TOOL, HookPoint.AFTER_TOOL)
REVERSED_POINTS = (HookPoint.AFTER_MODEL, HookPoint.AFTER_TOOL, HookPoint.ON_OUTPUT)


class BlockAs(str, Enum):
    """What a ``Block`` does.

    Attributes:
        OBSERVATION: Tool points only: the reason goes back to the model as the observation and the run continues.
            Anywhere else it behaves like ``ANSWER``.
        ANSWER: End the run successfully with the reason as the answer (the output has ``blocked: true``).
        FAIL: End the run with a failure that is never retried.
    """

    OBSERVATION = "observation"
    ANSWER = "answer"
    FAIL = "fail"


class HookErrorPolicy(str, Enum):
    """What to do when a hook itself raises or times out.

    Attributes:
        BLOCK: Fail closed: an observation at tool points, a failed run elsewhere.
        STOP: Fail the run.
        SKIP: Fail open: log and continue as if the hook allowed.
    """

    BLOCK = "block"
    STOP = "stop"
    SKIP = "skip"


_BLOCK_EXCEPTIONS: dict[BlockAs, type[Exception]] = {
    BlockAs.OBSERVATION: ToolBlockedException,
    BlockAs.ANSWER: HookAnswerException,
    BlockAs.FAIL: HookStopException,
}


@dataclass(frozen=True)
class Allow:
    """Let the value through unchanged."""


@dataclass(frozen=True)
class Modify:
    """Replace the value."""

    value: Any


@dataclass(frozen=True)
class Block:
    """Refuse. ``reason`` is shown to the model or user, so it must not contain the offending payload."""

    reason: str
    as_: BlockAs = BlockAs.OBSERVATION


@dataclass(frozen=True)
class Skip:
    """``before_tool`` only: do not run the tool, use ``result`` as its output (cache, mock)."""

    result: Any


@dataclass(frozen=True)
class Ask:
    """``before_tool`` only: ask a human to approve the call before it runs (through the agent's approval flow).

    The human sees the input as it is when every hook has run, so a hook listed earlier that restores real values
    (``pii`` with ``restore_in_tools``) makes the approval show them. ``editable_params`` are the arguments the human
    may change. A refusal reaches the model as an observation. If several hooks ask, the first one is used.
    """

    prompt: str | None = None
    feedback_method: FeedbackMethod = FeedbackMethod.CONSOLE
    editable_params: tuple[str, ...] = ()


Decision = Allow | Modify | Block | Skip | Ask
ALLOW = Allow()


@dataclass
class ToolCall:
    name: str
    input: Any
    tool_id: str | None = None
    is_sub_agent: bool = False
    group: str | None = None  # the MCP server the tool belongs to


@dataclass
class ToolResult:
    """What a tool returned: ``content``, its other output keys (e.g. ``raw_response``), or ``error`` if it failed."""

    content: Any = None
    output: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


@dataclass
class HookContext:
    """Per-run context of the hooks: one per ``Agent.execute`` (so one per Map item and per sub-agent run).

    ``state`` is shared by all hooks and all parallel tool calls: JSON-serializable values only (it is checkpointed),
    mutated under ``lock``. ``hook_state`` is the slice of the running hook. ``hooks`` are the ``(state key, hook)``
    pairs in effect for the run: those inherited from the parent agent, then the agent's own.
    """

    agent_name: str = ""
    agent_id: str = ""
    user_id: str | None = None
    session_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    state: dict[str, Any] = field(default_factory=dict)
    lock: Any = field(default_factory=threading.RLock)
    loop: int | None = None
    tool_call_id: str | None = None
    run_id: str | None = None
    hook_key: str = ""
    config: RunnableConfig | None = None
    hooks: list[tuple[str, "Hook"]] = field(default_factory=list)
    memory_transforms: list = field(default_factory=list)
    approval_lock: Any = field(default_factory=threading.Lock)
    pending_approvals: dict[str, tuple["HookContext", ToolCall]] = field(default_factory=dict)
    approver: Any = None  # the agent whose input stream carries this run's approval requests
    trusted: dict[str, Any] = field(default_factory=dict)

    def for_call(self, **changes) -> "HookContext":
        """A view for one call that shares ``state`` and ``lock`` with the run context."""
        return dataclasses.replace(self, **changes)

    @property
    def hook_state(self) -> dict[str, Any]:
        with self.lock:
            return self.state.setdefault("hooks", {}).setdefault(self.hook_key, {})

    def lookup(self, path: str) -> Any:
        """Read ``user_id``, ``session_id``, ``agent_name``, ``agent_id`` or ``metadata.<key>[.<key>...]`` of the
        run input (set by the client)."""
        head, *keys = path.split(".")
        if head == "metadata":
            return self._dig(self.metadata, keys)
        if not keys and head in ("user_id", "session_id", "agent_name", "agent_id"):
            return getattr(self, head)
        return None

    def lookup_trusted(self, path: str) -> Any:
        """Read a dotted path of the context the server passed in ``RunnableConfig.trusted_context`` (the client
        cannot set it). Access rules read this, never the run input."""
        head, *keys = path.split(".")
        return self._dig(self.trusted.get(head), keys)

    @staticmethod
    def _dig(value: Any, keys: list[str]) -> Any:
        for key in keys:
            value = value.get(key) if isinstance(value, dict) else None
        return value


current_hook_ctx: contextvars.ContextVar[HookContext | None] = contextvars.ContextVar("agent_hook_ctx", default=None)

HOOK_REGISTRY: dict[str, type["Hook"]] = {}


def sanitize_tool_name(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]", "", name.replace(" ", "-"))


def sanitize_tool_name_pattern(pattern: str) -> str:
    """Like ``sanitize_tool_name``, keeping the ``*`` and ``?`` wildcards."""
    return re.sub(r"[^a-zA-Z0-9_\-*?]", "", pattern.replace(" ", "-"))


def register_hook(name: str):
    """Class decorator: make ``type: <name>`` resolve to the decorated hook."""

    def decorator(cls):
        if HOOK_REGISTRY.get(name, cls) is not cls:
            raise ValueError(f"Hook type {name!r} is already registered")
        cls.hook_type = name
        HOOK_REGISTRY[name] = cls
        return cls

    return decorator


class Hook(BaseModel):
    """Base class of every hook. Override the points you need; reference a Python subclass from YAML with
    ``type: my_pkg.my_module.MyHook``. Unknown fields are rejected, so a typo cannot yield a hook that does nothing.

    Attributes:
        name: Shown in traces, stream events and block outcomes, and keys the hook's saved state (so it is unique
            per agent; unnamed hooks of one type are told apart by their order). Defaults to the hook type.
        tools: Tool points only: the tools the hook applies to (``*`` or empty = all). An entry is a tool name, an MCP
            server name (all its tools) or a ``*`` pattern of either, e.g. ``github`` or ``delete_*``.
        inherit: Also apply in the sub-agents this agent calls (and theirs), sharing the run state, so that for
            example a PII mapping stays consistent across agents.
        on_error: What to do if the hook itself raises or times out.
        timeout_seconds: Per-call budget; a hook that overruns counts as failed. The worker thread cannot be killed.
    """

    model_config = ConfigDict(extra="forbid")

    name: str | None = None
    tools: list[str] = Field(default_factory=list)
    inherit: bool = False
    on_error: HookErrorPolicy = HookErrorPolicy.BLOCK
    timeout_seconds: float | None = Field(default=None, gt=0)

    hook_type: ClassVar[str | None] = None

    @model_serializer(mode="wrap")
    def _serialize_with_type(self, handler):
        data = handler(self)
        cls = type(self)
        if cls.__dict__.get("hook_type"):
            return {"type": cls.hook_type, **data}
        if cls is Hook:
            return data
        return {"type": f"{cls.__module__}.{cls.__qualname__}", **data}

    def on_input(self, ctx: HookContext, text: str) -> Decision:
        return ALLOW

    def before_model(self, ctx: HookContext, messages: list) -> Decision:
        return ALLOW

    def after_model(self, ctx: HookContext, output: dict[str, Any]) -> Decision:
        return ALLOW

    def before_tool(self, ctx: HookContext, call: ToolCall) -> Decision:
        return ALLOW

    def after_tool(self, ctx: HookContext, call: ToolCall, result: ToolResult) -> Decision:
        return ALLOW

    def on_output(self, ctx: HookContext, answer: Any) -> Decision:
        return ALLOW

    def points(self) -> set[HookPoint]:
        """The points this hook acts on: by default those the subclass overrides. Built-ins narrow this to what
        their config enables, which also tells the agent whether it has to buffer the streamed answer."""
        return {point for point in HookPoint if getattr(type(self), point.value) is not getattr(Hook, point.value)}

    @property
    def type_name(self) -> str:
        return type(self).hook_type or type(self).__name__

    @property
    def display_name(self) -> str:
        return self.name or self.type_name

    def live_lookback(self) -> int | None:
        """How many characters this hook holds back to mask the answer while it streams, or ``None`` if it cannot
        (then the agent buffers the answer and streams it once, after the hooks)."""
        return None

    def live_text(self, ctx: HookContext, text: str) -> str:
        """The answer text as it should be streamed so far; only called when ``live_lookback`` is not ``None``."""
        raise NotImplementedError

    def matches_tool(self, tool_name: str, group: str | None = None) -> bool:
        """``tools`` entries are tool names, MCP server names or ``*``-globs of either (``github``, ``delete_*``)."""
        if not self.tools or "*" in self.tools:
            return True
        names = [sanitize_tool_name(tool_name), *([sanitize_tool_name(group)] if group else [])]
        patterns = [sanitize_tool_name_pattern(pattern) for pattern in self.tools]
        return any(fnmatch.fnmatchcase(name, pattern) for pattern in patterns for name in names)

    def hides_tool(self, tool_name: str, group: str | None = None) -> bool:
        """Whether the model should not be offered the tool at all (it could only be refused)."""
        return False

    def order(self, point: HookPoint) -> int:
        """Where the hook runs at ``point`` relative to the others: lower first, ties keep the list order (reversed
        at the ``after`` points). Detectors return 1 so that masking hooks always run before them."""
        return 0

    def refund(self, ctx: HookContext, call: ToolCall) -> None:
        """A call this hook let through did not run after all (another hook refused it, or the human declined):
        give back what ``before_tool`` took, e.g. a call count."""


def resolve_hook(item: Any) -> Hook:
    """Build a hook from an instance or a dict (``type: pii`` or ``type: pkg.mod.Class``, plus its fields)."""
    if isinstance(item, Hook):
        return item
    if not isinstance(item, dict):
        raise ValueError(f"A hook must be a mapping with a `type`, got {type(item).__name__}")
    hook_type = item.get("type")
    if not hook_type:
        raise ValueError(f"Hook is missing `type` (built-in: {sorted(HOOK_REGISTRY)}, or a dotted class path): {item}")
    hook_type = str(hook_type)
    fields = {key: value for key, value in item.items() if key != "type"}
    if True in fields and "on" not in fields:
        fields["on"] = fields.pop(True)

    if hook_type in HOOK_REGISTRY:
        hook_class = HOOK_REGISTRY[hook_type]
    elif "." in hook_type:
        module_name, _, class_name = hook_type.rpartition(".")
        try:
            hook_class = getattr(importlib.import_module(module_name), class_name)
        except (ImportError, AttributeError, ValueError) as e:
            raise ValueError(f"Hook type {hook_type!r} could not be imported: {e}") from e
        if not (isinstance(hook_class, type) and issubclass(hook_class, Hook)):
            raise ValueError(f"Hook type {hook_type!r} is not a Hook subclass")
    else:
        raise ValueError(f"Unknown hook type {hook_type!r}. Built-in types: {sorted(HOOK_REGISTRY)}")
    return hook_class.model_validate(fields)


def size_of(value: Any) -> int:
    """Size in characters of a value, for trace events (the content itself is never traced)."""
    if isinstance(value, str):
        return len(value)
    if isinstance(value, ToolResult):
        return size_of(value.content) + size_of(value.output) + size_of(value.error)
    if isinstance(value, (list, tuple)):
        return sum(size_of(item) for item in value)
    if isinstance(value, dict):
        return sum(size_of(item) for item in value.values())
    content = getattr(value, "content", None)
    if content is not None and not callable(content):
        return size_of(content)
    return 0 if value is None else len(repr(value))


class LiveAnswerFilter:
    """Masks the answer while it streams. Chunks are held back by ``lookback`` characters, so a match that is still
    incomplete (``ann@x.c``) is not sent in the clear; ``finish`` sends the rest. A match longer than the lookback
    can still leak."""

    def __init__(self, steps: list[Callable[[str], str]], lookback: int):
        self.steps = steps
        self.lookback = lookback
        self.raw = ""
        self.sent = ""

    def _masked(self) -> str:
        text = self.raw
        for step in self.steps:
            text = step(text)
        return text

    def _resync(self, masked: str) -> None:
        """A match longer than the lookback: part of it went out raw before it was recognised. Count the masked
        replacement as sent too, so the text after the match still streams (the raw part is the documented leak)."""
        if masked.startswith(self.sent):
            return
        limit = min(len(self.raw), len(masked))
        prefix = 0
        while prefix < limit and masked[prefix] == self.sent[prefix : prefix + 1]:
            prefix += 1
        suffix = 0
        while suffix < limit - prefix and masked[-1 - suffix] == self.raw[-1 - suffix]:
            suffix += 1
        self.sent = masked[: len(masked) - suffix]

    def feed(self, chunk: str) -> str:
        self.raw += chunk
        masked = self._masked()
        self._resync(masked)
        safe = masked[: max(len(masked) - self.lookback, 0)]
        if not safe.startswith(self.sent):
            return ""
        out, self.sent = safe[len(self.sent) :], safe
        return out

    def finish(self) -> str:
        masked = self._masked()
        self._resync(masked)
        out = masked[len(self.sent) :]
        self.raw = self.sent = ""
        return out

    def mask_all(self, text: str) -> str:
        """The whole text masked at once (for an answer that is not streamed in chunks)."""
        self.raw = text
        masked = self._masked()
        self.raw = self.sent = ""
        return masked


@dataclass
class HookRun:
    """Outcome of one point: the (possibly modified) value, a ``Skip``, or an ``Ask`` (with the hook that asked)."""

    value: Any
    skip: Skip | None = None
    ask: Ask | None = None
    asked_by: Hook | None = None


class HookRunner:
    def __init__(self, hooks: list[Hook], keys: list[str] | None = None):
        """``keys`` name each hook's slice of the run state (default: its position and name)."""
        self.hooks = list(hooks)
        self.keys = keys or [f"{index}:{hook.display_name}" for index, hook in enumerate(self.hooks)]

    def has(self, *points: HookPoint) -> bool:
        return any(hook.points() & set(points) for hook in self.hooks)

    def live_answer_filter(self, ctx: HookContext) -> LiveAnswerFilter | None:
        """A filter that masks the answer while it streams, if every hook that touches the answer can do that."""
        if self.has(HookPoint.AFTER_MODEL):
            return None
        answer_hooks = [(key, hook) for key, hook in zip(self.keys, self.hooks) if HookPoint.ON_OUTPUT in hook.points()]
        lookbacks = [hook.live_lookback() for _, hook in answer_hooks]
        if not answer_hooks or None in lookbacks:
            return None
        steps = [
            (lambda text, hook=hook, hook_ctx=ctx.for_call(hook_key=key): hook.live_text(hook_ctx, text))
            for key, hook in reversed(answer_hooks)
        ]
        return LiveAnswerFilter(steps, max(lookbacks))

    def run(
        self,
        point: HookPoint,
        ctx: HookContext,
        value: Any,
        *,
        call: ToolCall | None = None,
        events: list[dict] | None = None,
    ) -> HookRun:
        """Evaluate ``point`` over the hooks. A block is recorded in ``events``, then raised as
        ``ToolBlockedException`` (observation), ``HookAnswerException`` or ``HookStopException``.

        At ``before_tool`` ``value`` is ``call.input``.
        """
        indexed = list(enumerate(self.hooks))
        ask, asked_by = None, None
        if point in REVERSED_POINTS:
            indexed.reverse()
        indexed.sort(key=lambda item: item[1].order(point))
        passed: list[tuple[int, Hook]] = []

        for index, hook in indexed:
            if point not in hook.points():
                continue
            if point in TOOL_POINTS and call is not None and not hook.matches_tool(call.name, call.group):
                continue
            hook_ctx = ctx.for_call(hook_key=self.keys[index])
            try:
                decision = self._invoke(hook, point, hook_ctx, value, call)
            except (HookBlockedException, RecoverableAgentException):
                self._refund(passed, point, ctx, call)
                raise
            except Exception as e:
                decision = self._handle_error(hook, point, e, call, events)

            if isinstance(decision, Modify):
                _record(
                    events,
                    hook,
                    point,
                    "modify",
                    call,
                    changed=decision.value != value,
                    before_chars=size_of(value),
                    after_chars=size_of(decision.value),
                )
                value = decision.value
                if point == HookPoint.BEFORE_TOOL:
                    call = dataclasses.replace(call, input=value)
            elif isinstance(decision, Skip) and point == HookPoint.BEFORE_TOOL:
                _record(events, hook, point, "skip", call)
                self._refund(passed, point, ctx, call)
                return HookRun(value=value, skip=decision)
            elif isinstance(decision, Ask) and point == HookPoint.BEFORE_TOOL:
                ask, asked_by = ask or decision, asked_by or hook
            elif isinstance(decision, Block):
                kind = self._effective_block_as(point, decision)
                _record(events, hook, point, "block", call, outcome=kind.value)
                reason = decision.reason
                if kind == BlockAs.FAIL:
                    reason = f"Hook '{hook.display_name}' stopped the run at {point.value}: {reason}"
                self._refund(passed, point, ctx, call)
                raise _BLOCK_EXCEPTIONS[kind](reason, hook=hook.display_name, point=point.value)
            passed.append((index, hook))
        return HookRun(value=value, ask=ask, asked_by=asked_by)

    def _refund(
        self, passed: list[tuple[int, Hook]], point: HookPoint, ctx: HookContext, call: ToolCall | None
    ) -> None:
        if point == HookPoint.BEFORE_TOOL and call is not None:
            self.refund(ctx, call, passed)

    def refund(self, ctx: HookContext, call: ToolCall, passed: list[tuple[int, Hook]] | None = None) -> None:
        """Tell the hooks that let ``call`` through (all matching ones by default) that it did not run."""
        hooks = passed if passed is not None else list(enumerate(self.hooks))
        for index, hook in hooks:
            if hook.matches_tool(call.name, call.group):
                hook.refund(ctx.for_call(hook_key=self.keys[index]), call)

    @staticmethod
    def _effective_block_as(point: HookPoint, block: Block) -> BlockAs:
        if block.as_ == BlockAs.OBSERVATION and point not in TOOL_POINTS:
            return BlockAs.ANSWER
        return block.as_

    @staticmethod
    def _invoke(hook: Hook, point: HookPoint, ctx: HookContext, value: Any, call: ToolCall | None):
        method = getattr(hook, point.value)
        if point == HookPoint.AFTER_TOOL:
            args = (ctx, call, value)
        elif point == HookPoint.BEFORE_TOOL:
            args = (ctx, call)
        else:
            args = (ctx, value)

        if hook.timeout_seconds is None:
            return method(*args)

        executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="agent-hook")
        future = executor.submit(contextvars.copy_context().run, method, *args)
        try:
            return future.result(timeout=hook.timeout_seconds)
        except FutureTimeoutError as e:
            raise TimeoutError(f"hook exceeded timeout_seconds={hook.timeout_seconds}") from e
        finally:
            executor.shutdown(wait=False)

    @staticmethod
    def _handle_error(
        hook: Hook, point: HookPoint, error: Exception, call: ToolCall | None, events: list[dict] | None
    ) -> Decision:
        """Apply ``hook.on_error``. Only the exception type is logged at error level and traced: its message may
        quote the payload the hook was handling."""
        policy = hook.on_error
        logger.error(
            f"Hook {hook.display_name!r} failed at {point.value} ({type(error).__name__}); on_error={policy.value}"
        )
        logger.debug(f"Hook {hook.display_name!r} failure detail: {error}")
        _record(events, hook, point, "error", call, error_type=type(error).__name__, policy=policy.value)
        if policy == HookErrorPolicy.SKIP:
            return ALLOW
        reason = f"Hook '{hook.display_name}' failed at {point.value}; the request was not processed."
        if policy == HookErrorPolicy.BLOCK and point in TOOL_POINTS:
            return Block(reason, BlockAs.OBSERVATION)
        return Block(reason, BlockAs.FAIL)


def _record(
    events: list[dict] | None, hook: Hook, point: HookPoint, decision: str, call: ToolCall | None, **extra: Any
) -> None:
    """Append a structured trace event: hook, point, decision and sizes, never the payload."""
    if events is None:
        return
    event = {"hook": hook.display_name, "type": hook.type_name, "point": point.value, "decision": decision}
    if call is not None:
        event["tool"] = call.name
    events.append({**event, **extra})
