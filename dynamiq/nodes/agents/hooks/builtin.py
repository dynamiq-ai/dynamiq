"""Built-in hooks, registered under short names for YAML and UI use.

transform         JSONPath rewrite of tool input (merged into the model's arguments) and of tool output.
tool_policy       deny a tool, allow it only for matching users, or ask a human to approve each call.
call_limit        at most N calls per run.
regex             block or mask regex matches, with presets.
pii               detect, mask and restore PII with numbered placeholders.
prompt_injection  prompt-injection check of the user input and of tool results through a detector connection.
"""

import copy
import dataclasses
import importlib
import json
import re
import threading
from collections import OrderedDict
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from typing import Annotated, Any, Literal
from uuid import uuid4

import regex
from pydantic import (
    Field,
    PrivateAttr,
    ValidationError,
    WithJsonSchema,
    field_serializer,
    field_validator,
    model_validator,
)

from dynamiq.nodes.agents.hooks.core import (
    ALLOW,
    Ask,
    Block,
    BlockAs,
    Decision,
    Hook,
    HookContext,
    HookPoint,
    Modify,
    ToolCall,
    ToolResult,
    register_hook,
    sanitize_tool_name,
)
from dynamiq.nodes.agents.utils import decode_text_payload
from dynamiq.nodes.detectors import LlamaGuardDetector, PIIDetector, PromptInjectionDetector
from dynamiq.nodes.node import InputTransformer, Node, OutputTransformer
from dynamiq.nodes.types import NodeGroup
from dynamiq.prompts import Message, MessageRole, VisionMessage, VisionMessageTextContent
from dynamiq.runnables import RunnableConfig, RunnableStatus
from dynamiq.types.feedback import FeedbackMethod
from dynamiq.utils.jsonpath import _is_rooted, is_jsonpath
from dynamiq.utils.jsonpath import mapper as jsonpath_mapper
from dynamiq.utils.logger import logger

REGEX_TIMEOUT_SECONDS = 1.0

RunPoint = Literal["input", "tool_input", "tool_result", "output"]
_RUN_POINTS = {
    "input": HookPoint.ON_INPUT,
    "tool_input": HookPoint.BEFORE_TOOL,
    "tool_result": HookPoint.AFTER_TOOL,
    "output": HookPoint.ON_OUTPUT,
}


def iter_strings(value: Any) -> Iterator[str]:
    """Every string leaf of a payload: dicts, lists and tuples are walked, and so is a ``ToolResult``.

    Bytes that hold text (an HTTP tool returns most JSON, HTML and XML bodies this way) count as a string.
    """
    if isinstance(value, str):
        yield value
    elif isinstance(value, (bytes, bytearray)):
        if (text := decode_text_payload(value)) is not None:
            yield text
    elif isinstance(value, ToolResult):
        yield from iter_strings([value.content, value.output, value.error])
    elif isinstance(value, dict):
        yield from iter_strings(list(value.values()))
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from iter_strings(item)


def map_strings(value: Any, fn) -> Any:
    """Copy of a payload with ``fn`` applied to every string leaf (text bytes too: changed ones become ``str``)."""
    if isinstance(value, str):
        return fn(value)
    if isinstance(value, (bytes, bytearray)):
        text = decode_text_payload(value)
        if text is None:
            return value
        mapped = fn(text)
        return value if mapped == text else mapped
    if isinstance(value, ToolResult):
        return ToolResult(map_strings(value.content, fn), map_strings(value.output, fn), map_strings(value.error, fn))
    if isinstance(value, dict):
        return {key: map_strings(item, fn) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(map_strings(item, fn) for item in value)
    return value


def map_message_text(message: Message | VisionMessage, fn) -> Message | VisionMessage:
    """Copy of a message with ``fn`` applied to its text (images and files are untouched)."""
    if isinstance(message, Message):
        return message.model_copy(update={"content": fn(message.content or "")})
    parts = [
        part.model_copy(update={"text": fn(part.text)}) if isinstance(part, VisionMessageTextContent) else part
        for part in message.content
    ]
    return message.model_copy(update={"content": parts})


def compile_regex(pattern: str, label: str = "regex") -> "regex.Pattern":
    try:
        return regex.compile(pattern)
    except regex.error as e:
        raise ValueError(f"invalid {label} {pattern!r}: {e}") from e


@register_hook("transform")
class TransformHook(Hook):
    """JSONPath rewrite of tool input and output, the declarative way to select fields.

    ``input_transformer.selector`` is merged into the model's arguments, like the agent's per-tool
    ``input_transformer``: listed keys are set and every other argument is kept. A path that matches nothing is
    skipped with a warning, never written as ``None``. A selector value that is not a JSONPath is a constant.

    ``from_context`` sets arguments from the run context (``{argument: user_id | session_id | metadata.<key>}``), so
    that a value the model must not choose (a tenant, a user id) comes from the caller. It overrides what the model
    sent; a missing context value skips the argument with a warning.

    ``output_transformer`` is applied to ``{"content": <result>}``; a JSON text result is parsed first so that paths
    can select from it.
    """

    input_transformer: InputTransformer = Field(default_factory=InputTransformer)
    output_transformer: OutputTransformer = Field(default_factory=OutputTransformer)
    from_context: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate(self):
        if self.input_transformer.path:
            raise ValueError("transform.input_transformer.path is not supported; use `selector`")
        for label, transformer in (("input", self.input_transformer), ("output", self.output_transformer)):
            if transformer.path and not is_jsonpath(transformer.path):
                raise ValueError(f"transform.{label}_transformer.path {transformer.path!r} is not a valid JSONPath")
            for key, expression in (transformer.selector or {}).items():
                if _is_rooted(expression) and not is_jsonpath(expression):
                    raise ValueError(
                        f"transform.{label}_transformer.selector[{key!r}] {expression!r} is not a valid JSONPath"
                    )
        if not (
            self.input_transformer.selector
            or self.from_context
            or self.output_transformer.path
            or self.output_transformer.selector
        ):
            raise ValueError("transform needs an input_transformer.selector, `from_context` or an output_transformer")
        return self

    def points(self) -> set[HookPoint]:
        points = set()
        if self.input_transformer.selector or self.from_context:
            points.add(HookPoint.BEFORE_TOOL)
        if self.output_transformer.path or self.output_transformer.selector:
            points.add(HookPoint.AFTER_TOOL)
        return points

    def before_tool(self, ctx: HookContext, call: ToolCall) -> Decision:
        if not isinstance(call.input, dict):
            return ALLOW
        merged = dict(call.input)
        for key, expression in (self.input_transformer.selector or {}).items():
            if not _is_rooted(expression):
                merged[key] = expression
            elif (value := jsonpath_mapper(call.input, {key: expression}).get(key)) is not None:
                merged[key] = value
            else:
                logger.warning(f"transform hook for '{call.name}': {expression!r} matched nothing; {key!r} skipped")
        for key, path in self.from_context.items():
            if (value := ctx.lookup(path)) is not None:
                merged[key] = value
            else:
                logger.warning(f"transform hook for '{call.name}': context {path!r} is not set; {key!r} skipped")
        return Modify(merged) if merged != call.input else ALLOW

    def after_tool(self, ctx: HookContext, call: ToolCall, result: ToolResult) -> Decision:
        if result.error is not None:
            return ALLOW
        data, parsed = result.content, False
        if isinstance(data, str):
            try:
                loaded = json.loads(data)
            except ValueError:
                loaded = None
            if isinstance(loaded, (dict, list)):
                data, parsed = loaded, True
        transformed = Node.transform({"content": data}, self.output_transformer)
        content = transformed["content"] if isinstance(transformed, dict) and "content" in transformed else transformed
        if content is None:
            logger.warning(f"transform hook for '{call.name}': output_transformer matched nothing; content is None")
        elif parsed and not isinstance(content, str):
            content = json.dumps(content, ensure_ascii=False)
        return Modify(ToolResult(content=content, output=result.output))


class GuardHook(Hook):
    """A hook that can refuse. ``on_violation`` is ``observation`` (the model is told; at tool points only, elsewhere
    it ends the run with the message), ``answer`` or ``fail``."""

    message: str = "Blocked by policy."
    on_violation: BlockAs = BlockAs.OBSERVATION

    def violation(self, **detail: Any) -> Block:
        return Block(self.message, self.on_violation, detail or None)


@register_hook("tool_policy")
class ToolPolicyHook(GuardHook):
    """Deny a tool, allow it only when the run context matches, and/or ask a human to approve each call.

    Attributes:
        deny: Always block the matching tools.
        allow_if: ``{context path: expected value or list of values}``; every entry must match, otherwise the call is
            blocked. Rules read the trusted context the server passes in ``RunnableConfig.trusted_context`` (e.g.
            ``user_id``, ``metadata.role``), never the run input, which the client controls. Without a trusted
            value a rule does not match: ``allow_if`` blocks and ``approval_unless`` still asks.
        approval: Ask a human to approve each call before it runs, through the agent's approval flow. The human sees
            the arguments as they are when all hooks ran (list a ``pii`` hook with ``restore_in_tools`` first and the
            approval shows the real values). A refusal reaches the model as an observation.
        approval_unless: Trusted-context rules (same shape as ``allow_if``) that skip the approval, e.g. for admins.
        approval_message: What the human is asked (a template: ``{{input_data.<argument>}}`` is filled in). Defaults to
            a message naming the tool.
        feedback_method: ``console`` or ``stream`` (the agent's input stream).
        editable_params: Arguments the human may change when approving.
    """

    message: str = "This tool is not available."
    deny: bool = False
    allow_if: dict[str, Any] | None = None
    approval: bool = False
    approval_unless: dict[str, Any] | None = None
    approval_message: str | None = None
    feedback_method: FeedbackMethod = FeedbackMethod.CONSOLE
    editable_params: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _needs_a_rule(self):
        if not (self.deny or self.allow_if or self.approval):
            raise ValueError("tool_policy needs `deny`, `allow_if` or `approval`; otherwise it would allow everything")
        if self.approval_unless and not self.approval:
            raise ValueError("tool_policy `approval_unless` only makes sense with `approval: true`")
        return self

    def points(self) -> set[HookPoint]:
        return {HookPoint.BEFORE_TOOL}

    def hides_tool(self, tool_name: str, group: str | None = None) -> bool:
        return self.deny and self.on_violation == BlockAs.OBSERVATION and self.matches_tool(tool_name, group)

    @staticmethod
    def _matches(ctx: HookContext, rules: dict[str, Any]) -> bool:
        for path, expected in rules.items():
            value = ctx.lookup_trusted(path)
            if value is None or value not in (expected if isinstance(expected, list) else [expected]):
                return False
        return True

    def before_tool(self, ctx: HookContext, call: ToolCall) -> Decision:
        if self.deny or (self.allow_if and not self._matches(ctx, self.allow_if)):
            return self.violation()
        if self.approval and not (self.approval_unless and self._matches(ctx, self.approval_unless)):
            prompt = self.approval_message or f"Approve calling '{call.name}' with these arguments?"
            return Ask(prompt, self.feedback_method, tuple(self.editable_params))
        return ALLOW


@register_hook("call_limit")
class CallLimitHook(GuardHook):
    """At most ``max_per_run`` calls of the matching tools per run, sub-agent calls included.

    The counter lives in the run context, so it is exact with parallel tool calls and separate for every Map item.
    A call that does not run (another hook blocked it, the human declined it, or it paused for an approval and
    will be asked again on resume) gives its slot back.
    """

    message: str = "Call limit reached for this tool in this run. Do not call it again."
    max_per_run: int = Field(ge=1)

    def points(self) -> set[HookPoint]:
        return {HookPoint.BEFORE_TOOL}

    def before_tool(self, ctx: HookContext, call: ToolCall) -> Decision:
        with ctx.lock:
            state = ctx.hook_state
            if state.get("calls", 0) >= self.max_per_run:
                return self.violation()
            state["calls"] = state.get("calls", 0) + 1
        return ALLOW

    def refund(self, ctx: HookContext, call: ToolCall) -> None:
        with ctx.lock:
            state = ctx.hook_state
            state["calls"] = max(state.get("calls", 0) - 1, 0)


class TextGuardHook(GuardHook):
    """Shared by the hooks that scan text: the user input, the arguments the model sends to tools, tool results
    (failures included) and the final answer.

    Subclasses define ``_violates`` (``action: block``) and ``_mask`` (``action: mask``).

    Attributes:
        on: ``input`` (the user's message, once per run), ``tool_input`` (the arguments of each tool call, so that
            text the model was talked into sending cannot leave through a tool; not on by default). The tool never
            gets what a block or mask removed, but the model's own message that holds the call stays in the history,
            memory and checkpoints as the model wrote it,
            ``tool_result`` (each tool result), ``output`` (the answer).
    """

    on: list[RunPoint] = Field(default_factory=lambda: ["input", "tool_result", "output"], min_length=1)
    action: Literal["block", "mask"] = "block"
    live_stream: bool = Field(
        default=False,
        description=(
            "With `action: mask`, stream the answer while it is generated instead of buffering it until the hooks "
            "ran. The stream is held back by `stream_lookback` characters, so a match longer than that can leak."
        ),
    )
    stream_lookback: int = Field(default=64, ge=8)

    def points(self) -> set[HookPoint]:
        return {_RUN_POINTS[name] for name in self.on}

    def live_lookback(self) -> int | None:
        return self.stream_lookback if self.live_stream and self.action == "mask" else None

    def live_text(self, ctx: HookContext, text: str) -> str:
        return self._mask(ctx, text, "output")

    def _violates(self, text: str) -> bool:
        raise NotImplementedError

    def _mask(self, ctx: HookContext, text: str, origin: str) -> str:
        raise NotImplementedError

    def _screen(self, ctx: HookContext, value: Any, origin: str) -> Decision:
        if self.action == "block":
            return self.violation() if any(self._violates(text) for text in iter_strings(value)) else ALLOW
        masked = map_strings(value, lambda text: self._mask(ctx, text, origin))
        return Modify(masked) if masked != value else ALLOW

    def on_input(self, ctx: HookContext, text: str) -> Decision:
        return self._screen(ctx, text, "input")

    def before_tool(self, ctx: HookContext, call: ToolCall) -> Decision:
        return self._screen(ctx, call.input, "tool_input")

    def after_tool(self, ctx: HookContext, call: ToolCall, result: ToolResult) -> Decision:
        return self._screen(ctx, result, "tool_result")

    def on_output(self, ctx: HookContext, answer: Any) -> Decision:
        return self._screen(ctx, answer, "output")


REGEX_PRESETS: dict[str, list[str]] = {
    "injection_basic": [
        r"(?i)\b(ignore|forget|override|bypass)\s+(all\s+)?(of\s+)?(the\s+|your\s+|any\s+)?"
        r"(previous|prior|above|earlier|preceding|system)\s+(instructions?|prompts?|rules|directions?|guidelines)\b",
        r"(?i)\bdisregard\s+(all\s+)?(of\s+)?(the\s+|your\s+|any\s+)?"
        r"(system|previous|prior|above|earlier|preceding)\s+(prompts?|instructions?|directions?|rules|guidelines)\b",
        r"(?i)\b(reveal|print|show)\s+(me\s+)?(your\s+)?(system\s+prompt|hidden\s+instructions)\b",
        r"(?i)\byou\s+are\s+now\s+(in\s+)?(developer|dan|jailbreak)\s+mode\b",
    ],
    "secrets": [
        r"\bAKIA[0-9A-Z]{16}\b",
        r"\bsk-[A-Za-z0-9_\-]{20,}\b",
        r"-----BEGIN [A-Z ]*PRIVATE KEY-----",
        r"(?i)\b(api[_-]?key|secret|token|password)\s*[:=]\s*\S{8,}",
    ],
}


_ZERO_WIDTH = regex.compile("[\u200b-\u200f\u2060\ufeff\u00ad]")


def _strip_zero_width(text: str) -> tuple[str, Sequence[int]]:
    """Text without invisible characters, plus the index in ``text`` of each character that was kept."""
    if not _ZERO_WIDTH.search(text):
        return text, range(len(text))
    origins: list[int] = []
    pos = 0
    for match in _ZERO_WIDTH.finditer(text):
        origins.extend(range(pos, match.start()))
        pos = match.end()
    origins.extend(range(pos, len(text)))
    return _ZERO_WIDTH.sub("", text), origins


@register_hook("regex")
class RegexHook(TextGuardHook):
    """Block or mask text that matches a pattern.

    The ``injection_basic`` preset is a basic filter for the common English phrasings; it does not catch other
    languages or paraphrases. For real protection add a ``prompt_injection`` hook (a hosted detector).

    Matching runs on the ``regex`` engine with a timeout (``REGEX_TIMEOUT_SECONDS`` per scan): a catastrophic pattern
    raises instead of stalling the process, and the hook then fails per ``on_error``.

    Attributes:
        patterns: Regexes. ``presets``: named sets (``injection_basic``, ``secrets``). One of the two is required.
        replacement: What ``action: mask`` puts in place of a match.
    """

    message: str = "The request was blocked by a content policy."
    patterns: list[str] = Field(default_factory=list)
    presets: list[Literal["injection_basic", "secrets"]] = Field(default_factory=list)
    replacement: str = "[REDACTED]"

    _compiled: list["regex.Pattern"] = PrivateAttr(default_factory=list)

    @model_validator(mode="after")
    def _compile(self):
        patterns = [*self.patterns, *(pattern for name in self.presets for pattern in REGEX_PRESETS[name])]
        if not patterns:
            raise ValueError("regex needs at least one pattern or preset; otherwise it would match nothing")
        self._compiled = [compile_regex(pattern) for pattern in patterns]
        return self

    def _violates(self, text: str) -> bool:
        text = _ZERO_WIDTH.sub("", text)  # "ig\u200bnore" must not slip past a pattern
        return any(pattern.search(text, timeout=REGEX_TIMEOUT_SECONDS) for pattern in self._compiled)

    def _mask(self, ctx: HookContext, text: str, origin: str) -> str:
        # Match on the text without invisible characters (like _violates), but replace only the matched spans of the
        # original, so text that matches nothing comes back untouched.
        cleaned, origins = _strip_zero_width(text)
        spans = sorted(
            (match.start(), match.end())
            for pattern in self._compiled
            for match in pattern.finditer(cleaned, timeout=REGEX_TIMEOUT_SECONDS)
            if match.end() > match.start()
        )
        parts: list[str] = []
        pos = 0
        for start, end in spans:
            start, end = origins[start], origins[end - 1] + 1
            if start < pos:  # overlaps a span already replaced
                pos = max(pos, end)
                continue
            parts += [text[pos:start], self.replacement]
            pos = end
        parts.append(text[pos:])
        return "".join(parts)


PIIEntity = Literal["email", "phone", "credit_card", "ssn", "ip_address"]

_PII_PATTERNS: dict[str, str] = {
    "credit_card": (r"(?<![\d-])(?:\d{4}([ -])\d{4}\1\d{4}\1\d{1,7}|\d{4}([ -])\d{6}\2\d{5}|\d{13,19})(?![\d-]|[ ]\d)"),
    "ssn": r"(?<!\d)\d{3}-\d{2}-\d{4}(?!\d)",
    "email": r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9\-]+(?:\.[A-Za-z0-9\-]+)*\.[A-Za-z]{2,}",
    "ip_address": r"(?<![\d.])(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)(?![\d.])",
    "phone": r"(?<![\w.])\+?\d[\d\s().\-]{7,}\d(?![\w])",
}
_VALIDATED_ENTITIES = {"credit_card", "phone"}
_PLACEHOLDER = regex.compile(r"<[A-Z_]+_\d+(?:_\d+)?>")


def _luhn_ok(digits: str) -> bool:
    total = 0
    for position, char in enumerate(reversed(digits)):
        number = int(char) * (2 if position % 2 else 1)
        total += number - 9 if number > 9 else number
    return total % 10 == 0


def _is_valid_match(entity: str | None, text: str) -> bool:
    digits = re.sub(r"\D", "", text)
    if entity == "credit_card":
        return 13 <= len(digits) <= 19 and _luhn_ok(digits)
    if entity == "phone":
        formatted = text.lstrip().startswith(("+", "(", "0")) or bool(
            re.search(r"\d[-.]\d|\b\d{3} \d{3} \d{4}\b|\b(?:\d{2} ){4}\d{2}\b", text)
        )
        return formatted and 9 <= len(digits) <= 15 and not re.match(r"\d{4}-\d{2}-\d{2}", text.strip())
    return True


_SESSION_STORES: "OrderedDict[tuple, dict]" = OrderedDict()
_SESSION_STORES_LOCK = threading.RLock()
_MAX_SESSION_STORES = 1000


@contextmanager
def _locked_store(ctx: HookContext) -> Iterator[dict]:
    """The placeholder mapping. With a ``session_id`` or ``user_id`` (the ids memory is keyed by) it is kept per
    user and session (in process), so that the placeholders of earlier turns, which the history holds, keep their
    meaning and numbering continues; otherwise per run.
    The key has no ``agent_id``: an inherited hook runs in sub-agents under their own ids but shares one store."""
    if not (ctx.session_id or ctx.user_id):
        with ctx.lock:
            yield ctx.hook_state
        return
    key = (ctx.hook_key, ctx.user_id, ctx.session_id)
    with _SESSION_STORES_LOCK:
        if key not in _SESSION_STORES:
            _SESSION_STORES[key] = {"namespace": str(uuid4().int)}
        store = _SESSION_STORES[key]
        _SESSION_STORES.move_to_end(key)
        while len(_SESSION_STORES) > _MAX_SESSION_STORES:
            _SESSION_STORES.popitem(last=False)
        yield store


def _sub_validated(pattern, text: str, replace, entity: str) -> str:
    """Replace the matches that pass validation. A candidate that fails does not consume the text, so a real value
    that starts inside it is still found (overlapped search, leftmost valid match first)."""
    out, pos = [], 0
    for match in pattern.finditer(text, overlapped=True, timeout=REGEX_TIMEOUT_SECONDS):
        if match.start() < pos or not _is_valid_match(entity, match.group(0)):
            continue
        out.append(text[pos : match.start()])
        out.append(replace(match))
        pos = match.end()
    out.append(text[pos:])
    return "".join(out)


@register_hook("pii")
class PIIHook(TextGuardHook):
    """Detect PII and mask it with numbered placeholders (``<EMAIL_1>``) so that the agent keeps working.

    One mapping per run lives in the run context and is checkpointed with the agent. ``scope`` decides what is masked:

    * ``persisted`` (default): masking at the source. The user's message and each tool result are masked once, so the
      conversation history, memory, checkpoints and the summarizer only ever hold placeholders, and nothing is
      re-scanned on later LLM calls.
    * ``request``: masking of this request only. History, memory and checkpoints keep the raw values, and every LLM
      call is sent a masked copy of the history (so the whole history is scanned again on each call).

    Attributes:
        scope: ``persisted`` or ``request`` (see above); ``request`` needs ``action: mask``.
        entities: What to detect. ``custom_patterns``: ``{NAME: regex}`` for more entity types.
        restore_in_tools: Tools that receive the real values: placeholders in their arguments are replaced right
            before the call. Other tools get placeholders. ``*`` means all tools. A list restores every argument;
            a mapping restores only the named arguments (``{send-email: [to]}``), so injected text cannot route a
            value into another argument such as the body.
        restore_in_output: Put the real values back in the final answer, only those the user supplied (never values
            that came from a tool result).
    """

    message: str = "The request contains personal data and was blocked."
    on: list[RunPoint] = Field(default_factory=lambda: ["input", "tool_result"], min_length=1)
    action: Literal["block", "mask"] = "mask"
    scope: Literal["persisted", "request"] = "persisted"
    entities: list[PIIEntity] = Field(default_factory=lambda: list(_PII_PATTERNS))
    custom_patterns: dict[str, str] = Field(default_factory=dict)
    restore_in_tools: list[str] | dict[str, list[str]] = Field(default_factory=list)
    restore_in_output: bool = False

    _detectors: list[tuple[str, str | None, "regex.Pattern"]] = PrivateAttr(default_factory=list)

    @field_validator("custom_patterns")
    @classmethod
    def _check_custom_names(cls, value):
        for name in value:
            if not re.fullmatch(r"[A-Z_]+", name):
                raise ValueError(f"custom_patterns name {name!r} must be UPPER_SNAKE_CASE letters")
        return value

    @model_validator(mode="after")
    def _compile(self):
        if not self.entities and not self.custom_patterns:
            raise ValueError("pii needs at least one entity or custom pattern")
        if self.scope == "request" and self.action != "mask":
            raise ValueError("pii `scope: request` only applies to `action: mask`")
        built_in = [(name.upper(), name, _PII_PATTERNS[name]) for name in _PII_PATTERNS if name in self.entities]
        custom = [(name, None, pattern) for name, pattern in self.custom_patterns.items()]
        self._detectors = [
            (label, entity, compile_regex(pattern, f"pattern for {label}"))
            for label, entity, pattern in built_in + custom
        ]
        return self

    def points(self) -> set[HookPoint]:
        points = super().points()
        if self.scope == "request":
            points = (points - {HookPoint.ON_INPUT, HookPoint.AFTER_TOOL}) | {HookPoint.BEFORE_MODEL}
        if self.restore_in_tools:
            points.add(HookPoint.BEFORE_TOOL)
        if self.restore_in_output:
            points.add(HookPoint.ON_OUTPUT)
        return points

    def _placeholder(self, ctx: HookContext, label: str, value: str, origin: str) -> str:
        with _locked_store(ctx) as store:
            placeholders = store.setdefault("value_to_placeholder", {}).setdefault(label, {})
            if (placeholder := placeholders.get(value)) is None:
                counters = store.setdefault("counters", {})
                counters[label] = counters.get(label, 0) + 1
                namespace = store.get("namespace", "")
                suffix = f"{namespace}_{counters[label]}" if namespace else str(counters[label])
                placeholder = placeholders[value] = f"<{label}_{suffix}>"
                store.setdefault("placeholder_to_value", {})[placeholder] = value
            origins = store.setdefault("origin", {})
            if origin == "input" or placeholder not in origins:
                origins[placeholder] = origin
        return placeholder

    def _violates(self, text: str) -> bool:
        return any(
            _is_valid_match(entity, match.group(0))
            for _, entity, pattern in self._detectors
            for match in pattern.finditer(text, overlapped=True, timeout=REGEX_TIMEOUT_SECONDS)
        )

    def _mask(self, ctx: HookContext, text: str, origin: str) -> str:
        for label, entity, pattern in self._detectors:

            def replace(match, label=label, entity=entity):
                found = match.group(0)
                return self._placeholder(ctx, label, found, origin) if _is_valid_match(entity, found) else found

            if entity in _VALIDATED_ENTITIES:
                text = _sub_validated(pattern, text, replace, entity)
            else:
                text = pattern.sub(replace, text, timeout=REGEX_TIMEOUT_SECONDS)
        return text

    def _restore(self, ctx: HookContext, text: str, only_origin: str | None = None) -> str:
        with _locked_store(ctx) as store:
            values, origins = dict(store.get("placeholder_to_value", {})), dict(store.get("origin", {}))

        def put_back(match):
            placeholder = match.group(0)
            if placeholder not in values or (only_origin and origins.get(placeholder) != only_origin):
                return placeholder
            return values[placeholder]

        return _PLACEHOLDER.sub(put_back, text)

    def _unrestore(self, ctx: HookContext, text: str) -> str:
        """The reverse of ``_restore`` for input-origin values: real values go back to their placeholders."""
        with _locked_store(ctx) as store:
            values, origins = dict(store.get("placeholder_to_value", {})), dict(store.get("origin", {}))
        for placeholder, value in sorted(values.items(), key=lambda item: len(item[1]), reverse=True):
            if origins.get(placeholder) == "input" and value:
                text = text.replace(value, placeholder)
        return text

    def _restore_args(self, tool_name: str) -> list[str] | None:
        """The arguments of ``tool_name`` that get real values: ``None`` for none, an empty list for all of them."""
        if isinstance(self.restore_in_tools, dict):
            wanted = {sanitize_tool_name(name): args for name, args in self.restore_in_tools.items() if name != "*"}
            return wanted.get(sanitize_tool_name(tool_name), self.restore_in_tools.get("*"))
        names = {sanitize_tool_name(name) for name in self.restore_in_tools}
        return [] if "*" in self.restore_in_tools or sanitize_tool_name(tool_name) in names else None

    def live_text(self, ctx: HookContext, text: str) -> str:
        """The streamed answer is masked (and restored) on a copy of the run state, so that the partial texts of a
        stream leave no placeholder behind in the real mapping."""
        with _locked_store(ctx) as store:
            snapshot = {"hooks": {ctx.hook_key: copy.deepcopy(store)}}
        preview = dataclasses.replace(ctx, state=snapshot, lock=threading.RLock(), session_id=None, user_id=None)
        if "output" in self.on:
            text = self._mask(preview, text, "output")
        if self.restore_in_output:
            text = self._restore(preview, text, only_origin="input")
        return text

    def before_model(self, ctx: HookContext, messages: list) -> Decision:
        """``scope: request``: send a masked copy of the history; the history itself is left alone."""
        masked = []
        for message in messages:
            origin = self._origin_of(message)
            masked.append(
                map_message_text(message, lambda text, origin=origin: self._mask(ctx, text, origin))
                if origin in self.on
                else message
            )
        return Modify(masked) if masked != messages else ALLOW

    @staticmethod
    def _origin_of(message: Message | VisionMessage) -> str | None:
        """Where a history message came from: the user, a tool result, or the model (``None`` for the system)."""
        match message.role:
            case MessageRole.SYSTEM:
                return None
            case MessageRole.TOOL:
                return "tool_result"
            case MessageRole.ASSISTANT:
                return "output"
        text = message.content if isinstance(message, Message) else ""
        return "tool_result" if str(text).startswith("Observation") else "input"

    def before_tool(self, ctx: HookContext, call: ToolCall) -> Decision:
        """Screen the arguments (``tool_input``), then give the tools that are allowed to see them the real values
        back: a placeholder that the model made up of a value it was sent is restored for those tools only."""
        value = call.input
        if "tool_input" in self.on:
            decision = self._screen(ctx, value, "tool_input")
            if isinstance(decision, Block):
                return decision
            if isinstance(decision, Modify):
                value = decision.value
        args = self._restore_args(call.name)
        if args is not None:
            value = self._restored_input(ctx, value, args)
        return Modify(value) if value != call.input else ALLOW

    def _restored_input(self, ctx: HookContext, tool_input: Any, args: list[str]) -> Any:
        def put_back(value):
            return map_strings(value, lambda text: self._restore(ctx, text))

        if args and isinstance(tool_input, dict):
            return {key: put_back(value) if key in args else value for key, value in tool_input.items()}
        return put_back(tool_input)

    def on_output(self, ctx: HookContext, answer: Any) -> Decision:
        value = answer
        if "output" in self.on:
            decision = self._screen(ctx, answer, "output")
            if isinstance(decision, Block):
                return decision
            if isinstance(decision, Modify):
                value = decision.value
        if self.restore_in_output:
            ctx.memory_transforms.append(lambda final: map_strings(final, lambda text: self._unrestore(ctx, text)))
            value = map_strings(value, lambda text: self._restore(ctx, text, only_origin="input"))
        return Modify(value) if value != answer else ALLOW


_LEGACY_DETECTOR_KINDS = {"prompt_injection": PromptInjectionDetector, "llama_guard": LlamaGuardDetector}


def build_detector(value: Any) -> Any:
    """Build the detector node of a ``prompt_injection`` hook from its config.

    ``type: dynamiq.nodes.detectors.PromptInjectionDetector`` (a node type, like ``llm`` on an agent) is the form
    that is saved. A mapping without ``type`` is the older ``kind`` form (``prompt_injection`` by default or
    ``llama_guard``), still accepted. A dumped connection (a mapping with its dotted ``type``) is rebuilt."""
    if not isinstance(value, dict):
        return value
    fields = dict(value)
    if node_type := fields.pop("type", None):
        module_name, _, class_name = str(node_type).rpartition(".")
        try:
            node_class = getattr(importlib.import_module(module_name), class_name)
        except (ImportError, AttributeError, ValueError) as e:
            raise ValueError(f"detector type {node_type!r} could not be imported: {e}") from e
        if not (isinstance(node_class, type) and issubclass(node_class, Node)):
            raise ValueError(f"detector type {node_type!r} is not a node")
    else:
        kind = fields.pop("kind", "prompt_injection")
        if kind not in _LEGACY_DETECTOR_KINDS:
            raise ValueError(f"detector kind {kind!r} is not one of {sorted(_LEGACY_DETECTOR_KINDS)}")
        node_class = _LEGACY_DETECTOR_KINDS[kind]
    fields.pop("kind", None)
    connection = fields.get("connection")
    if isinstance(connection, dict) and connection.get("type"):
        conn_module, _, conn_name = str(connection["type"]).rpartition(".")
        try:
            conn_class = getattr(importlib.import_module(conn_module), conn_name)
        except (ImportError, AttributeError, ValueError) as e:
            raise ValueError(f"connection type {connection['type']!r} could not be imported: {e}") from e
        fields["connection"] = conn_class(**{key: item for key, item in connection.items() if key != "type"})
    if unknown := sorted(set(fields) - set(node_class.model_fields)):
        raise ValueError(f"detector {node_type or kind!r} has no field {unknown}")
    try:
        return node_class(**fields)
    except ValidationError as e:
        label = node_type or f"kind {kind}"
        raise ValueError(f"detector {label!r} is invalid: {' '.join(str(e).split())}") from e


_DETECTOR_JSON_SCHEMA = {
    "type": "object",
    "description": (
        "A detector node, configured like `llm` on an agent: `type` is the node's import path, `connection` the "
        "name of a workflow connection, `model` the model of the detector."
    ),
    "properties": {
        "type": {
            "type": "string",
            "enum": [
                f"{detector.__module__.rsplit('.', 1)[0]}.{detector.__name__}"
                for detector in (PromptInjectionDetector, LlamaGuardDetector, PIIDetector)
            ],
        },
        "kind": {
            "type": "string",
            "enum": list(_LEGACY_DETECTOR_KINDS),
            "deprecated": True,
            "description": "The older form: use `type` with the detector node's import path instead.",
        },
        "connection": {"type": "string"},
        "model": {"type": "string"},
    },
    "anyOf": [{"required": ["type"]}, {"required": ["kind"]}, {"required": ["connection"]}],
}


@register_hook("prompt_injection")
class PromptInjectionHook(GuardHook):
    """Check the user's input, the arguments of tool calls, tool results (failed calls included) and the final
    answer with a detector node.

    ``detector`` is configured like any node, the way ``llm`` is on an agent: its ``type`` is the node's import path
    and ``connection`` references a connection of the workflow (``connection: lakera-conn``). Any detector node
    plugs in; the hook only turns its output into a yes/no (``prompt_detected``, ``is_detected`` or ``is_safe``). The
    older ``kind: prompt_injection | llama_guard`` form is still accepted. A detector that errors fails per
    ``on_error`` (closed by default). Each message, tool call and tool result is checked once, as one text
    (several detector calls only when it is longer than ``max_chars``). The detector sees the text after the masking
    hooks (``pii``) ran, whatever the order of the list.
    """

    message: str = "Potential prompt injection detected; the content was withheld."
    detector: Annotated[Node, WithJsonSchema(_DETECTOR_JSON_SCHEMA)]
    on: list[RunPoint] = Field(
        default_factory=lambda: ["input", "tool_result"],
        min_length=1,
        description=(
            "`input`, `tool_input` (the arguments of each tool call), `tool_result` and `output` (the final answer); "
            "`input` and `tool_result` by default."
        ),
    )
    max_chars: int = Field(default=20000, gt=0, description="The longest text sent to the detector in one call.")

    @field_validator("detector", mode="before")
    @classmethod
    def _build_detector(cls, value):
        return build_detector(value)

    @field_validator("detector")
    @classmethod
    def _detector_is_a_detector_node(cls, detector: Node):
        """A node of another group (an LLM, a tool) would only fail at run time, and so would every request."""
        if detector.group != NodeGroup.DETECTORS:
            raise ValueError(
                f"detector must be a detector node (group 'detectors'), got {type(detector).__name__} "
                f"(group {detector.group.value!r}), e.g. dynamiq.nodes.detectors.PromptInjectionDetector"
            )
        if "message" not in detector.input_schema.model_fields:
            raise ValueError(
                f"detector {type(detector).__name__} must take a `message` input: the hook sends it the text to check"
            )
        return detector

    @field_serializer("detector")
    def _dump_detector(self, detector: Node):
        return detector.to_dict(include_secure_params=True)

    def points(self) -> set[HookPoint]:
        return {_RUN_POINTS[name] for name in self.on}

    def order(self, point: HookPoint) -> int:
        return 1

    def detect(self, ctx: HookContext, text: str) -> bool:
        """Whether the detector flags ``text``. Override to plug another detector."""
        config = ctx.config or RunnableConfig(callbacks=[])
        options = {"parent_run_id": ctx.run_id} if ctx.run_id else {}
        result = self.detector.run_sync(input_data={"message": text}, config=config, **options)
        if result.status != RunnableStatus.SUCCESS:
            raise RuntimeError(f"detector failed: {result.error.message if result.error else result.status}")
        output = result.output or {}
        for key in ("prompt_detected", "is_detected"):
            if key in output:
                return bool(output[key])
        if "is_safe" in output:
            return output["is_safe"] is False
        raise RuntimeError(
            f"detector {self.detector.name!r} returned none of prompt_detected, is_detected, is_safe: {sorted(output)}"
        )

    def _batches(self, value: Any) -> Iterator[str]:
        """All the text of ``value`` joined, cut into pieces of at most ``max_chars``."""
        text = "\n".join(part for part in iter_strings(value) if part)
        for start in range(0, len(text), self.max_chars):
            yield text[start : start + self.max_chars]

    def _check(self, ctx: HookContext, value: Any, origin: str) -> Decision:
        if any(self.detect(ctx, batch) for batch in self._batches(value)):
            return self.violation(detector=type(self.detector).__name__, origin=origin)
        return ALLOW

    def on_input(self, ctx: HookContext, text: str) -> Decision:
        return self._check(ctx, text, "input")

    def before_tool(self, ctx: HookContext, call: ToolCall) -> Decision:
        return self._check(ctx, call.input, "tool_input")

    def after_tool(self, ctx: HookContext, call: ToolCall, result: ToolResult) -> Decision:
        return self._check(ctx, result, "tool_result")

    def on_output(self, ctx: HookContext, answer: Any) -> Decision:
        return self._check(ctx, answer, "output")
