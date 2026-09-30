"""Before/after hooks for the agent's LLM calls (``Agent.model_hooks``).

``ModelHook`` is Python-first: subclass it and override ``apply_before`` / ``apply_after`` / ``should_block`` /
``should_block_output``; in YAML reference the subclass with ``type: my_pkg.hooks.MyHook``. Two built-in subclasses
cover the common YAML-only cases: ``RegexRedactModelHook`` and ``RegexGuardModelHook``.

A blocked call either fails the run (``on_block="fail"``, default) or ends it successfully with ``block_message``
as the answer (``on_block="answer"``). A hook that raises is handled by ``on_error`` like tool hooks.

Streaming: ``apply_after`` changes what is stored and parsed, not chunks already streamed to the client.
"""

import re
from enum import Enum
from typing import Any, ClassVar, Literal

from pydantic import Field, field_validator

from dynamiq.nodes.agents.exceptions import (
    HookStopException,
    ModelCallBlockedException,
    ModelHookStopException,
)
from dynamiq.nodes.agents.hooks import HookBase, HookErrorPolicy, _preview, _record
from dynamiq.prompts import Message, MessageRole, VisionMessage, VisionMessageTextContent
from dynamiq.utils.logger import logger


class ModelBlockAction(str, Enum):
    """What a blocked model call does.

    Attributes:
        FAIL: End the run with a failure. Default: loud, and never hands a refusal to downstream nodes as data.
        ANSWER: End the run successfully with ``block_message`` as the answer (the output carries ``blocked: true``).
    """

    FAIL = "fail"
    ANSWER = "answer"


class ModelHook(HookBase):
    """Before/after hook for LLM calls. Subclass and override what you need.

    Attributes:
        block_message: Message used as the failure reason or as the answer when a call is blocked.
        on_block: What a block does (see ``ModelBlockAction``).
        on_error: What to do if the hook itself raises. ``block`` acts like a block (per ``on_block``).
    """

    block_message: str = "Model call blocked by hook."
    on_block: ModelBlockAction = ModelBlockAction.FAIL
    on_error: HookErrorPolicy = HookErrorPolicy.BLOCK

    _is_base_hook: ClassVar[bool] = True

    def apply_before(self, messages: list[Message | VisionMessage]) -> list[Message | VisionMessage]:
        """Return the messages to send to the model. Do not mutate the given messages: copy them."""
        return messages

    def apply_after(self, output: dict[str, Any]) -> dict[str, Any]:
        """Return the (possibly rewritten) model output: ``{"content": ..., "tool_calls": ...}``."""
        return output

    def should_block(self, messages: list[Message | VisionMessage]) -> bool:
        """Veto the call before it is made (no tokens spent)."""
        return False

    def should_block_output(self, output: dict[str, Any]) -> bool:
        """Veto the reply after the call (tokens spent; the reply is discarded)."""
        return False

    def block(self) -> None:
        if self.on_block == ModelBlockAction.ANSWER:
            raise ModelCallBlockedException(self.block_message)
        raise ModelHookStopException(self.block_message)


def message_text(message: Message | VisionMessage) -> str:
    if isinstance(message, Message):
        return message.content or ""
    return " ".join(part.text for part in message.content if isinstance(part, VisionMessageTextContent))


def _map_text(message: Message | VisionMessage, fn) -> Message | VisionMessage:
    """Copy of ``message`` with ``fn`` applied to its text (images/files untouched)."""
    if isinstance(message, Message):
        return message.model_copy(update={"content": fn(message.content or "")})
    parts = [
        part.model_copy(update={"text": fn(part.text)}) if isinstance(part, VisionMessageTextContent) else part
        for part in message.content
    ]
    return message.model_copy(update={"content": parts})


def _run_model_hook(hook: ModelHook, index: int, phase: str, fn, fallback: Any, trace: list[dict] | None):
    """Run one hook step; a crash is handled per ``hook.on_error`` (deliberate hook exceptions pass through)."""
    try:
        return fn()
    except (HookStopException, ModelCallBlockedException):
        raise
    except Exception as e:
        policy = hook.on_error
        detail = f"{type(e).__name__}: {e}"
        logger.error(f"Model hook[{index}] {phase} failed ({detail}); on_error={policy.value}")
        _record(trace, phase="error", hook=index, step=phase, policy=policy.value, error=detail)
        if policy == HookErrorPolicy.SKIP:
            return fallback
        if policy == HookErrorPolicy.STOP:
            raise ModelHookStopException(f"Model hook[{index}] {phase} failed: {e}") from e
        # BLOCK: fail closed, acting like a block (per on_block) but with the failure as the reason.
        failed = hook.model_copy(update={"block_message": f"Model hook[{index}] {phase} failed: {e}"})
        failed.block()


def _blocked(hook: ModelHook, index: int, phase: str, trace: list[dict] | None) -> None:
    logger.warning(f"Model hook[{index}] BLOCKED the {phase} (on_block={hook.on_block.value}): {hook.block_message}")
    _record(trace, phase=f"block_{phase}", hook=index, on_block=hook.on_block.value, message=hook.block_message)
    hook.block()


def apply_before_model_hooks(
    hooks: list[ModelHook], messages: list[Message | VisionMessage], trace: list[dict] | None = None
) -> list[Message | VisionMessage]:
    """Run hooks in list order: rewrite, then veto. Returns the messages to send."""
    for index, hook in enumerate(hooks):
        rewritten = _run_model_hook(hook, index, "input", lambda h=hook: h.apply_before(messages), messages, trace)
        if rewritten is not messages:
            logger.info(f"Model hook[{index}] input: {len(messages)} -> {len(rewritten)} message(s), rewritten")
            _record(trace, phase="input", hook=index, messages=len(rewritten))
            messages = rewritten
        if _run_model_hook(hook, index, "block", lambda h=hook: h.should_block(messages), False, trace):
            _blocked(hook, index, "input", trace)
    return messages


def apply_after_model_hooks(
    hooks: list[ModelHook], output: dict[str, Any], trace: list[dict] | None = None
) -> tuple[dict[str, Any], bool]:
    """Run hooks in reverse order: veto, then rewrite. Returns ``(output, rewritten)``.

    Like tool output hooks, the original reply is not logged or traced (only sizes and the rewritten preview).
    """
    rewritten_any = False
    for index in range(len(hooks) - 1, -1, -1):
        hook = hooks[index]
        if _run_model_hook(hook, index, "block_output", lambda h=hook: h.should_block_output(output), False, trace):
            _blocked(hook, index, "output", trace)
        rewritten = _run_model_hook(hook, index, "output", lambda h=hook: h.apply_after(output), output, trace)
        if rewritten is not output:
            before_chars = len(str(output.get("content", "")))
            after = _preview(rewritten.get("content", ""))
            logger.info(f"Model hook[{index}] output: {before_chars} chars -> {after}")
            _record(trace, phase="output", hook=index, before_chars=before_chars, after=after)
            output, rewritten_any = rewritten, True
    return output, rewritten_any


class _RegexRules(ModelHook):
    """Shared regex compilation for the built-in hooks."""

    apply_to: Literal["input", "output", "both"] = "both"

    @staticmethod
    def _compile(patterns: list[str]) -> list[re.Pattern]:
        return [re.compile(p) for p in patterns]

    def _checks_input(self) -> bool:
        return self.apply_to in ("input", "both")

    def _checks_output(self) -> bool:
        return self.apply_to in ("output", "both")


def _validate_patterns(patterns: list[str]) -> list[str]:
    for pattern in patterns:
        try:
            re.compile(pattern)
        except re.error as e:
            raise ValueError(f"invalid regex {pattern!r}: {e}") from e
    return patterns


class RegexRedactModelHook(_RegexRules):
    """Mask regex matches in the messages sent to the model and/or in the model's reply. YAML-only.

    ``system`` messages are left alone. The agent's own history is never modified: only what is sent is masked.
    """

    patterns: list[str] = Field(default_factory=list, description="Regexes to mask.")
    replacement: str = "[REDACTED]"

    @field_validator("patterns")
    @classmethod
    def _check_patterns(cls, value):
        return _validate_patterns(value)

    def _mask(self, text: str) -> str:
        for pattern in self._compile(self.patterns):
            text = pattern.sub(self.replacement, text)
        return text

    def apply_before(self, messages):
        if not self._checks_input() or not self.patterns:
            return messages
        return [m if m.role == MessageRole.SYSTEM else _map_text(m, self._mask) for m in messages]

    def apply_after(self, output):
        content = output.get("content")
        if not self._checks_output() or not self.patterns or not isinstance(content, str):
            return output
        masked = self._mask(content)
        return output if masked == content else {**output, "content": masked}


class RegexGuardModelHook(_RegexRules):
    """Block the call (input) or discard the reply (output) when any regex matches. YAML-only guardrail.

    ``system`` messages are not scanned.
    """

    block_if_matches: list[str] = Field(default_factory=list, description="Regexes that trigger the block.")

    @field_validator("block_if_matches")
    @classmethod
    def _check_patterns(cls, value):
        return _validate_patterns(value)

    def _matches(self, text: str) -> bool:
        return any(p.search(text) for p in self._compile(self.block_if_matches))

    def should_block(self, messages):
        if not self._checks_input() or not self.block_if_matches:
            return False
        return any(self._matches(message_text(m)) for m in messages if m.role != MessageRole.SYSTEM)

    def should_block_output(self, output):
        content = output.get("content")
        return bool(
            self._checks_output() and self.block_if_matches and isinstance(content, str) and self._matches(content)
        )
