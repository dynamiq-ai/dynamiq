"""Inline ``thought`` extraction for streaming FC arguments.

Splits the LLM's streaming JSON object into two output streams: the tool's
real params (with ``thought`` removed) and just the thought value content.
"""

INNER_THOUGHTS_DEFAULT_KEY = "thought"


class JSONInnerThoughtsExtractor:
    """Streaming JSON parser that routes the ``thought`` field into a separate buffer."""

    def __init__(
        self,
        inner_thoughts_key: str = INNER_THOUGHTS_DEFAULT_KEY,
        wait_for_first_key: bool = False,
    ) -> None:
        self.inner_thoughts_key = inner_thoughts_key
        self.wait_for_first_key = wait_for_first_key

        # Cumulative buffers across all process_fragment calls.
        self.main_buffer: str = ""
        self.inner_thoughts_buffer: str = ""
        self.main_json_held_buffer: str = ""

        # Parser state.
        self.state: str = "start"
        self.in_string: bool = False
        self.escaped: bool = False
        self.current_key: str = ""
        self.is_inner_thoughts_value: bool = False
        self.inner_thoughts_processed: bool = False
        self.hold_main_json: bool = wait_for_first_key

        # Deferred top-level separator: emitted before the next main field, or dropped.
        self.pending_comma: bool = False

        # Top-level transitions only fire at depth == 1; deeper structures pass through.
        self.depth: int = 0

    @property
    def thought_complete(self) -> bool:
        """Whether the thought field's value has been fully processed."""
        return self.inner_thoughts_processed

    @property
    def held_main_buffer(self) -> str:
        """Held bytes not yet flushed; drained at end-of-stream when thought was missing."""
        return self.main_json_held_buffer

    def process_fragment(self, fragment: str) -> tuple[str, str]:
        """Feed a chunk; returns ``(main_delta, thought_delta)`` for this fragment."""
        updates_main: list[str] = []
        updates_thought: list[str] = []
        for c in fragment:
            main_chunk, thought_chunk = self._process_char(c)
            if main_chunk:
                updates_main.append(main_chunk)
            if thought_chunk:
                updates_thought.append(thought_chunk)
        return "".join(updates_main), "".join(updates_thought)

    def _emit_main(self, s: str) -> str:
        """Append to main buffer (held or live); return the delta to surface."""
        if self.hold_main_json:
            self.main_json_held_buffer += s
            return ""
        self.main_buffer += s
        return s

    def _flush_held_buffer(self) -> str:
        """Move held bytes to the live buffer and surface them as a delta."""
        if not self.main_json_held_buffer:
            self.hold_main_json = False
            return ""
        delta = self.main_json_held_buffer
        self.main_buffer += delta
        self.main_json_held_buffer = ""
        self.hold_main_json = False
        return delta

    def _emit_thought(self, s: str) -> str:
        self.inner_thoughts_buffer += s
        return s

    def _process_char(self, c: str) -> tuple[str, str]:
        """Process a single character and return its ``(main, thought)`` delta."""

        if self.escaped:
            self.escaped = False
            return self._consume_value_char(c)

        if c == "\\":
            self.escaped = True
            if self.in_string:
                return self._consume_value_char(c)
            return "", ""

        if c == '"':
            return self._handle_quote()

        if self.in_string:
            return self._consume_value_char(c)

        # Structural characters (outside any string).
        if c == "{":
            return self._handle_open_object()

        if c == "[":
            return self._handle_open_bracket()

        if c == "}":
            return self._handle_close_object()

        if c == "]":
            return self._handle_close_bracket()

        if self.depth >= 2:
            # Inside nested object/array — passthrough to value's target buffer.
            return self._consume_value_char(c)

        if c == ":" and self.state == "colon":
            return self._handle_colon()

        # A top-level comma always ends the current value — including after a
        # number/array/object value, where state stays "value" (no closing quote).
        if c == "," and self.state in ("comma_or_end", "value"):
            return self._handle_comma()

        if self.state == "value":
            # Non-string scalar in top-level value (number, bool, null).
            return self._consume_value_char(c)

        # Whitespace or non-structural chars outside any value — ignore.
        return "", ""

    def _consume_value_char(self, c: str) -> tuple[str, str]:
        """Route char to thought or main based on whose value we're in."""
        if self.in_string and self.state == "key":
            self.current_key += c
            return "", ""
        if self.is_inner_thoughts_value and self.depth == 1:
            return "", self._emit_thought(c)
        return self._emit_main(c), ""

    def _handle_quote(self) -> tuple[str, str]:
        """Handle an unescaped ``"``."""
        self.in_string = not self.in_string

        if self.in_string:
            # Opening quote.
            if self.depth >= 2:
                # Inside nested value — passthrough.
                return self._emit_main('"'), ""

            if self.state in ("start", "comma_or_end"):
                # Start of a new top-level key — flush held bytes if thought is done.
                main_delta = ""
                if self.wait_for_first_key and self.hold_main_json and self.inner_thoughts_processed:
                    main_delta = self._flush_held_buffer()
                self.state = "key"
                self.current_key = ""
                return main_delta, ""

            if self.state == "value":
                if self.is_inner_thoughts_value:
                    return "", ""
                return self._emit_main('"'), ""

            return "", ""

        # Closing quote.
        if self.depth >= 2:
            return self._emit_main('"'), ""

        if self.state == "key":
            self.state = "colon"
            return "", ""

        if self.state == "value":
            if self.is_inner_thoughts_value:
                self.inner_thoughts_processed = True
                self.state = "comma_or_end"
                return "", ""
            self.state = "comma_or_end"
            return self._emit_main('"'), ""

        return "", ""

    def _handle_open_object(self) -> tuple[str, str]:
        self.depth += 1
        if self.depth == 1:
            # Outermost ``{``.
            return self._emit_main("{"), ""
        # Nested object literal as a value — passthrough.
        if self.is_inner_thoughts_value and self.depth == 2:
            return "", self._emit_thought("{")
        return self._emit_main("{"), ""

    def _handle_open_bracket(self) -> tuple[str, str]:
        self.depth += 1
        if self.is_inner_thoughts_value and self.depth == 2:
            return "", self._emit_thought("[")
        return self._emit_main("["), ""

    def _handle_close_object(self) -> tuple[str, str]:
        self.depth -= 1
        if self.depth >= 1:
            # Closing a nested object — passthrough.
            if self.is_inner_thoughts_value and self.depth == 1:
                return "", self._emit_thought("}")
            return self._emit_main("}"), ""

        # Outermost `}`. A deferred separator (thought-last case) is dropped:
        # nothing follows it, so no comma was ever emitted to revoke.
        self.pending_comma = False
        self.state = "end"
        if self.hold_main_json:
            self.main_json_held_buffer += "}"
            return "", ""
        self.main_buffer += "}"
        return "}", ""

    def _handle_close_bracket(self) -> tuple[str, str]:
        self.depth -= 1
        if self.is_inner_thoughts_value and self.depth == 1:
            return "", self._emit_thought("]")
        return self._emit_main("]"), ""

    def _handle_colon(self) -> tuple[str, str]:
        """Top-level `:` — colon → value transition."""
        self.state = "value"
        self.is_inner_thoughts_value = self.current_key == self.inner_thoughts_key
        if self.is_inner_thoughts_value:
            # Skip the `"thought":` prefix from main.
            return "", ""
        # Surface any deferred separator right before this field's key.
        prefix = "," if self.pending_comma else ""
        self.pending_comma = False
        return self._emit_main(f'{prefix}"{self.current_key}":'), ""

    def _handle_comma(self) -> tuple[str, str]:
        """Top-level `,` — separates fields."""
        if self.is_inner_thoughts_value:
            # Drop comma after thought to avoid a dangling separator.
            self.is_inner_thoughts_value = False
            self.state = "start"
            return "", ""

        # Defer the separator until the next main field confirms it's needed.
        self.pending_comma = True
        self.state = "start"
        return "", ""
