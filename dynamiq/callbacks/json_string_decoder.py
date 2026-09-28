r"""Incremental decoding of JSON string bodies for streaming.

Streamed text a client renders directly must be decoded from JSON source first, or
escapes reach the user verbatim (``The request \"make work.\"``). Streamed text a
client parses again must keep its escapes. This module covers the first case.
"""

_ESCAPES = {'"': '"', "\\": "\\", "/": "/", "b": "\b", "f": "\f", "n": "\n", "r": "\r", "t": "\t"}

_HEX_DIGITS = frozenset("0123456789abcdefABCDEF")

# A lone surrogate is not a character and raises when the event is UTF-8 encoded.
_REPLACEMENT = "�"


class JSONStringDecoder:
    r"""Decode the body of a JSON string, one fragment at a time.

    Feed the raw source *between* a JSON string's quotes, in whatever fragments it
    arrives; get back decoded text. An escape sequence split across a fragment
    boundary — a fragment ending on a lone ``\`` , or partway through ``\uXXXX`` — is
    held internally until the rest arrives, so callers never see half an escape.

    For any complete, valid string body::

        "".join(dec.feed(f) for f in fragments) + dec.flush()
            == json.loads('"' + "".join(fragments) + '"')

    Decoding is lenient where ``json.loads`` would raise: an unknown escape (``\q``)
    yields the character itself, and a malformed ``\uXXXX`` is passed through as
    written, so a single bad byte degrades one character rather than the stream.
    """

    def __init__(self) -> None:
        self._escaped = False
        self._unicode_digits: str | None = None
        self._pending_high_surrogate: int | None = None

    def feed(self, fragment: str) -> str:
        """Decode a fragment, returning the text that resolved from it."""
        if not fragment:
            return ""
        decoded = []
        for char in fragment:
            piece = self._feed_char(char)
            if piece:
                decoded.append(piece)
        return "".join(decoded)

    def flush(self) -> str:
        """Finish the string, draining a surrogate whose partner never arrived.

        A trailing escape cut off mid-sequence is discarded: the string was truncated,
        so there is no character to recover.
        """
        self._escaped = False
        self._unicode_digits = None
        return self._drain_surrogate()

    def _drain_surrogate(self) -> str:
        if self._pending_high_surrogate is None:
            return ""
        self._pending_high_surrogate = None
        return _REPLACEMENT

    def _emit(self, text: str) -> str:
        """Return ``text``, preceded by any orphaned high surrogate it displaces."""
        return self._drain_surrogate() + text

    def _feed_char(self, char: str) -> str:
        if self._unicode_digits is not None:
            return self._feed_unicode_digit(char)

        if self._escaped:
            self._escaped = False
            if char == "u":
                self._unicode_digits = ""
                return ""
            return self._emit(_ESCAPES.get(char, char))

        if char == "\\":
            self._escaped = True
            return ""

        return self._emit(char)

    def _feed_unicode_digit(self, char: str) -> str:
        r"""Collect the four hex digits of a ``\uXXXX`` escape, then resolve it."""
        self._unicode_digits += char
        if len(self._unicode_digits) < 4:
            return ""

        digits, self._unicode_digits = self._unicode_digits, None
        if not all(digit in _HEX_DIGITS for digit in digits):
            return self._emit(f"\\u{digits}")

        code = int(digits, 16)

        if 0xD800 <= code <= 0xDBFF:
            orphan = self._drain_surrogate()
            self._pending_high_surrogate = code
            return orphan

        if 0xDC00 <= code <= 0xDFFF:
            if self._pending_high_surrogate is None:
                return self._emit(_REPLACEMENT)
            high = self._pending_high_surrogate
            self._pending_high_surrogate = None
            return chr(0x10000 + ((high - 0xD800) << 10) + (code - 0xDC00))

        return self._emit(chr(code))
