"""Unit tests for ``JSONInnerThoughtsExtractor``."""

import json

import pytest

from dynamiq.callbacks.inner_thoughts_extractor import JSONInnerThoughtsExtractor


def _drive(raw: str, *, wait_for_first_key: bool = False, char_by_char: bool = False):
    """Feed ``raw`` into a fresh extractor and return ``(main, thought)``.

    When ``wait_for_first_key=True``, the held buffer (if any remains at the
    end) is appended to ``main`` so callers can compare the final state.
    """
    ext = JSONInnerThoughtsExtractor(wait_for_first_key=wait_for_first_key)
    if char_by_char:
        for ch in raw:
            ext.process_fragment(ch)
    else:
        ext.process_fragment(raw)
    main = ext.main_buffer
    if wait_for_first_key and ext.held_main_buffer:
        main += ext.held_main_buffer
    return main, ext.inner_thoughts_buffer


class TestThoughtPositions:
    def test_thought_first(self):
        main, thought = _drive('{"thought":"hi","query":"weather"}')
        assert json.loads(main) == {"query": "weather"}
        assert thought == "hi"

    def test_thought_middle(self):
        main, thought = _drive('{"a":"x","thought":"hi","b":"y"}')
        assert json.loads(main) == {"a": "x", "b": "y"}
        assert thought == "hi"

    def test_thought_last(self):
        # Trailing comma must be stripped before the closing }.
        main, thought = _drive('{"a":"x","thought":"hi"}')
        assert json.loads(main) == {"a": "x"}
        assert thought == "hi"

    def test_thought_missing(self):
        # With wait_for_first_key=False, main streams unchanged.
        main, thought = _drive('{"query":"weather","limit":5}')
        assert json.loads(main) == {"query": "weather", "limit": 5}
        assert thought == ""

    def test_thought_missing_with_wait_for_first_key(self):
        # With wait=True, held buffer never flushes; the safety net is the
        # ``held_main_buffer`` property which the streaming layer flushes
        # at end of stream.
        main, thought = _drive(
            '{"query":"weather","limit":5}', wait_for_first_key=True
        )
        assert json.loads(main) == {"query": "weather", "limit": 5}
        assert thought == ""


class TestWaitForFirstKey:
    def test_thought_late_held_then_flushed(self):
        ext = JSONInnerThoughtsExtractor(wait_for_first_key=True)
        # Feed pre-thought field — main delta must be empty (held).
        main_a, thought_a = ext.process_fragment('{"a":"x",')
        assert main_a == "", "main should be held while thought is unresolved"
        assert thought_a == ""

        # Feed thought.
        main_b, thought_b = ext.process_fragment('"thought":"hi"')
        assert thought_b == "hi"
        assert main_b == "", "main still held until next key starts"

        # Feed next field — held buffer flushes when its opening quote arrives.
        main_c, thought_c = ext.process_fragment(',"b":"y"}')
        assert thought_c == ""
        # First chunk delta carries the held bytes + the new field bytes.
        assert main_c.startswith("{")
        assert json.loads(ext.main_buffer) == {"a": "x", "b": "y"}

    def test_thought_first_flushes_quickly(self):
        ext = JSONInnerThoughtsExtractor(wait_for_first_key=True)
        main, thought = ext.process_fragment('{"thought":"hi","q":"x"}')
        assert thought == "hi"
        assert json.loads(ext.main_buffer) == {"q": "x"}


class TestThoughtInStringValue:
    def test_word_thought_inside_other_value(self):
        # "thought" appears as substring inside the value of `query`.
        main, thought = _drive('{"thought":"hi","query":"what is a thought?"}')
        assert json.loads(main) == {"query": "what is a thought?"}
        assert thought == "hi"

    def test_quoted_thought_key_inside_value(self):
        # Even an escaped `"thought":` inside a value must not be routed.
        raw = '{"thought":"hi","query":"contains \\"thought\\": pattern"}'
        main, thought = _drive(raw)
        assert json.loads(main) == {"query": 'contains "thought": pattern'}
        assert thought == "hi"


class TestNestedStructures:
    def test_nested_object_param(self):
        main, thought = _drive('{"thought":"hi","config":{"x":1,"y":2}}')
        assert json.loads(main) == {"config": {"x": 1, "y": 2}}
        assert thought == "hi"

    def test_deeply_nested(self):
        raw = '{"thought":"hi","a":{"b":{"c":{"d":42}}}}'
        main, thought = _drive(raw)
        assert json.loads(main) == {"a": {"b": {"c": {"d": 42}}}}
        assert thought == "hi"

    def test_nested_thought_key_treated_as_regular(self):
        # Inner `thought` is just a regular field — must NOT be routed.
        raw = '{"thought":"outer","config":{"thought":"inner","x":1}}'
        main, thought = _drive(raw)
        assert json.loads(main) == {"config": {"thought": "inner", "x": 1}}
        assert thought == "outer"

    def test_array_param(self):
        raw = '{"thought":"hi","items":[{"id":1},{"id":2}]}'
        main, thought = _drive(raw)
        assert json.loads(main) == {"items": [{"id": 1}, {"id": 2}]}
        assert thought == "hi"

    def test_array_of_strings(self):
        raw = '{"thought":"hi","tags":["a","b","c"]}'
        main, thought = _drive(raw)
        assert json.loads(main) == {"tags": ["a", "b", "c"]}
        assert thought == "hi"


class TestEscapeSequences:
    def test_escaped_quote_in_thought(self):
        raw = '{"thought":"he said \\"hi\\"","q":"x"}'
        main, thought = _drive(raw)
        assert json.loads(main) == {"q": "x"}
        assert thought == 'he said \\"hi\\"'

    def test_escaped_backslash(self):
        raw = '{"thought":"path \\\\ here","q":"x"}'
        main, thought = _drive(raw)
        assert json.loads(main) == {"q": "x"}

    def test_newline_in_thought(self):
        raw = '{"thought":"line1\\nline2","q":"x"}'
        main, thought = _drive(raw)
        assert json.loads(main) == {"q": "x"}
        assert thought == "line1\\nline2"


class TestChunkedFeeding:
    @pytest.mark.parametrize(
        "raw",
        [
            '{"thought":"hello world","q":"x"}',
            '{"q":"x","thought":"hello world"}',
            '{"thought":"hi","config":{"a":1,"b":2}}',
            '{"thought":"hi","items":[1,2,3]}',
        ],
        ids=["thought_first", "thought_last", "nested_object", "array"],
    )
    def test_char_by_char_matches_whole(self, raw):
        whole_main, whole_thought = _drive(raw)
        char_main, char_thought = _drive(raw, char_by_char=True)
        assert whole_main == char_main
        assert whole_thought == char_thought


class TestStreamingDeltas:
    """Ensure ``process_fragment`` returns proper deltas, not just cumulative state."""

    def test_thought_streams_progressively(self):
        ext = JSONInnerThoughtsExtractor()
        chunks = ['{"th', 'ought":"hel', 'lo wor', 'ld","q":"x"}']
        thoughts = []
        for chunk in chunks:
            _, td = ext.process_fragment(chunk)
            thoughts.append(td)
        assert "".join(thoughts) == "hello world"

    def test_main_streams_progressively_without_wait(self):
        ext = JSONInnerThoughtsExtractor(wait_for_first_key=False)
        chunks = ['{"thought":"hi","q":"x', '","r":"y"}']
        mains = []
        for chunk in chunks:
            md, _ = ext.process_fragment(chunk)
            mains.append(md)
        assert json.loads("".join(mains)) == {"q": "x", "r": "y"}


class TestDeltaBufferInvariant:
    """The streamed main deltas must always reconstruct ``main_buffer`` and valid JSON.

    Guards against eagerly emitting a separator that is later only stripped from
    the buffer — leaving a dangling comma in the delta stream (thought-last case).
    """

    RAWS = [
        '{"thought":"hi","q":"x"}',
        '{"q":"x","thought":"hi"}',  # thought last — the regression case
        '{"a":"x","thought":"hi","b":"y"}',
        '{"a":"x","b":"y"}',  # no thought
        '{"thought":"hi","config":{"a":1,"b":2}}',
        '{"a":"x","items":[1,2,3],"thought":"hi"}',
    ]

    @pytest.mark.parametrize("raw", RAWS)
    @pytest.mark.parametrize("wait", [False, True])
    def test_delta_sum_matches_buffer(self, raw, wait):
        ext = JSONInnerThoughtsExtractor(wait_for_first_key=wait)
        mains = []
        for ch in raw:
            md, _ = ext.process_fragment(ch)
            mains.append(md)
        streamed = "".join(mains)
        # Core invariant: deltas reconstruct the buffer exactly (no phantom comma).
        assert streamed == ext.main_buffer
        # Effective output the streaming layer surfaces = deltas + drained held bytes.
        effective = streamed + ext.held_main_buffer
        parsed = json.loads(effective)
        assert "thought" not in parsed
