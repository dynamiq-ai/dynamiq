"""An edit must land where the caller meant it to land.

Two ways it could silently land somewhere else: an anchor that matches in more than one
place, and a file whose CRLF line endings stop a ``\\n``-joined anchor from matching at
all. Both are refused or handled rather than guessed at.
"""

import pytest

from dynamiq.nodes.tools.file_tools import FileWriteTool
from dynamiq.runnables import RunnableStatus
from dynamiq.storages.file.in_memory import InMemoryFileStore

THREE_ANCHORS = "def a():\n    return None\n\ndef b():\n    return None\n\ndef c():\n    return None\n"


@pytest.fixture
def store():
    return InMemoryFileStore()


@pytest.fixture
def tool(store):
    return FileWriteTool(file_store=store)


def _write(tool, path, edits, **extra):
    return tool.run(input_data={"file_path": path, "action": "edit", "edits": edits, **extra})


def _content(store, path) -> str:
    return store.retrieve(path).decode()


class TestAmbiguousEditIsRefused:
    def test_multiple_matches_are_refused(self, tool, store):
        store.store("app.py", THREE_ANCHORS.encode(), content_type="text/x-python")

        result = _write(tool, "app.py", [{"find": "return None", "replace": "return 1"}])

        assert result.status == RunnableStatus.SUCCESS, "refusal is reported to the model, not raised"
        assert "ambiguous" in result.output["content"].lower()

    def test_file_is_left_untouched_when_refused(self, tool, store):
        store.store("app.py", THREE_ANCHORS.encode(), content_type="text/x-python")

        _write(tool, "app.py", [{"find": "return None", "replace": "return 1"}])

        assert _content(store, "app.py") == THREE_ANCHORS

    def test_message_reports_the_match_count_and_which_edit(self, tool, store):
        """The model needs the count to know how much more context to add, and the edit
        number to know which one to change when a request carries several."""
        store.store("app.py", THREE_ANCHORS.encode(), content_type="text/x-python")

        result = _write(tool, "app.py", [{"find": "return None", "replace": "return 1"}])
        message = result.output["content"]

        assert "matched 3 places" in message
        assert "edit 1" in message

    def test_replace_all_still_replaces_every_match(self, tool, store):
        store.store("app.py", THREE_ANCHORS.encode(), content_type="text/x-python")

        result = _write(tool, "app.py", [{"find": "return None", "replace": "return 1", "replace_all": True}])

        assert "3 replacement(s)" in result.output["content"]
        assert _content(store, "app.py").count("return 1") == 3

    def test_unique_anchor_still_applies(self, tool, store):
        """The common case must be unchanged."""
        store.store("app.py", THREE_ANCHORS.encode(), content_type="text/x-python")

        result = _write(tool, "app.py", [{"find": "def b():", "replace": "def bee():"}])

        assert "1 replacement(s)" in result.output["content"]
        assert "def bee():" in _content(store, "app.py")

    def test_one_ambiguous_edit_blocks_the_whole_batch(self, tool, store):
        """Partial application would leave the file in a state neither side asked for."""
        store.store("app.py", THREE_ANCHORS.encode(), content_type="text/x-python")

        result = _write(
            tool,
            "app.py",
            [
                {"find": "def b():", "replace": "def bee():"},
                {"find": "return None", "replace": "return 1"},
            ],
        )

        assert "ambiguous" in result.output["content"].lower()
        assert _content(store, "app.py") == THREE_ANCHORS, "no edit may land if any is ambiguous"

    def test_ambiguity_is_judged_against_the_evolving_content(self, tool, store):
        """An earlier edit in the same batch can remove the duplicate that made a later
        anchor ambiguous, so counting against the original file would refuse a batch
        that is in fact unambiguous by the time it runs."""
        store.store("app.py", b"dup\ndup\nkeep\n", content_type="text/plain")

        result = _write(
            tool,
            "app.py",
            [
                {"find": "dup\ndup", "replace": "dup"},  # leaves a single "dup"
                {"find": "dup", "replace": "done"},  # now unique
            ],
        )

        assert "ambiguous" not in result.output["content"].lower()
        assert _content(store, "app.py") == "done\nkeep\n"

    def test_ambiguity_created_by_an_earlier_edit_is_caught(self, tool, store):
        """The mirror case: the batch starts unambiguous and becomes ambiguous."""
        store.store("app.py", b"a\nb\n", content_type="text/plain")

        result = _write(
            tool,
            "app.py",
            [
                {"find": "a", "replace": "b"},  # file becomes "b\nb\n"
                {"find": "b", "replace": "c"},  # now matches twice
            ],
        )

        assert "ambiguous" in result.output["content"].lower()
        assert _content(store, "app.py") == "a\nb\n", "no partial write may survive"

    def test_overlapping_matches_count_as_ambiguous(self, tool, store):
        """`str.count` skips overlaps, so "aa" in "aaa" would look unique even though an
        edit could mean either position."""
        store.store("app.txt", b"aaa", content_type="text/plain")

        result = _write(tool, "app.txt", [{"find": "aa", "replace": "b"}])

        assert "ambiguous" in result.output["content"].lower()
        assert store.retrieve("app.txt") == b"aaa"

    def test_replace_all_reports_the_replacements_it_actually_made(self, tool, store):
        """Overlapping positions decide ambiguity, but `str.replace` is non-overlapping,
        so the reported count must follow the replacement, not the position count."""
        store.store("app.txt", b"aaa", content_type="text/plain")

        result = _write(tool, "app.txt", [{"find": "aa", "replace": "B", "replace_all": True}])

        assert store.retrieve("app.txt") == b"Ba"
        assert "1 replacement(s)" in result.output["content"]

    def test_missing_anchor_still_reported_separately(self, tool, store):
        """The pre-existing not-found path must keep its own wording."""
        store.store("app.py", THREE_ANCHORS.encode(), content_type="text/x-python")

        result = _write(tool, "app.py", [{"find": "nonexistent", "replace": "x"}])

        assert "not found" in result.output["content"]
        assert "ambiguous" not in result.output["content"].lower()


class TestCRLFFiles:
    def test_lf_anchor_matches_a_crlf_file(self, tool, store):
        store.store("crlf.py", b"def a():\r\n    return None\r\n", content_type="text/x-python")

        result = _write(tool, "crlf.py", [{"find": "def a():\n    return None", "replace": "def a():\n    return 1"}])

        assert "1 replacement(s)" in result.output["content"]

    def test_crlf_line_endings_are_preserved_on_write(self, tool, store):
        store.store("crlf.py", b"def a():\r\n    return None\r\n", content_type="text/x-python")

        _write(tool, "crlf.py", [{"find": "def a():\n    return None", "replace": "def a():\n    return 1"}])

        written = store.retrieve("crlf.py")
        assert written == b"def a():\r\n    return 1\r\n"
        assert b"\n" not in written.replace(b"\r\n", b"")

    def test_explicit_crlf_anchor_still_works(self, tool, store):
        """Callers that already send \\r\\n must not regress."""
        store.store("crlf.py", b"a\r\nb\r\n", content_type="text/plain")

        result = _write(tool, "crlf.py", [{"find": "a\r\nb", "replace": "a\r\nc"}])

        assert "1 replacement(s)" in result.output["content"]
        assert store.retrieve("crlf.py") == b"a\r\nc\r\n"

    def test_lf_file_is_untouched_by_crlf_handling(self, tool, store):
        store.store("lf.py", b"a\nb\n", content_type="text/plain")

        _write(tool, "lf.py", [{"find": "a\nb", "replace": "a\nc"}])

        assert store.retrieve("lf.py") == b"a\nc\n"

    def test_mixed_line_endings_are_not_normalized(self, tool, store):
        """Rewriting untouched lines would be a silent, unrequested change."""
        store.store("mixed.txt", b"one\r\ntwo\nthree\r\n", content_type="text/plain")

        _write(tool, "mixed.txt", [{"find": "two", "replace": "2"}])

        assert store.retrieve("mixed.txt") == b"one\r\n2\nthree\r\n"

    def test_ambiguity_is_detected_in_crlf_files_too(self, tool, store):
        store.store("crlf.py", b"x = 1\r\nx = 1\r\n", content_type="text/plain")

        result = _write(tool, "crlf.py", [{"find": "x = 1", "replace": "x = 2"}])

        assert "ambiguous" in result.output["content"].lower()
        assert store.retrieve("crlf.py") == b"x = 1\r\nx = 1\r\n"
