import re
from typing import ClassVar

from dynamiq.components.splitters.base import SplitterComponentBase


class RecursiveCharacterSplitterComponent(SplitterComponentBase):
    """Recursive character splitter.

    Walks a hierarchy of separators (largest -> smallest) and recursively splits any
    chunk that still exceeds ``chunk_size``.
    """

    DEFAULT_SEPARATORS: ClassVar[list[str]] = ["\n\n", "\n", " ", ""]

    def __init__(
        self,
        separators: list[str] | None = None,
        is_separator_regex: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.separators = list(separators) if separators is not None else list(self.DEFAULT_SEPARATORS)
        self.is_separator_regex = is_separator_regex

    def split_text(self, text: str) -> list[str]:
        return self._split_text(text, self.separators)

    def _constructor_kwargs(self) -> dict:
        kwargs = super()._constructor_kwargs()
        kwargs.update(separators=self.separators, is_separator_regex=self.is_separator_regex)
        return kwargs

    def _split_text(self, text: str, separators: list[str]) -> list[str]:
        final_chunks: list[str] = []
        separator = separators[-1]
        new_separators: list[str] = []
        for index, candidate in enumerate(separators):
            candidate_pattern = candidate if self.is_separator_regex else re.escape(candidate)
            if candidate == "":
                separator = candidate
                break
            if re.search(candidate_pattern, text):
                separator = candidate
                new_separators = separators[index + 1 :]
                break

        merge_separator = "" if self.keep_separator else separator
        # A regex matches different text in different places, so each piece is rejoined with what stood before it.
        rejoin_matches = self.is_separator_regex and not self.keep_separator and separator != ""
        if rejoin_matches:
            splits, gaps = self._split_with_matches(text, separator)
        else:
            splits, gaps = self._split_with_separator(text, separator), []
        good_splits: list[str] = []
        good_gaps: list[str] = []
        for index, piece in enumerate(splits):
            if self._length(piece) < self.chunk_size:
                if good_splits and rejoin_matches:
                    good_gaps.append(gaps[index - 1])
                good_splits.append(piece)
                continue
            if good_splits:
                final_chunks.extend(
                    self._merge_splits(good_splits, merge_separator, good_gaps if rejoin_matches else None)
                )
                good_splits, good_gaps = [], []
            if not new_separators:
                final_chunks.append(piece)
            else:
                final_chunks.extend(self._split_text(piece, new_separators))
        if good_splits:
            final_chunks.extend(self._merge_splits(good_splits, merge_separator, good_gaps if rejoin_matches else None))
        return final_chunks

    def _split_with_matches(self, text: str, pattern: str) -> tuple[list[str], list[str]]:
        """Split on a regex, returning the non-empty pieces and the text between each piece and the next."""
        pieces: list[str] = []
        gaps: list[str] = []
        pending_gap = ""
        start = 0
        for match in re.finditer(pattern, text):
            piece = text[start : match.start()]
            if piece:
                if pieces:
                    gaps.append(pending_gap)
                pieces.append(piece)
                pending_gap = ""
            pending_gap += match.group(0)
            start = match.end()
        piece = text[start:]
        if piece:
            if pieces:
                gaps.append(pending_gap)
            pieces.append(piece)
        return pieces, gaps

    def _split_with_separator(self, text: str, separator: str) -> list[str]:
        if separator == "":
            return list(text)
        pattern = separator if self.is_separator_regex else re.escape(separator)
        if self.keep_separator:
            parts = re.split(f"({pattern})", text)
            merged = [parts[0]] if parts else []
            for index in range(1, len(parts), 2):
                separator_text = parts[index]
                following = parts[index + 1] if index + 1 < len(parts) else ""
                merged.append(separator_text + following)
            return [piece for piece in merged if piece]
        return [piece for piece in re.split(pattern, text) if piece]
