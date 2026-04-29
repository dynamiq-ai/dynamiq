import enum

from dynamiq.components.splitters.language import Language, get_separators_for_language
from dynamiq.components.splitters.recursive_character import RecursiveCharacterSplitterComponent


class CodeParser(str, enum.Enum):
    """Supported code parsers for the :class:`CodeSplitterComponent`."""

    REGEX = "regex"
    TREE_SITTER = "tree_sitter"


class CodeSplitterComponent(RecursiveCharacterSplitterComponent):
    """Code-aware splitter.

    Uses the language separator presets from :func:`get_separators_for_language` for
    fast regex-based splitting, with an optional ``tree_sitter`` parser for
    AST-aware boundaries.
    """

    def __init__(
        self,
        language: Language = Language.PYTHON,
        parser: CodeParser = CodeParser.REGEX,
        **kwargs,
    ) -> None:
        kwargs.setdefault("separators", get_separators_for_language(language))
        kwargs.setdefault("is_separator_regex", True)
        kwargs.setdefault("keep_separator", True)
        super().__init__(**kwargs)
        self.language = language
        self.parser = parser

    def split_text(self, text: str) -> list[str]:
        if self.parser == CodeParser.TREE_SITTER:
            tree_chunks = self._split_with_tree_sitter(text)
            if tree_chunks is not None:
                return tree_chunks
        return super().split_text(text)

    def _split_with_tree_sitter(self, text: str) -> list[str] | None:
        try:
            from tree_sitter_languages import get_parser
        except ImportError:
            return None
        parser = get_parser(self.language.value)
        tree = parser.parse(text.encode("utf-8"))
        node_kinds = {"function_definition", "class_definition", "method_definition", "function_declaration"}
        boundaries: list[tuple[int, int]] = []
        cursor = tree.root_node.walk()

        def visit(node) -> None:
            if node.type in node_kinds:
                boundaries.append((node.start_byte, node.end_byte))
            for child in node.children:
                visit(child)

        visit(cursor.node)
        if not boundaries:
            return None
        boundaries.sort()
        chunks: list[str] = []
        previous_end = 0
        encoded = text.encode("utf-8")
        for start, end in boundaries:
            if start > previous_end:
                chunks.append(encoded[previous_end:start].decode("utf-8", errors="ignore"))
            chunks.append(encoded[start:end].decode("utf-8", errors="ignore"))
            previous_end = end
        if previous_end < len(encoded):
            chunks.append(encoded[previous_end:].decode("utf-8", errors="ignore"))
        merged: list[str] = []
        for chunk in chunks:
            if self._length(chunk) <= self.chunk_size:
                merged.append(chunk)
            else:
                merged.extend(super().split_text(chunk))
        return [chunk for chunk in merged if chunk.strip()]
