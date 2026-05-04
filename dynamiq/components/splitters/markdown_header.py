import re
from copy import deepcopy
from typing import Any

from dynamiq.types import Document
from dynamiq.utils.logger import logger


class MarkdownHeaderSplitterComponent:
    """Splits Markdown text on header markers, carrying the header path in metadata.

    Each output chunk's ``metadata`` is enriched with ``{header_key: header_text}``
    for every active header level above the chunk.
    """

    DEFAULT_HEADERS_TO_SPLIT_ON: list[tuple[str, str]] = [
        ("#", "h1"),
        ("##", "h2"),
        ("###", "h3"),
        ("####", "h4"),
        ("#####", "h5"),
        ("######", "h6"),
    ]

    def __init__(
        self,
        headers_to_split_on: list[tuple[str, str]] | None = None,
        strip_headers: bool = True,
        return_each_line: bool = False,
    ) -> None:
        self.headers_to_split_on = sorted(
            headers_to_split_on or self.DEFAULT_HEADERS_TO_SPLIT_ON,
            key=lambda item: len(item[0]),
            reverse=True,
        )
        self.strip_headers = strip_headers
        self.return_each_line = return_each_line

    def run(self, documents: list[Document]) -> dict[str, list[Document]]:
        if not isinstance(documents, list):
            raise TypeError("MarkdownHeaderSplitter expects a list of Documents as input.")
        results: list[Document] = []
        for doc in documents:
            if doc.content is None:
                raise ValueError(f"MarkdownHeaderSplitter requires text content; document ID {doc.id} has none.")
            for chunk in self.split_text(doc.content):
                metadata = deepcopy(doc.metadata) if doc.metadata else {}
                metadata["source_id"] = doc.id
                metadata.update(chunk["metadata"])
                results.append(Document(content=chunk["content"], metadata=metadata))
        logger.debug(f"MarkdownHeaderSplitter: split {len(documents)} documents into {len(results)} chunks.")
        return {"documents": results}

    def split_text(self, text: str) -> list[dict[str, Any]]:
        lines = text.split("\n")
        active_headers: dict[str, str] = {}
        header_stack: list[tuple[int, str]] = []
        chunks: list[dict[str, Any]] = []
        current_lines: list[str] = []
        in_code_block = False
        opening_fence = ""

        def flush(metadata: dict[str, str]) -> None:
            content = "\n".join(line for line in current_lines if line)
            if content.strip():
                chunks.append({"content": content, "metadata": dict(metadata)})

        for line in lines:
            stripped = line.strip()
            if not in_code_block:
                opening_fence = self._opening_code_fence(stripped)
                if opening_fence:
                    in_code_block = True
            elif self._is_closing_code_fence(stripped, opening_fence):
                in_code_block = False
                opening_fence = ""

            header_match = None
            if not in_code_block:
                for prefix, key in self.headers_to_split_on:
                    if stripped.startswith(prefix) and (len(stripped) == len(prefix) or stripped[len(prefix)] == " "):
                        header_match = (prefix, key, stripped[len(prefix) :].strip())
                        break

            if header_match is not None:
                if current_lines:
                    flush(active_headers)
                    current_lines = []
                level = len(header_match[0])
                while header_stack and header_stack[-1][0] >= level:
                    _, popped_key = header_stack.pop()
                    active_headers.pop(popped_key, None)
                header_stack.append((level, header_match[1]))
                active_headers[header_match[1]] = header_match[2]
                if not self.strip_headers:
                    current_lines.append(line)
                continue

            if self.return_each_line and stripped:
                chunks.append({"content": line, "metadata": dict(active_headers)})
            else:
                current_lines.append(line)

        if current_lines:
            flush(active_headers)
        return chunks

    @staticmethod
    def _opening_code_fence(stripped: str) -> str:
        match = re.match(r"^(`{3,}|~{3,})(.*)$", stripped)
        if match is None:
            return ""
        fence = match.group(1)
        if fence[0] == "`" and "`" in match.group(2):
            return ""
        return fence

    @staticmethod
    def _is_closing_code_fence(stripped: str, opening_fence: str) -> bool:
        if not opening_fence:
            return False
        fence_char = re.escape(opening_fence[0])
        min_length = len(opening_fence)
        return re.match(rf"^{fence_char}{{{min_length},}}\s*$", stripped) is not None
