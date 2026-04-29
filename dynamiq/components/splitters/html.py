from copy import deepcopy
from importlib.util import find_spec
from typing import Any

from dynamiq.types import Document
from dynamiq.utils.logger import logger


class HTMLHeaderSplitterComponent:
    """Splits HTML content on header tags (``h1``..``h6``), keeping a header-path in metadata."""

    DEFAULT_HEADERS_TO_SPLIT_ON: list[tuple[str, str]] = [
        ("h1", "h1"),
        ("h2", "h2"),
        ("h3", "h3"),
        ("h4", "h4"),
    ]

    def __init__(
        self,
        headers_to_split_on: list[tuple[str, str]] | None = None,
        return_each_element: bool = False,
    ) -> None:
        self.headers_to_split_on = headers_to_split_on or self.DEFAULT_HEADERS_TO_SPLIT_ON
        self.return_each_element = return_each_element

    def run(self, documents: list[Document]) -> dict[str, list[Document]]:
        if not isinstance(documents, list):
            raise TypeError("HTMLHeaderSplitter expects a list of Documents as input.")
        results: list[Document] = []
        for doc in documents:
            if doc.content is None:
                raise ValueError(f"HTMLHeaderSplitter requires text content; document ID {doc.id} has none.")
            for chunk in self.split_text(doc.content):
                metadata = deepcopy(doc.metadata) if doc.metadata else {}
                metadata["source_id"] = doc.id
                metadata.update(chunk["metadata"])
                results.append(Document(content=chunk["content"], metadata=metadata))
        logger.debug(f"HTMLHeaderSplitter: split {len(documents)} documents into {len(results)} chunks.")
        return {"documents": results}

    def split_text(self, text: str) -> list[dict[str, Any]]:
        try:
            from bs4 import BeautifulSoup
        except ImportError as exc:
            raise ImportError(
                "HTMLHeaderSplitter requires the 'beautifulsoup4' package. "
                "Install with `pip install beautifulsoup4 lxml`."
            ) from exc

        soup = BeautifulSoup(text, "lxml" if _has_lxml() else "html.parser")
        header_tags = {tag for tag, _ in self.headers_to_split_on}

        chunks: list[dict[str, Any]] = []
        active_headers: dict[str, str] = {}
        buffer_lines: list[str] = []

        def flush() -> None:
            content = "\n".join(line for line in buffer_lines if line.strip())
            if content.strip():
                chunks.append({"content": content, "metadata": dict(active_headers)})

        for element in soup.body.descendants if soup.body else soup.descendants:
            if getattr(element, "name", None) in header_tags:
                if buffer_lines:
                    flush()
                    buffer_lines = []
                level = int(element.name[1])
                for tag, key in list(active_headers.items()):
                    if tag.startswith("h") and int(tag[1]) >= level:
                        active_headers.pop(tag, None)
                _, metadata_key = next(pair for pair in self.headers_to_split_on if pair[0] == element.name)
                active_headers[metadata_key] = element.get_text(strip=True)
                continue
            if getattr(element, "name", None) is None:  # NavigableString
                text_value = str(element).strip()
                if text_value:
                    if self.return_each_element:
                        chunks.append({"content": text_value, "metadata": dict(active_headers)})
                    else:
                        buffer_lines.append(text_value)
        if buffer_lines:
            flush()
        return chunks


class HTMLSectionSplitterComponent(HTMLHeaderSplitterComponent):
    """Variant that returns one chunk per HTML section, mirroring LangChain's ``HTMLSectionSplitter``."""

    def __init__(
        self,
        headers_to_split_on: list[tuple[str, str]] | None = None,
        xpath_filter: str | None = None,
    ) -> None:
        super().__init__(headers_to_split_on=headers_to_split_on, return_each_element=False)
        self.xpath_filter = xpath_filter


def _has_lxml() -> bool:
    return find_spec("lxml") is not None
