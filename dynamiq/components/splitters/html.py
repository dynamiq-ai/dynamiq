from copy import deepcopy
from importlib.util import find_spec
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict, model_validator

from dynamiq.types import Document
from dynamiq.utils.logger import logger


class HTMLHeaderSplitterComponent(BaseModel):
    """Splits HTML content on header tags (``h1``..``h6``), keeping a header-path in metadata."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    DEFAULT_HEADERS_TO_SPLIT_ON: ClassVar[list[tuple[str, str]]] = [
        ("h1", "h1"),
        ("h2", "h2"),
        ("h3", "h3"),
        ("h4", "h4"),
    ]

    headers_to_split_on: list[tuple[str, str]] | None = None
    return_each_element: bool = False

    @model_validator(mode="after")
    def set_default_headers(self) -> "HTMLHeaderSplitterComponent":
        if self.headers_to_split_on is None:
            self.headers_to_split_on = list(self.DEFAULT_HEADERS_TO_SPLIT_ON)
        return self

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
        header_levels_by_key = {metadata_key: int(tag[1:]) for tag, metadata_key in self.headers_to_split_on}

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
                level = int(element.name[1:])
                for metadata_key in list(active_headers):
                    if header_levels_by_key[metadata_key] >= level:
                        active_headers.pop(metadata_key, None)
                _, metadata_key = next(pair for pair in self.headers_to_split_on if pair[0] == element.name)
                active_headers[metadata_key] = element.get_text(strip=True)
                continue
            if getattr(element, "name", None) is None:  # NavigableString
                if any(getattr(parent, "name", None) in header_tags for parent in getattr(element, "parents", [])):
                    continue
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

    xpath_filter: str | None = None

    def split_text(self, text: str) -> list[dict[str, Any]]:
        if not self.xpath_filter:
            return super().split_text(text)

        try:
            from lxml import etree, html  # nosec B410
        except ImportError as exc:
            raise ImportError("HTMLSectionSplitter xpath_filter requires the 'lxml' package.") from exc

        try:
            tree = html.fromstring(text)
            selected = tree.xpath(self.xpath_filter)
        except etree.XPathError as exc:
            raise ValueError(f"Invalid xpath_filter: {self.xpath_filter}") from exc

        scoped_html: list[str] = []
        for item in selected:
            if hasattr(item, "tag"):
                scoped_html.append(html.tostring(item, encoding="unicode"))
            elif item is not None:
                scoped_html.append(str(item))
        if not scoped_html:
            return []

        return super().split_text("\n".join(scoped_html))


def _has_lxml() -> bool:
    return find_spec("lxml") is not None
