import copy
import html as html_module
import html.parser
import re
from io import BytesIO
from pathlib import Path
from typing import Any, Literal

from lxml import etree  # nosec
from lxml import html as lxml_html  # nosec

from dynamiq.components.converters.base import BaseConverter
from dynamiq.components.converters.utils import get_filename_for_bytesio
from dynamiq.types import Document, DocumentCreationMode


class HTMLConverter(BaseConverter):
    """
    A component for converting HTML files to Documents using lxml.

    Initializes the object with the configuration for converting documents using
    lxml HTML parser.

    Args:
        document_creation_mode (Literal["one-doc-per-file"], optional):
            Determines how to create Documents from the HTML content. Currently only supports:
            - `"one-doc-per-file"`: Creates one Document per file.
                All content is converted to markdown format.
            Defaults to `"one-doc-per-file"`.

    Usage example:
        ```python
        from dynamiq.components.converters.html import HTMLConverter

        converter = HTMLConverter()
        documents = converter.run(paths=["a/file/path.html", "a/directory/path"])["documents"]
        ```
    """

    document_creation_mode: Literal[DocumentCreationMode.ONE_DOC_PER_FILE] = DocumentCreationMode.ONE_DOC_PER_FILE

    def _process_file(self, file: Path | BytesIO, metadata: dict[str, Any]) -> list[Any]:
        """
        Process a file and return a list of Documents.

        Args:
            file: Path to a file or BytesIO object
            metadata: Metadata to attach to the documents

        Returns:
            List of Documents
        """
        # Get the file path or name for BytesIO
        if isinstance(file, BytesIO):
            filepath = get_filename_for_bytesio(file)
        else:
            filepath = str(file)

        # Read the file content
        if isinstance(file, BytesIO):
            file.seek(0)
            html_content = file.read().decode("utf-8")
        else:
            with open(file, encoding="utf-8") as f:
                html_content = f.read()

        # Parse the HTML content
        elements = self._parse_html_content(html_content)

        # Create documents from the HTML elements
        return self._create_documents(
            filepath=filepath,
            elements=elements,
            document_creation_mode=self.document_creation_mode,
            metadata=metadata,
        )

    def _parse_html_content(self, html_content: str) -> lxml_html.HtmlElement:
        """
        Parse HTML content using lxml.

        Args:
            html_content: HTML content to parse

        Returns:
            lxml HTML element
        """
        try:
            return lxml_html.fromstring(html_content)
        except etree.ParserError:
            # Handle malformed HTML by cleaning it first
            html_content = self._clean_html(html_content)
            return lxml_html.fromstring(html_content)

    @staticmethod
    def _clean_html(html_content: str) -> str:
        """
        Clean malformed HTML.

        Args:
            html_content: HTML content to clean

        Returns:
            Cleaned HTML content
        """
        parser = html.parser.HTMLParser()
        try:
            return parser.unescape(html_content)
        except AttributeError:
            # For Python 3.9+, HTMLParser.unescape is deprecated
            return html_module.unescape(html_content)

    def _create_documents(
        self,
        filepath: str,
        elements: lxml_html.HtmlElement,
        document_creation_mode: DocumentCreationMode,
        metadata: dict[str, Any],
        **kwargs,
    ) -> list[Document]:
        """
        Create Documents from the HTML elements.
        """
        if document_creation_mode != DocumentCreationMode.ONE_DOC_PER_FILE:
            raise ValueError("HTMLConverter only supports one-doc-per-file mode")

        markdown_content = self._convert_to_markdown(elements)

        # Clean up excessive newlines
        markdown_content = re.sub(r"\n{4,}", "\n\n", markdown_content)
        markdown_content = markdown_content.strip()

        metadata = copy.deepcopy(metadata)
        metadata["file_path"] = filepath

        return [Document(content=markdown_content, metadata=metadata)]

    def _convert_to_markdown(self, element: lxml_html.HtmlElement) -> str:
        """
        Convert HTML element to Markdown.

        Args:
            element: lxml HTML element

        Returns:
            Markdown string
        """
        markdown = MarkdownConverter()
        return markdown.convert_element(element)


class MarkdownConverter:
    """Helper class to convert HTML elements to Markdown format"""

    def __init__(self):
        self.tag_handlers = {
            "h1": self.handle_heading,
            "h2": self.handle_heading,
            "h3": self.handle_heading,
            "h4": self.handle_heading,
            "h5": self.handle_heading,
            "h6": self.handle_heading,
            "p": self.handle_paragraph,
            "a": self.handle_link,
            "img": self.handle_image,
            "strong": self.handle_strong,
            "b": self.handle_strong,
            "em": self.handle_emphasis,
            "i": self.handle_emphasis,
            "code": self.handle_code,
            "pre": self.handle_preformatted,
            "ul": self.handle_unordered_list,
            "ol": self.handle_ordered_list,
            "li": self.handle_list_item,
            "br": self.handle_linebreak,
            "hr": self.handle_horizontal_rule,
            "table": self.handle_table,
            "tr": self.handle_table_row,
            "th": self.handle_table_cell,
            "td": self.handle_table_cell,
            "blockquote": self.handle_blockquote,
            "div": self.handle_block,
            "span": self.handle_inline,
        }

        self.list_state = []
        self.in_table = False
        self.table_headers = []
        self.table_rows = []
        self.current_row = []

    def convert_element(self, element: lxml_html.HtmlElement | str, parent_tag=None) -> str:
        """Convert an HTML element to Markdown recursively"""
        if element is None:
            return ""

        if isinstance(element, str) or element.tag == "text":
            text = element if isinstance(element, str) else element.text_content().strip()
            if text and parent_tag not in ["pre", "code"]:
                text = " ".join(text.split())
            return text

        tag = element.tag

        if tag is etree.Comment or tag in ["script", "style"]:
            return ""

        if tag == "html" or tag == "body" or tag == "document":
            return self.handle_document(element)

        if tag in self.tag_handlers:
            return self.tag_handlers[tag](element)

        return self.process_children(element)

    def process_children(self, element: lxml_html.HtmlElement, add_linebreaks=False) -> str:
        """Process all children of an element and join their markdown"""
        result = []

        if element.text and element.text.strip():
            result.append(element.text)

        for child in element:
            child_md = self.convert_element(child)
            if child_md:
                result.append(child_md)
            if child.tail and child.tail.strip():
                result.append(child.tail)

        separator = "\n" if add_linebreaks else ""
        return separator.join([r for r in result if r and r.strip()])

    def handle_document(self, element: lxml_html.HtmlElement) -> str:
        """Handle document element (html or body)"""
        return self.process_children(element, True)

    def handle_heading(self, element: lxml_html.HtmlElement) -> str:
        """Handle heading elements (h1-h6)"""
        level = int(element.tag[1])
        content = element.text_content().strip()
        return f"\n\n{'#' * level} {content}\n\n"

    def handle_paragraph(self, element: lxml_html.HtmlElement) -> str:
        """Handle paragraph elements"""
        content = element.text or ""

        for child in element:
            if child.tag in self.tag_handlers:
                content += self.tag_handlers[child.tag](child)
            else:
                content += self.process_children(child)

            if child.tail:
                content += child.tail

        if not content.strip():
            return ""

        return f"\n\n{content}\n\n"

    def handle_link(self, element: lxml_html.HtmlElement) -> str:
        """Handle anchor elements"""
        href = element.get("href", "")
        content = element.text_content() or href

        if href.startswith("#"):
            anchor = href[1:]
            return f"[{content}](#{anchor})"

        return f"[{content}]({href})"

    def handle_image(self, element: lxml_html.HtmlElement) -> str:
        """Handle image elements"""
        src = element.get("src", "")
        alt = element.get("alt", "")
        title = element.get("title", "")
        if title:
            return f'![{alt}]({src} "{title}")'
        return f"![{alt}]({src})"

    def handle_strong(self, element: lxml_html.HtmlElement) -> str:
        """Handle strong/bold elements"""
        content = element.text_content().strip()
        return f"**{content}**"

    def handle_emphasis(self, element: lxml_html.HtmlElement) -> str:
        """Handle emphasis/italic elements"""
        content = element.text_content().strip()
        return f"*{content}*"

    def handle_code(self, element: lxml_html.HtmlElement) -> str:
        """Handle inline code elements"""
        if element.getparent() is not None and element.getparent().tag == "pre":
            return element.text_content()

        content = element.text_content()
        return f"`{content}`"

    def handle_preformatted(self, element: lxml_html.HtmlElement) -> str:
        """Handle preformatted code blocks"""
        code_element = element.find("code")
        if code_element is not None:
            content = code_element.text_content()
            language = ""
            for cls in code_element.get("class", "").split():
                if cls.startswith("language-"):
                    language = cls[9:]
                    break
            return f"\n```{language}\n{content}\n```\n"

        content = element.text_content()
        return f"\n```\n{content}\n```\n"

    def handle_unordered_list(self, element: lxml_html.HtmlElement) -> str:
        """Handle unordered list elements"""
        self.list_state.append({"type": "ul", "index": 0})
        content = self.process_children(element, True)
        self.list_state.pop()
        return f"\n{content}\n"

    def handle_ordered_list(self, element: lxml_html.HtmlElement) -> str:
        """Handle ordered list elements"""
        start = element.get("start", "1")
        try:
            start = int(start)
        except ValueError:
            start = 1

        self.list_state.append({"type": "ol", "index": start - 1})
        content = self.process_children(element, True)
        self.list_state.pop()
        return f"\n{content}\n"

    def handle_list_item(self, element: lxml_html.HtmlElement) -> str:
        """Handle list item elements"""
        if self.list_state and self.list_state[-1]["type"] == "ol":
            self.list_state[-1]["index"] += 1

        indent = "  " * (len(self.list_state) - 1)
        if self.list_state and self.list_state[-1]["type"] == "ul":
            bullet = "*"
        else:
            bullet = f"{self.list_state[-1]['index']}."

        content = element.text or ""

        for child in element:
            if child.tag in self.tag_handlers:
                content += self.tag_handlers[child.tag](child)
            else:
                content += self.process_children(child)

            if child.tail:
                content += child.tail

        return f"{indent}{bullet} {content}"

    def handle_linebreak(self, element: lxml_html.HtmlElement) -> str:
        """Handle line break elements"""
        return "\n"

    def handle_horizontal_rule(self, element: lxml_html.HtmlElement) -> str:
        """Handle horizontal rule elements"""
        return "\n\n---\n\n"

    def handle_table(self, element: lxml_html.HtmlElement) -> str:
        """Handle table elements"""
        self.in_table = True
        self.table_headers = []
        self.table_rows = []

        self.process_children(element)

        if not self.table_headers and self.table_rows:
            self.table_headers = self.table_rows[0]
            self.table_rows = self.table_rows[1:]

        if not self.table_headers:
            self.in_table = False
            return ""

        result = []
        result.append("| " + " | ".join(self.table_headers) + " |")
        result.append("| " + " | ".join(["---"] * len(self.table_headers)) + " |")

        for row in self.table_rows:
            padded_row = row + [""] * (len(self.table_headers) - len(row))
            result.append("| " + " | ".join(padded_row) + " |")

        self.in_table = False
        return "\n\n" + "\n".join(result) + "\n\n"

    def handle_table_row(self, element: lxml_html.HtmlElement) -> str:
        """Handle table row elements"""
        if not self.in_table:
            return ""

        self.current_row = []
        self.process_children(element)

        if element.findall("th"):
            self.table_headers = self.current_row
        else:
            self.table_rows.append(self.current_row)

        return ""

    def handle_table_cell(self, element: lxml_html.HtmlElement) -> str:
        """Handle table cell elements"""
        if not self.in_table:
            return ""

        content = element.text_content()
        self.current_row.append(content.strip())
        return ""

    def handle_blockquote(self, element: lxml_html.HtmlElement) -> str:
        """Handle blockquote elements"""
        content = element.text_content()
        lines = content.split("\n")
        quoted_lines = [f"> {line}" if line.strip() else ">" for line in lines]
        return "\n\n" + "\n".join(quoted_lines) + "\n\n"

    def handle_block(self, element: lxml_html.HtmlElement) -> str:
        """Handle block-level elements like div"""
        content = self.process_children(element)
        return f"\n\n{content}\n\n"

    def handle_inline(self, element: lxml_html.HtmlElement) -> str:
        """Handle inline elements like span"""
        return self.process_children(element)
