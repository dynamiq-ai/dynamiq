from typing import Any

import mistune
from bs4 import BeautifulSoup, Tag


def markdown_to_requests(md_text: str) -> list[dict[str, Any]]:
    """
    Convert a Markdown string into a list of Google Docs API requests to insert text and apply styles.

    Args:
        md_text (str): Input Markdown text.

    Returns:
        list[dict[str, Any]]: A list of Google Docs API requests to recreate the markdown formatting.
    """
    html = mistune.create_markdown()(md_text)
    soup = BeautifulSoup(html, "html.parser")

    requests = []
    index = 1

    for elem in soup.body or soup:
        if isinstance(elem, Tag):
            index = process_block(elem, requests, index)

    return requests


def process_inline_styles(text_block: Tag, base_index: int, requests: list[dict[str, Any]]) -> None:
    """
    Process inline styles (bold, italic, links) within a given text block.

    Args:
        text_block (Tag): BeautifulSoup Tag containing the text and inline styles.
        base_index (int): The starting index of this block's text in the overall document.
        requests (list): The list of previous requests
    """
    offset = 0
    for tag in text_block.find_all(["strong", "b", "em", "i", "a"]):
        tag_text = tag.get_text()
        rel_start = text_block.get_text().find(tag_text, offset)
        rel_end = rel_start + len(tag_text)

        style = {}
        if tag.name in ["strong", "b"]:
            style["bold"] = True
        if tag.name in ["em", "i"]:
            style["italic"] = True

        if style:
            requests.append(
                {
                    "updateTextStyle": {
                        "range": {"startIndex": base_index + rel_start, "endIndex": base_index + rel_end},
                        "textStyle": style,
                        "fields": ",".join(style.keys()),
                    }
                }
            )

        if tag.name == "a" and tag.has_attr("href"):
            requests.append(
                {
                    "updateTextStyle": {
                        "range": {"startIndex": base_index + rel_start, "endIndex": base_index + rel_end},
                        "textStyle": {"link": {"url": tag["href"]}},
                        "fields": "link",
                    }
                }
            )

        offset = rel_end


def process_block(
    block: Tag,
    requests: list[dict[str, Any]],
    index: int,
    indent: int = 0,
    list_type: str | None = None,
) -> int:
    """
    Recursively process a block-level HTML tag.

    Args:
        block (Tag): BeautifulSoup Tag representing a block element.
        requests (list): The list of previous requests.
        index (int): The current character index in the document.
        indent Optional[int]: Current list indentation level.
        list_type Optional[str | None]: List type if inside a list ('ul' or 'ol').

    Returns:
        int: Updated character index after processing this block.
    """
    # Headings and Paragraphs
    if block.name and block.name.startswith("h") or block.name == "p":
        text = block.get_text() + "\n"
        start = index
        end = start + len(text)

        requests.append({"insertText": {"location": {"index": index}, "text": text}})
        process_inline_styles(block, start, requests)
        index = end

        if block.name and block.name.startswith("h"):
            level = int(block.name[1])
            requests.append(
                {
                    "updateParagraphStyle": {
                        "range": {"startIndex": start, "endIndex": end},
                        "paragraphStyle": {"namedStyleType": f"HEADING_{min(level, 6)}"},
                        "fields": "namedStyleType",
                    }
                }
            )

    # Lists (ul/ol)
    elif block.name in ["ul", "ol"]:
        for li in block.find_all("li", recursive=False):
            index = process_block(li, requests, index, indent=indent + 1, list_type=block.name)

    # List Item
    elif block.name == "li":
        inline_container = BeautifulSoup(features="html.parser").new_tag("span")
        for content in block.contents:
            if isinstance(content, Tag) and content.name in ["ul", "ol"]:
                break
            inline_container.append(content)

        first_text = inline_container.get_text().strip() + "\n"
        start = index
        end = start + len(first_text)

        requests.append({"insertText": {"location": {"index": index}, "text": first_text}})
        process_inline_styles(inline_container, start, requests)
        index = end

        bullet_preset = "NUMBERED_DECIMAL_ALPHA_ROMAN" if list_type == "ol" else "BULLET_DISC_CIRCLE_SQUARE"
        requests.append(
            {"createParagraphBullets": {"range": {"startIndex": start, "endIndex": end}, "bulletPreset": bullet_preset}}
        )

        for child in block.children:
            if isinstance(child, Tag) and child.name in ["ul", "ol"]:
                index = process_block(child, requests, index, indent=indent + 1, list_type=child.name)

    # Fallback for other styles
    elif block.get_text(strip=True):
        text = block.get_text().strip() + "\n"
        start = index
        end = start + len(text)
        requests.append({"insertText": {"location": {"index": index}, "text": text}})
        process_inline_styles(block, start, requests)
        index = end

    return index
