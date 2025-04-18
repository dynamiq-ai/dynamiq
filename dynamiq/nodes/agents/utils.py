import base64
import io
import json
import re
from typing import Any, Sequence

import filetype
from lxml import etree as LET  # nosec: B410

from dynamiq.nodes.agents.exceptions import JSONParsingError, ParsingError, TagNotFoundError, XMLParsingError
from dynamiq.prompts import (
    Message,
    MessageRole,
    VisionMessage,
    VisionMessageImageContent,
    VisionMessageImageURL,
    VisionMessageTextContent,
)
from dynamiq.utils.logger import logger

TOOL_MAX_TOKENS = 64000


class XMLParser:
    """
    Utility class for parsing XML-like output, often generated by LLMs.
    Prioritizes lxml for robustness, with fallbacks for common issues.
    """

    @staticmethod
    def _clean_content(text: str) -> str:
        """
        Cleans the input string to remove common LLM artifacts and isolate XML.
        - Removes markdown code fences (```json, ```, etc.).
        - Strips leading/trailing whitespace.
        - Attempts to extract the first well-formed XML block if surrounded by text.
        """
        if not isinstance(text, str):
            return ""

        cleaned = re.sub(r"^```(?:[a-zA-Z]+\s*)?|```$", "", text.strip(), flags=re.MULTILINE)
        cleaned = cleaned.strip()

        xml_match = re.search(r"<(\w+)\b[^>]*>.*?</\1>", cleaned, re.DOTALL)
        if xml_match:
            cleaned = xml_match.group(0)

        return cleaned

    @staticmethod
    def _parse_with_lxml(cleaned_text: str) -> LET._Element | None:
        """
        Attempts to parse the cleaned text using lxml with recovery.
        """
        if not cleaned_text:
            return None
        try:
            parser = LET.XMLParser(recover=True, encoding="utf-8")
            root = LET.fromstring(cleaned_text.encode("utf-8"), parser=parser)  # nosec: B320
            return root
        except LET.XMLSyntaxError as e:
            logger.warning(f"XMLParser: lxml parsing failed even with recovery: {e}. Content: {cleaned_text[:200]}...")
            return None
        except Exception as e:
            logger.error(f"XMLParser: Unexpected error during lxml parsing: {e}. Content: {cleaned_text[:200]}...")
            return None

    @staticmethod
    def _extract_data_lxml(
        root: LET._Element, required_tags: Sequence[str], optional_tags: Sequence[str] = None
    ) -> dict[str, str]:
        """
        Extracts text content from specified tags using XPath.
        Handles cases where the provided 'root' might be a child element
        by navigating up to the parent if necessary.
        Distinguishes between missing tags and found-but-empty tags.
        """
        data = {}
        optional_tags = optional_tags or []
        all_tags = list(required_tags) + list(optional_tags)

        for tag in all_tags:
            tag_content = None
            element_found = False
            elements = root.xpath(f".//{tag}")
            if elements:
                element_found = True
                for elem in elements:
                    text = "".join(elem.itertext()).strip()
                    if text:
                        tag_content = text
                        break

            if not element_found:
                try:
                    if root.getparent() is not None:
                        parent_elements = root.xpath(f"../{tag}")
                        if parent_elements:
                            element_found = True
                            for elem in parent_elements:
                                text_parent_child = "".join(elem.itertext()).strip()
                                if text_parent_child:
                                    tag_content = text_parent_child
                                    break
                except AttributeError:
                    logger.debug(
                        f"XMLParser: Root element '{root.tag}' has no parent, cannot search siblings via parent."
                    )
                    pass
                except Exception as e:
                    logger.warning(f"XMLParser: Error during parent navigation XPath for tag '{tag}': {e}")
                    pass

            if tag_content is not None:
                data[tag] = tag_content
            elif element_found and tag in required_tags:
                raise TagNotFoundError(f"Required tag <{tag}> found but contains no text content.")
            elif not element_found and tag in required_tags:
                raise TagNotFoundError(
                    f"Required tag <{tag}> not found in the XML structure "
                    f"relative to the parsed root element ('{root.tag}') or its parent."
                )

        missing_required_after_all = [tag for tag in required_tags if tag not in data]
        if missing_required_after_all:
            raise TagNotFoundError(f"Required tags missing after extraction: {', '.join(missing_required_after_all)}")

        return data

    @staticmethod
    def _parse_json_fields(data: dict[str, str], json_fields: Sequence[str]) -> dict[str, Any]:
        """
        Parses specified fields in the data dictionary as JSON.
        """
        parsed_data = data.copy()
        for field in json_fields:
            if field in parsed_data:
                try:
                    json_string = re.sub(r"^```(?:json)?\s*|```$", "", parsed_data[field].strip())
                    parsed_data[field] = json.loads(json_string)
                except json.JSONDecodeError as e:
                    error_message = (
                        f"Failed to parse JSON content for field '{field}'. "
                        f"Error: {e}. Original content: '{parsed_data[field][:100]}...'"
                    )
                    guidance = (
                        " Ensure the value is valid JSON with double quotes for keys and strings, "
                        'and proper escaping for special characters (e.g., \\n for newlines, \\" for quotes).'
                    )
                    raise JSONParsingError(error_message + guidance)
                except Exception as e:
                    raise JSONParsingError(f"Unexpected error parsing JSON for field '{field}': {e}")
        return parsed_data

    @staticmethod
    def parse(
        text: str,
        required_tags: Sequence[str],
        optional_tags: Sequence[str] = None,
        json_fields: Sequence[str] = None,
        attempt_wrap: bool = True,
    ) -> dict[str, Any]:
        """
        Parses XML-like text to extract structured data.

        Args:
            text (str): The raw text input potentially containing XML.
            required_tags (Sequence[str]): A list/tuple of tag names that MUST be present.
            optional_tags (Sequence[str], optional): A list/tuple of optional tag names. Defaults to None.
            json_fields (Sequence[str], optional): A list/tuple of fields whose content should be parsed as JSON.
            attempt_wrap (bool): If initial parsing fails, try wrapping the content in <root> and parse again.

        Returns:
            dict[str, Any]: A dictionary containing the extracted data.
                           Values for json_fields will be parsed JSON objects/primitives.

        Raises:
            XMLParsingError: If the text cannot be parsed as XML even with recovery/wrapping.
            TagNotFoundError: If any of the required_tags are not found or are empty.
            JSONParsingError: If any field listed in json_fields contains invalid JSON.
            ParsingError: For other generic parsing issues.
        """
        optional_tags = optional_tags or []
        json_fields = json_fields or []

        cleaned_text = XMLParser._clean_content(text)
        if not cleaned_text:
            if required_tags:
                raise ParsingError("Input text is empty or became empty after cleaning.")
            else:
                return {}

        root = XMLParser._parse_with_lxml(cleaned_text)

        if root is None and attempt_wrap:
            logger.info("XMLParser: Initial lxml parse failed, attempting to wrap content in <root>...")
            wrapped_text = f"<root>{cleaned_text}</root>"
            root = XMLParser._parse_with_lxml(wrapped_text)
        if root is None:
            raise XMLParsingError(
                f"Failed to parse content as XML even after cleaning "
                f"and wrapping attempts. Content: {cleaned_text[:200]}..."
            )

        try:
            extracted_data = XMLParser._extract_data_lxml(root, required_tags, optional_tags)
        except TagNotFoundError as e:
            raise e
        except Exception as e:
            raise ParsingError(f"Error extracting data using XPath: {e}")

        if json_fields:
            try:
                final_data = XMLParser._parse_json_fields(extracted_data, json_fields)
            except JSONParsingError as e:
                raise e
            except Exception as e:
                # Catch potential errors during JSON parsing
                raise ParsingError(f"Unexpected error during JSON field processing: {e}")
        else:
            final_data = extracted_data

        return final_data

    @staticmethod
    def extract_first_tag_lxml(text: str, tags: Sequence[str]) -> str | None:
        """
        Extracts the text content of the first tag found from the list using lxml.
        Useful for simple cases like extracting just the final answer.
        """
        cleaned_text = XMLParser._clean_content(text)
        if not cleaned_text:
            return None

        root = XMLParser._parse_with_lxml(cleaned_text)

        if root is None:
            wrapped_text = f"<root>{cleaned_text}</root>"
            root = XMLParser._parse_with_lxml(wrapped_text)

        if root is None:
            logger.warning(f"XMLParser: extract_first_tag_lxml failed to parse: {cleaned_text[:200]}...")
            return None

        for tag in tags:
            elements = root.xpath(f".//{tag}")
            if elements:
                for elem in elements:
                    content = "".join(elem.itertext()).strip()
                    if content:
                        return content
        return None

    @staticmethod
    def extract_first_tag_regex(text: str, tags: Sequence[str]) -> str | None:
        """
        Fallback method: Extracts the text content of the first tag found using regex.
        Less reliable than lxml, use only when lxml fails completely.
        """
        if not isinstance(text, str):
            return None

        for tag in tags:
            match = re.search(f"<{tag}\\b[^>]*>(.*?)</{tag}>", text, re.DOTALL | re.IGNORECASE)
            if match:
                content = match.group(1).strip()
                if content:
                    return content
        return None


def create_message_from_input(input_data: dict) -> Message | VisionMessage:
    """
    Create appropriate message type based on input data,
    automatically detecting and handling images from either images or files fields

    Args:
        input_data (dict): Input data dictionary containing:
            - 'input': Text input string
            - 'images': List of image data (URLs, bytes, or BytesIO objects)
            - 'files': List of file data (bytes or BytesIO objects)

    Returns:
        Message or VisionMessage: Appropriate message type for the input
    """
    text_input = input_data.get("input", "")
    images = input_data.get("images", []) or []
    files = input_data.get("files", []) or []

    if not isinstance(images, list):
        images = [images]
    else:
        images = list(images)

    for file in files:
        if is_image_file(file):
            logger.debug(f"File detected as image, adding to vision processing: {getattr(file, 'name', 'unnamed')}")
            images.append(file)

    if not images:
        return Message(role=MessageRole.USER, content=text_input)

    content = []

    if text_input:
        content.append(VisionMessageTextContent(text=text_input))

    for image in images:
        try:
            if isinstance(image, str):
                if image.startswith(("http://", "https://", "data:")):
                    image_url = image
                else:
                    with open(image, "rb") as file:
                        image_bytes = file.read()
                        image_url = bytes_to_data_url(image_bytes)
            else:
                if isinstance(image, io.BytesIO):
                    image_bytes = image.getvalue()
                else:
                    image_bytes = image
                image_url = bytes_to_data_url(image_bytes)

            content.append(VisionMessageImageContent(image_url=VisionMessageImageURL(url=image_url)))
        except Exception as e:
            logger.error(f"Error processing image: {str(e)}")

    return VisionMessage(content=content, role=MessageRole.USER)


def is_image_file(file) -> bool:
    """
    Determine if a file is an image by examining its content

    Args:
        file: File-like object or bytes

    Returns:
        bool: True if the file is an image, False otherwise
    """
    try:
        if isinstance(file, io.BytesIO):
            pos = file.tell()
            file.seek(0)
            file_bytes = file.read(32)
            file.seek(pos)
        elif isinstance(file, bytes):
            file_bytes = file[:32]
        else:
            return False

        signatures = {
            b"\xff\xd8\xff": "jpg/jpeg",  # JPEG
            b"\x89PNG\r\n\x1a\n": "png",  # PNG
            b"GIF87a": "gif",  # GIF87a
            b"GIF89a": "gif",  # GIF89a
            b"RIFF": "webp",  # WebP
            b"MM\x00*": "tiff",  # TIFF (big endian)
            b"II*\x00": "tiff",  # TIFF (little endian)
            b"BM": "bmp",  # BMP
        }

        for sig, fmt in signatures.items():
            if file_bytes.startswith(sig):
                return True

        if isinstance(file, io.BytesIO):
            pos = file.tell()
            file.seek(0)
            mime = filetype.guess_mime(file.read(4096))
            file.seek(pos)
            return mime is not None and mime.startswith("image/")
        elif isinstance(file, bytes):
            mime = filetype.guess_mime(file)
            return mime is not None and mime.startswith("image/")

        return False
    except Exception as e:
        logger.error(f"Error checking if file is an image: {str(e)}")
        return False


def bytes_to_data_url(image_bytes: bytes) -> str:
    """
    Convert image bytes to a data URL

    Args:
        image_bytes (bytes): Raw image bytes

    Returns:
        str: Data URL string (format: data:image/jpeg;base64,...)
    """
    try:
        mime_type = filetype.guess_mime(image_bytes)
        if not mime_type:
            if image_bytes[:2] == b"\xff\xd8":
                mime_type = "image/jpeg"
            elif image_bytes[:8] == b"\x89PNG\r\n\x1a\n":
                mime_type = "image/png"
            elif image_bytes[:6] in (b"GIF87a", b"GIF89a"):
                mime_type = "image/gif"
            else:
                mime_type = "application/octet-stream"

        encoded = base64.b64encode(image_bytes).decode("utf-8")
        return f"data:{mime_type};base64,{encoded}"
    except Exception as e:
        logger.error(f"Error converting image to data URL: {str(e)}")
        raise ValueError(f"Failed to convert image to data URL: {str(e)}")


def process_tool_output_for_agent(content: Any, max_tokens: int = TOOL_MAX_TOKENS, truncate: bool = True) -> str:
    """
    Process tool output for agent consumption.

    This function converts various types of tool outputs into a string representation.
    It handles dictionaries (with or without a 'content' key), lists, tuples, and other
    types by converting them to a string. If the resulting string exceeds the maximum
    allowed length (calculated from max_tokens), it is truncated.

    Args:
        content: The output from tool execution, which can be of various types.
        max_tokens: Maximum allowed token count for the content. The effective character
            limit is computed as max_tokens * 4 (assuming ~4 characters per token).
        truncate: Whether to truncate the content if it exceeds the maximum length.

    Returns:
        A processed string suitable for agent consumption.
    """
    if not isinstance(content, str):
        if isinstance(content, dict):
            if "content" in content:
                inner_content = content["content"]
                content = inner_content if isinstance(inner_content, str) else json.dumps(inner_content, indent=2)
            else:
                content = json.dumps(content, indent=2)
        elif isinstance(content, (list, tuple)):
            content = "\n".join(str(item) for item in content)
        else:
            content = str(content)

    max_len_in_char: int = max_tokens * 4  # This assumes an average of 4 characters per token.

    if len(content) > max_len_in_char and truncate:
        half_length: int = (max_len_in_char - 100) // 2
        truncation_message: str = "\n...[Content truncated]...\n"
        content = content[:half_length] + truncation_message + content[-half_length:]

    return content


def extract_thought_from_intermediate_steps(intermediate_steps):
    """Extract thought process from the intermediate steps structure."""
    if not intermediate_steps:
        return None

    for step_key, step_value in intermediate_steps.items():
        if isinstance(step_value, dict) and "model_observation" in step_value:
            model_obs = step_value["model_observation"]

            if isinstance(model_obs, dict):
                if "initial" in model_obs:
                    initial = model_obs["initial"]

                    if initial.startswith("{") and '"thought"' in initial:
                        try:
                            json_data = json.loads(initial)
                            if "thought" in json_data:
                                return json_data["thought"]
                        except json.JSONDecodeError:
                            pass

                    if "<thought>" in initial:
                        thought_match = re.search(r"<thought>\s*(.*?)\s*</thought>", initial, re.DOTALL)
                        if thought_match:
                            return thought_match.group(1)

                    thought_match = re.search(r"Thought:\s*(.*?)(?:\n\n|\nAnswer:)", initial, re.DOTALL)
                    if thought_match:
                        return thought_match.group(1)

    return None
