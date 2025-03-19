import base64
import io

import filetype

from dynamiq.prompts import (
    Message,
    MessageRole,
    VisionMessage,
    VisionMessageImageContent,
    VisionMessageImageURL,
    VisionMessageTextContent,
)
from dynamiq.utils.logger import logger


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
