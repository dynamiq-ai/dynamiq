import base64
import io

import pytest

from dynamiq.prompts.prompts import (
    MessageRole,
    Prompt,
    VisionMessage,
    VisionMessageImageContent,
    VisionMessageImageURL,
    VisionMessageType,
)


@pytest.fixture
def sample_png_bytes():
    """
    Returns bytes for a very small valid PNG image (1x1 pixel).
    This base64 string is properly padded and will decode without error.
    """
    # "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAAC0lEQVR42mNk"
    # "YGBgAAAABAABJzQnCgAAAABJRU5ErkJggg=="
    # is just a 1x1 px transparent PNG.
    data = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAAC0lEQVR42mNk" "YGBgAAAABAABJzQnCgAAAABJRU5ErkJggg=="
    return base64.b64decode(data)


def test_vision_prompt_with_bytesio(sample_png_bytes):
    """
    Tests that a BytesIO image is recognized and transformed into a proper
    base64-encoded data URL in the formatted vision message.
    """
    # Wrap our PNG bytes in a BytesIO object.
    image_bytes_io = io.BytesIO(sample_png_bytes)

    # Create a Prompt that expects an image URL with a Jinja placeholder {{ image }}.
    prompt = Prompt(
        id="test_vision",
        messages=[
            VisionMessage(
                content=[VisionMessageImageContent(image_url=VisionMessageImageURL(url="{{ image }}"))],
                role=MessageRole.USER,
            )
        ],
    )

    # Format the messages with the image parameter pointing to our BytesIO.
    out_messages = prompt.format_messages(image=image_bytes_io)

    # We expect exactly one message in the output.
    assert len(out_messages) == 1
    msg = out_messages[0]

    # Ensure the role is "user".
    assert msg["role"] == MessageRole.USER

    # Ensure the content is a list with one vision item.
    assert isinstance(msg["content"], list)
    assert len(msg["content"]) == 1

    # Extract the image content dictionary.
    image_content = msg["content"][0]
    assert image_content["type"] == VisionMessageType.IMAGE_URL

    # Check that the URL has a base64-encoded PNG.
    image_url = image_content["image_url"]["url"]
    assert image_url.startswith("data:image/png;base64,")

    # Optional: decode and check that it starts with the PNG header bytes.
    encoded_part = image_url.split("base64,")[-1]
    decoded_part = base64.b64decode(encoded_part)
    assert decoded_part.startswith(b"\x89PNG\r\n\x1a\n")


def test_vision_prompt_with_regular_url():
    """
    Tests that if the user passes a normal string URL (no bytes), it
    remains unchanged in the formatted messages.
    """
    prompt = Prompt(
        id="test_vision_url",
        messages=[
            VisionMessage(content=[VisionMessageImageContent(image_url=VisionMessageImageURL(url="{{ image_url }}"))])
        ],
    )

    test_image_url = "https://example.com/sample.png"
    out_messages = prompt.format_messages(image_url=test_image_url)
    msg = out_messages[0]

    assert len(out_messages) == 1
    assert msg["role"] == MessageRole.USER
    assert len(msg["content"]) == 1

    image_content = msg["content"][0]
    assert image_content["type"] == VisionMessageType.IMAGE_URL
    assert image_content["image_url"]["url"] == test_image_url


def test_vision_prompt_unsupported_type():
    """
    Tests that using an unsupported type for the image parameter
    (e.g., an integer) raises a ValueError.
    """
    prompt = Prompt(
        id="test_unsupported",
        messages=[
            VisionMessage(content=[VisionMessageImageContent(image_url=VisionMessageImageURL(url="{{ image }}"))])
        ],
    )

    with pytest.raises(ValueError):
        prompt.format_messages(image=12345)
