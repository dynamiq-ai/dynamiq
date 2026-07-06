import base64
import io

from dynamiq.nodes.agents.utils import create_message_from_input, is_video_file
from dynamiq.prompts import Message, VisionMessage, VisionMessageFileContent, VisionMessageImageContent

# Minimal bytes recognized by `filetype` as video/mp4 (a bare ftyp box, no real media).
MP4_BYTES = b"\x00\x00\x00\x18ftypmp42\x00\x00\x00\x00mp42isom" + b"\x00" * 20

# 1x1 px transparent PNG, for asserting videos and images aren't cross-detected.
PNG_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAAC0lEQVR42mNk" "YGBgAAAABAABJzQnCgAAAABJRU5ErkJggg=="
)


class TestIsVideoFile:
    def test_detects_mp4_bytes(self):
        assert is_video_file(MP4_BYTES) is True

    def test_detects_mp4_bytesio(self):
        assert is_video_file(io.BytesIO(MP4_BYTES)) is True

    def test_bytesio_position_is_preserved(self):
        buf = io.BytesIO(MP4_BYTES)
        buf.seek(5)
        is_video_file(buf)
        assert buf.tell() == 5

    def test_rejects_image_bytes(self):
        assert is_video_file(PNG_BYTES) is False

    def test_rejects_non_file_input(self):
        assert is_video_file("not a file") is False
        assert is_video_file(None) is False


class TestCreateMessageFromInputVideo:
    def test_text_only_returns_plain_message(self):
        message = create_message_from_input({"input": "hello"})
        assert isinstance(message, Message)
        assert message.content == "hello"

    def test_video_bytes_produce_file_content_block(self):
        message = create_message_from_input({"input": "describe this clip", "videos": [MP4_BYTES]})

        assert isinstance(message, VisionMessage)
        assert len(message.content) == 2  # text + video

        video_block = message.content[1]
        assert isinstance(video_block, VisionMessageFileContent)
        assert video_block.file.file_data.startswith("data:video/mp4;base64,")

    def test_video_url_passed_through_unchanged(self):
        message = create_message_from_input({"videos": ["https://example.com/clip.mp4"]})

        video_block = message.content[0]
        assert isinstance(video_block, VisionMessageFileContent)
        assert video_block.file.file_data == "https://example.com/clip.mp4"

    def test_video_detected_among_generic_files(self):
        """A video passed via the generic `files` field (not `videos`) is still detected
        by content sniffing and routed into a video content block."""
        message = create_message_from_input({"files": [MP4_BYTES]})

        assert isinstance(message, VisionMessage)
        assert isinstance(message.content[0], VisionMessageFileContent)

    def test_mixed_images_and_videos(self):
        message = create_message_from_input({"images": [PNG_BYTES], "videos": [MP4_BYTES]})

        block_types = [type(c) for c in message.content]
        assert VisionMessageImageContent in block_types
        assert VisionMessageFileContent in block_types