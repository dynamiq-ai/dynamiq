import base64


class BaseCodec:
    """Abstract base class for encoding and decoding."""

    def encode(self, value: str) -> str:
        """Encode a string value.

        Args:
            value (str): The string to encode.

        Returns:
            str: The encoded string.
        """
        raise NotImplementedError

    def decode(self, value: str | bytes) -> str:
        """Decode a string or bytes value.

        Args:
            value (str | bytes): The value to decode.

        Returns:
            str: The decoded string.
        """
        raise NotImplementedError


class Base64Codec(BaseCodec):
    """Base64 encoding and decoding implementation."""

    def encode(self, value: str) -> str:
        """Encode a string using Base64.

        Args:
            value (str): The string to encode.

        Returns:
            str: The Base64 encoded string.
        """
        return base64.b64encode(value.encode()).decode()

    def decode(self, value: str | bytes) -> str:
        """Decode a Base64 encoded string or bytes.

        Args:
            value (str | bytes): The value to decode.

        Returns:
            str: The decoded string.
        """
        return base64.b64decode(value).decode()
