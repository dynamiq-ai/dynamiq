import json

from dynamiq.nodes.validators.base import BaseValidator


class ValidJSON(BaseValidator):
    """
    Class that provides functionality to check if a value matches a basic JSON structure.
    """

    def validate(self, content: str | dict):
        """
        Validates if the provided string is a properly formatted JSON.

        Args:
            content(str): The value to check.

        Raises:
            ValueError: If the value is not a properly formatted JSON.

        """
        try:
            if not isinstance(content, str):
                content = json.dumps(content)

            json.loads(content)
        except (json.decoder.JSONDecodeError, TypeError) as error:
            raise ValueError(
                f"Value is not valid JSON. Value: '{content}'. Error details: {str(error)}"
            )
