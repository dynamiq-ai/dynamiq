from typing import Any

from dynamiq.nodes.validators.base import BaseValidator


class ValidChoices(BaseValidator):
    """
    Class that provides functionality to check if the provided value is within the list of valid choices.

    Args:
        choices(List[Any]): A list of values representing the acceptable choices.

    """

    choices: list[Any] = None

    def validate(self, content: Any):
        """
        Validates if the provided value is among the acceptable choices.

        Args:
            content(Any): The value to validate.

        Raises:
            ValueError: If the provided value is not in valid choices.
        """
        if isinstance(content, str):
            content = content.strip()

        if content not in self.choices:
            raise ValueError(
                f"Value is not in valid choices. Value: '{content}'. Choices: '{self.choices}'."
            )
