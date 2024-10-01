import ast

from dynamiq.nodes.validators.base import BaseValidator


class ValidPython(BaseValidator):
    """
    Class that provides functionality to check if a value matches a basic Python code standards.
    """

    def validate(self, content: str):
        """
        Validates the provided Python code to determine if it is syntactically correct.

        Args:
            content (str): The Python code to validate.

        Raises:
            ValueError: Raised if the provided value is not syntactically correct Python code.
        """
        try:
            ast.parse(content)
        except SyntaxError as e:
            raise ValueError(
                f"Value is not valid python code. Value: '{content}'. Error details: {e.msg}"
            )
