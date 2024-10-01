import enum
import re

from dynamiq.nodes.validators.base import BaseValidator


class MatchType(str, enum.Enum):
    FULL_MATCH = "fullmatch"
    SEARCH = "search"


class RegexMatch(BaseValidator):
    """
    Validates that a value matches a regular expression.

    Args:
        regex: A regular expression pattern.
        match_type: Match type to check input value for a regex search or full-match option.
    """

    regex: str
    match_type: MatchType | None = MatchType.FULL_MATCH

    def validate(self, content: str):
        """
        Validates if the provided value matches the given regular expression pattern.

        Args:
            content (str): The value to validate.

        Raises:
            ValueError: If the provided value does not match the given pattern.
        """
        compiled_pattern = re.compile(self.regex)
        match_method = getattr(compiled_pattern, self.match_type)
        if not match_method(content):
            raise ValueError(
                f"Value does not match the valid pattern. Value: '{content}'. Pattern: '{self.regex}'",
            )
