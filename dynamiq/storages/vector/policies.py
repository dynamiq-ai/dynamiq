from enum import Enum


class DuplicatePolicy(str, Enum):
    """
    Enumeration of policies for handling duplicate items.

    Attributes:
        NONE (str): No specific policy for handling duplicates.
        SKIP (str): Skip duplicate items without modifying existing ones.
        OVERWRITE (str): Overwrite existing items with duplicate entries.
        FAIL (str): Raise an error when encountering duplicate items.
    """

    NONE = "none"
    SKIP = "skip"
    OVERWRITE = "overwrite"
    FAIL = "fail"
