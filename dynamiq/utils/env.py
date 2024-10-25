import os
from typing import Any

from dynamiq.utils.logger import logger


def get_env_var(var_name: str, default_value: Any = None):
    """Retrieves the value of an environment variable.

    This function attempts to retrieve the value of the specified environment variable. If the
    variable is not found and no default value is provided, it raises a ValueError.

    Args:
        var_name (str): The name of the environment variable to retrieve.
        default_value (str, optional): The default value to return if the environment variable
            is not found. Defaults to None.

    Returns:
        str: The value of the environment variable.

    Raises:
        ValueError: If the environment variable is not found and no default value is provided.

    Examples:
        >>> get_env_var("HOME")
        '/home/user'
        >>> get_env_var("NONEXISTENT_VAR", "default")
        'default'
        >>> get_env_var("NONEXISTENT_VAR")
        Traceback (most recent call last):
            ...
        ValueError: Environment variable 'NONEXISTENT_VAR' not found.
    """
    value = os.environ.get(var_name, default_value)

    if value is None:
        logger.warning(f"Environment variable '{var_name}' not found")

    return value
