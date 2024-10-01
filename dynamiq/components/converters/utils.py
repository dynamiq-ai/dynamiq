from io import BytesIO

import filetype

from dynamiq.utils.utils import generate_uuid


def get_filename_for_bytesio(file: BytesIO) -> str:
    """
    Get a filepath for a BytesIO object.

    Args:
        file (BytesIO): The BytesIO object.

    Returns:
        str: A filename for the BytesIO object.

    Raises:
        ValueError: If the file extension couldn't be guessed.
    """
    filename = getattr(file, "name", None)
    if filename is None:
        file_extension = filetype.guess_extension(file)
        if file_extension:
            filename = f"{generate_uuid()}.{file_extension}"
        else:
            raise ValueError(
                "Unable to determine file extension. BytesIO object lacks name and "
                "extension couldn't be guessed."
            )
    return filename
