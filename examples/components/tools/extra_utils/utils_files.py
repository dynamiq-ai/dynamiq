import io


def read_file_as_bytesio(file_path: str, filename: str = None, description: str = None) -> io.BytesIO:
    """
    Reads the content of a file and returns it as a BytesIO object with custom attributes for filename and description.

    Args:
        file_path (str): The path to the file.
        filename (str, optional): Custom filename for the BytesIO object.
        description (str, optional): Custom description for the BytesIO object.

    Returns:
        io.BytesIO: The file content in a BytesIO object with custom attributes.
    """
    with open(file_path, "rb") as f:
        file_content = f.read()

    file_io = io.BytesIO(file_content)

    file_io.name = filename if filename else "uploaded_file"
    file_io.description = description if description else "No description provided"

    return file_io
