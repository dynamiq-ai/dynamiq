import os
from io import BytesIO

from dynamiq import ROOT_PATH
from dynamiq.utils import generate_uuid


def list_data_folder_paths(folder_path=os.path.join(os.path.dirname(ROOT_PATH), "examples/data/")) -> list[str]:
    file_names = os.listdir(folder_path)
    file_paths = [os.path.join(folder_path, file_name) for file_name in file_names]

    return file_paths


def read_bytes_io_files(file_paths: list[str]):
    files = []
    metadata = []

    # Read files into BytesIO objects
    for path in file_paths:
        with open(path, "rb") as upload_file:
            bytes_io = BytesIO(upload_file.read())
            bytes_io.name = upload_file.name
            files.append(bytes_io)

            file_id = generate_uuid()
            metadata.append({"file_id": file_id})

    return {"files": files, "metadata": metadata}
