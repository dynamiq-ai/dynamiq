import os
from io import BytesIO

from dynamiq import Workflow
from dynamiq.connections.managers import get_connection_manager

DOCX_FILE_PATH = "../data/file.docx"
PPTX_FILE_PATH = "../data/file.pptx"
TXT_FILE_PATH = "../data/file.txt"
MARKDOWN_FILE_PATH = "../data/file.md"
HTML_FILE_PATH = "../data/file.html"
PDF_FILE_PATH = "../data/file.pdf"

files = [
    DOCX_FILE_PATH,
    PPTX_FILE_PATH,
    TXT_FILE_PATH,
    MARKDOWN_FILE_PATH,
    HTML_FILE_PATH,
    PDF_FILE_PATH,
]

if __name__ == "__main__":
    yaml_file_path = os.path.join(os.path.dirname(__file__), "config_default.yaml")

    with get_connection_manager() as cm:
        wf = Workflow.from_yaml_file(file_path=yaml_file_path, connection_manager=cm, init_components=True)

        files_data = []
        for file_path in files:
            with open(file_path, "rb") as f:
                file_data = BytesIO(f.read())
                file_data.name = os.path.basename(file_path)
                files_data.append(file_data)

        result = wf.run(input_data={"files": files_data})
        print(result)
