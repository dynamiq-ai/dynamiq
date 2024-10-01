from io import BytesIO

from dynamiq.nodes import llms
from dynamiq.nodes.converters.llm_text_extractor import LLMPDFConverter

# Please use your own file path
PDF_FILE_PATH = "layout-parser-paper.pdf"


def main():
    # Initialize the LLM
    llm = llms.OpenAI(
        name="OpenAI Vision",
        model="gpt-4o",
        postponned_init=True,
    )

    # Initialize the PDF text extractor
    converter = LLMPDFConverter(llm=llm)

    # Example file paths
    file_paths = [PDF_FILE_PATH]
    files = []

    # Read files into BytesIO objects
    for path in file_paths:
        with open(path, "rb") as upload_file:
            bytes_io = BytesIO(upload_file.read())
            files.append(bytes_io)

    # Execute the extractor
    output = converter.execute(
        input_data={
            "file_paths": file_paths,
            "files": files,
        }
    )

    # Print the output
    print(output)


if __name__ == "__main__":
    main()
