import io
from pathlib import Path

from dynamiq.prompts import Prompt, VisionMessage, VisionMessageImageContent
from examples.llm_setup import setup_llm

llm = setup_llm()

file_path = "image_file_path"

if __name__ == "__main__":

    file_path_obj = Path(file_path).resolve()
    if not file_path_obj.exists():
        raise FileNotFoundError(f"The file {file_path} does not exist.")
    if not file_path_obj.is_file():
        raise OSError(f"The path {file_path} is not a valid file.")

    with file_path_obj.open("rb") as file:
        image_bytes = file.read()

    image_bytes_io = io.BytesIO(image_bytes)

    prompt = Prompt(
        id="1",
        messages=[VisionMessage(content=[VisionMessageImageContent(image_url={"url": "{{image}}"})])],
    )

    result = llm.run(input_data={"image": image_bytes_io}, prompt=prompt)
    print(result)
