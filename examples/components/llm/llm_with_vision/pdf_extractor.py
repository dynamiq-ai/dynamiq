import base64
from io import BytesIO

from pdf2image import convert_from_path
from PIL import Image

from dynamiq.connections import OpenAI as OpenAIConnection
from dynamiq.nodes import llms
from dynamiq.prompts import (
    Prompt,
    VisionMessage,
    VisionMessageImageContent,
    VisionMessageImageURL,
    VisionMessageTextContent,
)

# Please use your own file path
PDF_FILE_PATH = "layout-parser-paper.pdf"


def convert_image_to_url(image: Image) -> str:
    """
    Converts a PIL Image to a base64-encoded URL.

    Args:
        image (Image): The image to convert.

    Returns:
        str: The base64-encoded URL of the image.
    """
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    decoded_image = base64.b64encode(buffered.getvalue()).decode("utf-8")
    url = f"data:image/jpeg;base64,{decoded_image}"
    return url


def convert_images_to_urls(images: list[Image]) -> list[str]:
    """
    Converts a list of PIL Images to a list of base64-encoded URLs.

    Args:
        images (List[Image]): The list of images to convert.

    Returns:
        List[str]: The list of base64-encoded URLs.
    """
    return [convert_image_to_url(image) for image in images]


def get_vision_prompt_template() -> Prompt:
    """
    Creates a vision prompt template.

    Returns:
        Prompt: The vision prompt template.
    """
    text_message = VisionMessageTextContent(text="{{extraction_instruction}}")
    image_message = VisionMessageImageContent(image_url=VisionMessageImageURL(url="{{img_url}}"))
    vision_message = VisionMessage(content=[text_message, image_message], role="user")
    vision_prompt = Prompt(messages=[vision_message])
    return vision_prompt


def extract_text_from_images(file_path: str, extraction_instruction: str) -> str:
    """
    Extracts text from images in a PDF file using a vision prompt.

    Args:
        file_path (str): The path to the PDF file.
        extraction_instruction (str): The instruction for text extraction.

    Returns:
        str: The extracted text in Markdown format.
    """
    # Convert PDF to images
    images = convert_from_path(file_path)
    urls = convert_images_to_urls(images)

    # Initialize vision prompt
    vision_prompt = get_vision_prompt_template()

    # Initialize LLM node
    llm_node = llms.OpenAI(
        name="OpenAI Vision",
        model="gpt-4o",
        prompt=vision_prompt,
        connection=OpenAIConnection(),
    )

    # Generate outputs for each image
    outputs = []
    for image_url in urls:
        output = llm_node.execute(
            input_data={
                "extraction_instruction": extraction_instruction,
                "img_url": image_url,
            }
        )
        outputs.append(output)

    # Combine outputs into a single document
    document_content = "".join(output["content"] for output in outputs)
    return document_content


def main():
    """
    Main function to extract text from a PDF file and print the result.
    """
    extraction_instruction = """
        Please extract the English text from the provided images and present it in Markdown format.
        Maintain the Markdown syntax for all formatting elements such as headings, images, links,
        bold text, tables, etc. Do not enclose the text with ```markdown...```. Do not extract the
        header and footer. Do not write your own text. Only extract the text from the images.
    """
    document_content = extract_text_from_images(PDF_FILE_PATH, extraction_instruction)
    print(document_content)


if __name__ == "__main__":
    main()
