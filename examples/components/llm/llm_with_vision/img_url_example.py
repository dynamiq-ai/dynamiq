# flake8: noqa
import base64

from dynamiq.connections import AWS as AWSConnection
from dynamiq.connections import Gemini as GeminiConnection
from dynamiq.connections import OpenAI as OpenAIConnection
from dynamiq.nodes import llms
from dynamiq.prompts import (
    Prompt,
    VisionMessage,
    VisionMessageImageContent,
    VisionMessageImageURL,
    VisionMessageTextContent,
)


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def get_prompt():
    text_message = VisionMessageTextContent(text="{{user_message}}")
    image_message = VisionMessageImageContent(image_url=VisionMessageImageURL(url="{{img_url}}"))

    vision_message = VisionMessage(content=[text_message, image_message], role="user")

    prompt = Prompt(id="1", messages=[vision_message])
    return prompt


def run_openai_img_url(prompt):
    openai_node = llms.OpenAI(
        name="OpenAI Vision Answer Generation",
        model="gpt-4o-mini",
        prompt=prompt,
        connection=OpenAIConnection(),
    )

    openai_output = openai_node.execute(
        input_data={
            "user_message": "What’s in this image?",
            "img_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg",
        }
    )
    print(openai_output)


# Gemini
def run_gemini(prompt):

    gemini_node = llms.Gemini(
        name="Gemini Vision Answer Generation",
        model="gemini/gemini-1.5-flash",
        prompt=prompt,
        connection=GeminiConnection(),
    )

    gemini_output = gemini_node.execute(
        input_data={
            "user_message": "What’s in this image?",
            "img_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg",
        }
    )

    print(gemini_output)


# Bedrock Claude Sonnet
def run_anthropic_bedrock(prompt):

    anthropic_bedrock_node = llms.Bedrock(
        name="Bedrock Vision Answer Generation",
        model="bedrock/anthropic.claude-3-sonnet-20240229-v1:0",
        prompt=prompt,
        connection=AWSConnection(),
    )

    anthropic_output = anthropic_bedrock_node.execute(
        input_data={
            "user_message": "What’s in this image?",
            "img_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg",
        }
    )

    print(anthropic_output)


# Local image
def run_openai_img_local(prompt, image_path):
    base64_image = encode_image(image_path)

    openai_node = llms.OpenAI(
        name="OpenAI Vision Answer Generation",
        model="gpt-4o-mini",
        prompt=prompt,
        connection=OpenAIConnection(),
    )

    openai_output = openai_node.execute(
        input_data={
            "user_message": "What’s in this image?",
            "img_url": f"data:image/jpeg;base64,{base64_image}",
        }
    )
    print(openai_output)


def main():
    prompt = get_prompt()
    run_openai_img_url(prompt)
    run_anthropic_bedrock(prompt)
    run_gemini(prompt)

    # Please use your own pptx file path
    image_file_path = "../../data/img.jpeg"
    run_openai_img_local(prompt, image_file_path)


if __name__ == "__main__":
    main()
