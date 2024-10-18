import json

from pydantic import BaseModel, ValidationError

from dynamiq.nodes.llms import OpenAI
from dynamiq.nodes.types import InferenceMode
from dynamiq.prompts import Prompt


class Document(BaseModel):
    title: str
    abstract: str
    tags: list[str]


def validate_json_response(response: str) -> Document:
    try:
        parsed_event = Document.model_validate_json(response)
        print("Pydantic Model Response:", parsed_event)
        return parsed_event
    except ValidationError as e:
        print(f"Validation Error: {e}")
        print("Raw Response:", response)
        return None


def run_openai_node(prompt: Prompt, schema: dict, inference_mode: InferenceMode):
    openai_node = OpenAI(model="gpt-4o-mini")
    response = openai_node.run(input_data={}, schema=schema, prompt=prompt, inference_mode=inference_mode)
    return response


# Example 1: Using OpenAI with structured output (JSON mode)
prompt = Prompt(
    messages=[
        {"role": "system", "content": "Extract the document information."},
        {
            "role": "user",
            "content": (
                "I like reading the book 'Harry Potter 7' which contains text"
                "about a young magical boy and magic. It can be described as fiction,"
                "story, children's literature."
            ),
        },
    ]
)

# Example 2: JSON mode with explicit JSON structure
prompt_json = Prompt(
    messages=[
        {
            "role": "system",
            "content": "Extract the document information in JSON format with fields: title, abstract, tags.",
        },
        {
            "role": "user",
            "content": (
                "I like reading the book 'Harry Potter 7' which contains text"
                "about a young magical boy and magic. It can be described as fiction,"
                "story, children's literature."
            ),
        },
    ]
)

response = run_openai_node(prompt=prompt, schema=Document, inference_mode=InferenceMode.STRUCTURED_OUTPUT)

if response and "content" in response.output:
    document = validate_json_response(response.output["content"])

response_json = run_openai_node(
    prompt=prompt_json, schema={"type": "json_object"}, inference_mode=InferenceMode.STRUCTURED_OUTPUT
)

if response_json and "content" in response_json.output:
    try:
        json_data = json.loads(response_json.output["content"])
        print("JSON Response:", json_data)
    except json.JSONDecodeError as e:
        print(f"JSON Decoding Error: {e}")
        print("Raw JSON Response:", response_json.output.get("content"))
