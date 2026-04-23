"""End-to-end tests for Agent.response_format across inference modes.

Verifies that when an Agent is configured with a user-provided schema
(pydantic BaseModel or JSON schema dict), the final content it returns
is the parsed structured value - not a string.
"""

import pytest
from pydantic import BaseModel

from dynamiq import connections
from dynamiq.nodes.agents import Agent
from dynamiq.nodes.llms.openai import OpenAI
from dynamiq.nodes.types import InferenceMode


class Document(BaseModel):
    title: str
    abstract: str
    tags: list[str]


DICT_SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "abstract": {"type": "string"},
        "tags": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["title", "abstract", "tags"],
    "additionalProperties": False,
}


INPUT_MESSAGE = (
    "Summarize this book into a document: 'Harry Potter 7' is a fantasy novel "
    "about a young magical boy facing dark magic. It is fiction, a story, and "
    "children's literature."
)


INFERENCE_MODES = [
    InferenceMode.XML,
    InferenceMode.STRUCTURED_OUTPUT,
    InferenceMode.FUNCTION_CALLING,
]


def _make_agent(inference_mode: InferenceMode, response_format) -> Agent:
    llm = OpenAI(model="gpt-5.4-mini", connection=connections.OpenAI(), temperature=0.1)
    return Agent(
        name="StructuredAgent",
        llm=llm,
        tools=[],
        inference_mode=inference_mode,
        response_format=response_format,
        max_loops=5,
    )


@pytest.mark.flaky(reruns=2)
@pytest.mark.parametrize("inference_mode", INFERENCE_MODES)
def test_agent_response_format_dict_schema(inference_mode):
    """A JSON-schema dict response_format returns a parsed dict in every mode."""
    agent = _make_agent(inference_mode, response_format=DICT_SCHEMA)
    result = agent.run(input_data={"input": INPUT_MESSAGE})

    assert result.status.value == "success"
    content = result.output["content"]
    assert isinstance(content, dict)
    assert {"title", "abstract", "tags"} <= content.keys()
    assert isinstance(content["title"], str) and content["title"]
    assert isinstance(content["abstract"], str) and content["abstract"]
    assert isinstance(content["tags"], list)
    assert all(isinstance(t, str) for t in content["tags"])


@pytest.mark.flaky(reruns=2)
@pytest.mark.parametrize("inference_mode", INFERENCE_MODES)
def test_agent_response_format_pydantic_model(inference_mode):
    """A pydantic BaseModel is accepted as ``response_format`` input.

    The agent normalizes the class to its JSON schema dict at validation time
    and returns the final answer as a parsed dict conforming to that schema.
    Callers who want a typed instance pass the dict through their BaseModel.
    """
    agent = _make_agent(inference_mode, response_format=Document)
    result = agent.run(input_data={"input": INPUT_MESSAGE})

    assert result.status.value == "success"
    content = result.output["content"]
    assert isinstance(content, dict)
    assert {"title", "abstract", "tags"} <= content.keys()

    # Caller-side validation recovers the typed instance.
    doc = Document.model_validate(content)
    assert doc.title
    assert doc.abstract
    assert isinstance(doc.tags, list)
    assert all(isinstance(t, str) for t in doc.tags)
