import json

from dynamiq.connections import OpenAI as OpenAIConnection
from dynamiq.nodes.llms import OpenAI
from dynamiq.prompts import Message, Prompt


def execute_llm(system_prompt: str, user_prompt: str, to_json: bool = False) -> dict:
    """Executes an LLM request"""
    llm = OpenAI(
        model="gpt-4o-mini",
        connection=OpenAIConnection(),
        max_tokens=3000,
        prompt=Prompt(
            messages=[
                Message(role="system", content=system_prompt),
                Message(role="user", content=user_prompt),
            ]
        ),
    )

    try:
        response = llm.run(input_data={}).output.get("content", "").strip()

        if to_json:
            try:
                return json.loads(response.lstrip("'`json").rstrip("'`"))
            except json.JSONDecodeError:
                return {}

        return response
    except AttributeError:
        return {} if to_json else ""
