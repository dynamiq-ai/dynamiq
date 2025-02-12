import json
import re

from dynamiq.connections import OpenAI as OpenAIConnection
from dynamiq.nodes.llms import OpenAI
from dynamiq.prompts import Message, Prompt


def _extract_code_block(text):
    match = re.search(r"```(.*?)```", text, re.DOTALL)
    return match.group(1).strip() if match else text


def execute_agent(system_prompt: str, user_prompt: str, to_json: bool = False) -> dict:
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

    response = llm.run(input_data={}).output.get("content", "").strip()

    if to_json:
        try:
            response = _extract_code_block(response)
            return json.loads(response.lstrip("'`json").rstrip("'`"))
        except json.JSONDecodeError as e:
            print(f"JSON decoding failed: {e}. Response: {response}")
            return {}

    return response
