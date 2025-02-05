import json

from dynamiq.connections import OpenAI as OpenAIConnection
from dynamiq.nodes.agents.simple import SimpleAgent
from dynamiq.nodes.llms import OpenAI


def execute_agent(system_prompt: str, user_prompt: str, to_json: bool = False) -> dict:
    """Executes an LLM request"""
    llm = OpenAI(
        model="gpt-4o-mini",
        connection=OpenAIConnection(),
        max_tokens=3000,
    )

    agent = SimpleAgent(
        name="Agent",
        llm=llm,
        role=system_prompt,
        id="agent",
    )

    try:
        response = agent.run(input_data={"input": user_prompt})
        response = response.output.get("content", "").strip()

        if to_json:
            try:
                return json.loads(response.lstrip("'`json").rstrip("'`"))
            except json.JSONDecodeError:
                return {}

        return response
    except AttributeError:
        return {} if to_json else ""
