import os

from dotenv import load_dotenv

from dynamiq.connections import OpenAI as OpenAIConnection
from dynamiq.nodes.agents.react import ReActAgent
from dynamiq.nodes.llms.openai import OpenAI
from dynamiq.nodes.tools.python import Python

load_dotenv()

OPENAI_KEY = os.getenv("OPENAI_KEY")
FLOWS_ENGINE_API_KEY = os.getenv("FLOWS_ENGINE_API_KEY")
URL = os.getenv("URL")


headers = {
    "Authorization": f"apikey {FLOWS_ENGINE_API_KEY}",
}

wikipedia_tool_code = f"""
import requests

def run(input_data):
    query = input_data.get('query')
    r = requests.post("{URL}", json = {{"query": "{{ \\n search(q: \\"{{query}}\\")\\n }}"}}, headers={headers})
    return r.json()
"""

weather_tool_code = f"""
import requests

def run(input_data):
    city = input_data.get('city')
    r = requests.post("{URL}", json = {{"query": "{{ \\n weatherByCity(q: \\"Los Angeles\\") \\n \
        \\n {{ \\n main \\n {{  \\n humidity \\n pressure \\n feels_like \\n temp \\n }} \\n}} }}"}},\
        headers={headers})
    return r.json()
# """

if __name__ == "__main__":
    llm = OpenAI(
        name="OpenAI LLM",
        connection=OpenAIConnection(api_key=OPENAI_KEY),
        model="gpt-4o-mini",
        temperature=0.1,
    )

    wikipedia_tool = Python(
        name="Wikipedia Tool",
        description="Allow to access information from Wikipedia. Provide query with 'query' parameter.",
        code=wikipedia_tool_code,
    )

    weather_tool = Python(
        name="Weather Tool",
        description="Allow to access weather information in specific city. Provide name of city with 'city' parameter.",
        code=weather_tool_code,
    )

    agent = ReActAgent(
        name="AI Agent",
        llm=llm,
        tools=[wikipedia_tool, weather_tool],
        role="You are helpful assistant that answers on question",  # noqa: E501
    )

    result = agent.run(input_data={"input": "What is weather like in Kyiv."}, config=None)

    print("Agent's response:")
    print(result.output["content"])
