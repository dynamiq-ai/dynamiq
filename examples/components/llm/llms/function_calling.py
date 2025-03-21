import json
import logging

from dynamiq import Workflow, runnables
from dynamiq.connections.managers import ConnectionManager
from dynamiq.flows import Flow
from dynamiq.nodes.llms import OpenAI
from dynamiq.prompts import Message, Prompt
from dynamiq.prompts.prompts import Tool, ToolFunction, ToolFunctionParameters


def get_current_time(location: str) -> str:
    """Get the current time in a given location."""
    location_times = {
        "tokyo": {"location": "Tokyo", "current_time": "14:00"},
        "san francisco": {"location": "San Francisco", "current_time": "22:00"},
        "paris": {"location": "Paris", "current_time": "06:00"},
    }

    normalized_location = location.lower()

    if normalized_location in location_times:
        response_data = location_times[normalized_location]
    else:
        response_data = {"location": location, "current_time": "unknown"}

    return json.dumps(response_data)


def main():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    cm = ConnectionManager()

    tools = [
        Tool(
            function=ToolFunction(
                name="get_current_time",
                description="Get the current time in a given location",
                parameters=ToolFunctionParameters(
                    type="object",
                    required=["location"],
                    properties={
                        "location": {
                            "type": "string",
                            "description": "The city, e.g. San Francisco",
                        }
                    },
                ),
            ),
        )
    ]
    openai_node = OpenAI(
        name="openAI",
        model="gpt-3.5-turbo-1106",
        prompt=Prompt(
            messages=[
                Message(
                    role="user",
                    content="What time is it in San Francisco, Tokyo, and Paris?",
                ),
            ],
            tools=tools,
        ),
        temperature=0.1,
    )

    wf = Workflow(
        id="wf",
        flow=Flow(
            id="wf",
            nodes=[openai_node],
            connection_manager=cm,
        ),
    )
    response = wf.run(
        input_data={},
        config=runnables.RunnableConfig(callbacks=[]),
    )
    tool_calls = response.output[openai_node.id]["output"].get("tool_calls", {})
    if tool_calls:
        available_functions = {
            "get_current_time": get_current_time,
        }

        for tool_call in list(tool_calls.values()):
            logger.info(f"\nExecuting tool call\n{tool_call}")
            function_name = tool_call["function"]["name"]
            function_to_call = available_functions[function_name]
            function_args = tool_call["function"]["arguments"]
            function_response = function_to_call(
                location=function_args.get("location"),
            )
            logger.info(f"Result from tool call\n{function_response}\n")
    logger.info(f"Workflow result:{response}")


if __name__ == "__main__":
    main()
