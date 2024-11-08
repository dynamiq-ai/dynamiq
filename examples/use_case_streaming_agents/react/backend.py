from dynamiq.callbacks.streaming import StreamingIteratorCallbackHandler
from dynamiq.connections import E2B, ScaleSerp
from dynamiq.memory import Memory
from dynamiq.memory.backend.in_memory import InMemory
from dynamiq.nodes.agents.react import ReActAgent
from dynamiq.nodes.tools.e2b_sandbox import E2BInterpreterTool
from dynamiq.nodes.tools.scale_serp import ScaleSerpTool
from dynamiq.runnables import RunnableConfig
from dynamiq.types.streaming import StreamingConfig, StreamingMode
from examples.llm_setup import setup_llm


def setup_agent(agent_role: str, streaming_enabled: bool) -> ReActAgent:
    """
    Initializes an AI agent with a specified role and streaming configuration.
    """
    llm = setup_llm()
    memory = Memory(backend=InMemory())
    streaming_config = StreamingConfig(enabled=streaming_enabled)
    tool_search = ScaleSerpTool(connection=ScaleSerp())
    tool_code = E2BInterpreterTool(connection=E2B())
    agent = ReActAgent(
        name="Agent",
        llm=llm,
        role=agent_role,
        id="agent",
        memory=memory,
        tools=[tool_code, tool_search],
        streaming=streaming_config,
        streaming_mode=StreamingMode.ALL,
    )
    return agent


def generate_agent_response(agent: ReActAgent, user_input: str):
    """
    Processes the user input using the agent. Supports both streaming and non-streaming responses.
    """
    if agent.streaming.enabled:
        streaming_handler = StreamingIteratorCallbackHandler()
        agent.run(
            input_data={"input": user_input}, config=RunnableConfig(callbacks=[streaming_handler], streaming=True)
        )

        response_text = ""

        for chunk in streaming_handler:
            if isinstance(chunk.data, dict):
                if "choices" in chunk.data:
                    delta = chunk.data.get("choices", [{}])[0].get("delta", {})
                    content = delta.get("content", "")

                    if content:
                        response_text += content
                        yield content

                elif "content" in chunk.data:
                    content = chunk.data.get("content", {})

                    if isinstance(content, dict):
                        content_type = content.get("type", "")

                        if content_type == "tool_execution":
                            tool_execution_data = content.get("content", {}).get("output", "")
                            print("Tool Execution Data:", tool_execution_data)
                            yield "\n\nObservation:\n\n"
                            yield tool_execution_data
                            yield "\n\n"

            elif isinstance(chunk.data, str):
                response_text += chunk.data
                yield chunk.data

            else:
                print(f"Unexpected chunk data type: {type(chunk.data)}")

    else:
        result = agent.run({"input": user_input})
        response_text = result.output.get("content", "")
        yield response_text
