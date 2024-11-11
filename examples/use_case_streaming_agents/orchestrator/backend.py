from dynamiq.callbacks.streaming import StreamingIteratorCallbackHandler
from dynamiq.connections import E2B
from dynamiq.memory import Memory
from dynamiq.memory.backend.in_memory import InMemory
from dynamiq.nodes.agents.orchestrators.adaptive import AdaptiveOrchestrator
from dynamiq.nodes.agents.orchestrators.adaptive_manager import AdaptiveAgentManager
from dynamiq.nodes.agents.react import ReActAgent
from dynamiq.nodes.agents.simple import SimpleAgent
from dynamiq.nodes.tools.e2b_sandbox import E2BInterpreterTool
from dynamiq.runnables import RunnableConfig
from dynamiq.types.streaming import StreamingConfig, StreamingMode
from examples.llm_setup import setup_llm


def setup_agent(agent_role: str, streaming_enabled: bool, streaming_mode: str) -> ReActAgent:
    """
    Initializes an AI agent with a specified role and streaming configuration.
    """
    llm = setup_llm()
    memory = Memory(backend=InMemory())
    mode_mapping = {"Final": StreamingMode.FINAL, "All": StreamingMode.ALL}
    mode = mode_mapping.get(streaming_mode, StreamingMode.FINAL)
    streaming_config = StreamingConfig(enabled=streaming_enabled, mode=mode)
    tool_code = E2BInterpreterTool(connection=E2B())

    agent_coding = ReActAgent(
        name="Coding Agent",
        id="agent",
        llm=llm,
        tools=[tool_code],
        role=agent_role,
        memory=memory,
    )

    agent_writer = SimpleAgent(
        name="Writer Agent",
        id="writer",
        llm=llm,
        role="An agent that can write reports and summaries.",
        memory=memory,
    )

    agent_manager = AdaptiveAgentManager(
        name="Adaptive Agent Manager",
        llm=llm,
    )

    orchestrator = AdaptiveOrchestrator(
        name="Adaptive Orchestrator",
        agents=[agent_coding, agent_writer],
        manager=agent_manager,
        streaming=streaming_config,
    )
    return orchestrator


def generate_agent_response(orchestrator: AdaptiveOrchestrator, user_input: str):
    """
    Processes the user input using the agent. Supports both streaming and non-streaming responses.
    """
    if orchestrator.streaming.enabled:
        streaming_handler = StreamingIteratorCallbackHandler()
        orchestrator.run(
            input_data={"input": user_input}, config=RunnableConfig(callbacks=[streaming_handler], streaming=True)
        )

        response_text = ""

        for chunk in streaming_handler:
            content = chunk.data
            if content:
                response_text += " " + content.get("content", "")
                yield content

    else:
        result = orchestrator.run({"input": user_input})
        response_text = result.output.get("content", "")
        yield response_text
