from dynamiq.callbacks.streaming import StreamingIteratorCallbackHandler
from dynamiq.connections import E2B, ScaleSerp
from dynamiq.memory import Memory
from dynamiq.memory.backends.in_memory import InMemory
from dynamiq.nodes.agents.orchestrators.adaptive import AdaptiveOrchestrator
from dynamiq.nodes.agents.orchestrators.adaptive_manager import AdaptiveAgentManager
from dynamiq.nodes.agents.react import ReActAgent
from dynamiq.nodes.agents.simple import SimpleAgent
from dynamiq.nodes.tools.e2b_sandbox import E2BInterpreterTool
from dynamiq.nodes.tools.scale_serp import ScaleSerpTool
from dynamiq.runnables import RunnableConfig
from dynamiq.types.streaming import StreamingConfig, StreamingMode
from examples.llm_setup import setup_llm

# Agent Constants
AGENT_ROLE_CODING = (
    "An Expert Agent with advanced programming skills, capable of solving any problem using coding expertise. "
    "The goal is to provide the best solution for each request, utilizing all algorithmic knowledge and coding skills."
)

AGENT_ROLE_SEARCH = (
    "An Expert Agent with exceptional search skills, capable of finding any information efficiently and accurately."
)

AGENT_ROLE_WRITER = (
    "An Expert Agent with advanced writing skills, "
    "capable of crafting reports, summaries, and other text-based content. "
    "The goal is to provide the best solution for each request using markdown language. "
    "Maintains a friendly and professional tone with proper citations of sources."
)


def setup_agent(streaming_enabled: bool, streaming_mode: str) -> ReActAgent:
    """
    Initializes an AI agent with a specified role and streaming configuration.
    """
    llm = setup_llm()
    memory = Memory(backend=InMemory())
    mode_mapping = {"Answer": StreamingMode.FINAL, "Steps": StreamingMode.ALL}
    mode = mode_mapping.get(streaming_mode, StreamingMode.FINAL)
    streaming_config = StreamingConfig(enabled=streaming_enabled, mode=mode)
    tool_code = E2BInterpreterTool(connection=E2B())

    tool_search = ScaleSerpTool(connection=ScaleSerp())

    agent_coding = ReActAgent(
        name="Coding Agent",
        id="agent",
        llm=llm,
        tools=[tool_code],
        role=AGENT_ROLE_CODING,
        memory=memory,
    )

    agent_search = ReActAgent(
        name="Search Agent",
        id="agent",
        llm=llm,
        tools=[tool_search],
        role=AGENT_ROLE_SEARCH,
        memory=memory,
    )

    agent_writer = SimpleAgent(
        name="Writer Agent",
        id="writer",
        llm=llm,
        memory=memory,
        role=AGENT_ROLE_WRITER,
    )

    agent_manager = AdaptiveAgentManager(
        name="Adaptive Agent Manager",
        llm=llm,
    )

    orchestrator = AdaptiveOrchestrator(
        name="Adaptive Orchestrator",
        agents=[agent_coding, agent_writer, agent_search],
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
            input_data={"input": user_input, "user_id": "1", "session_id": "1"},
            config=RunnableConfig(callbacks=[streaming_handler]),
        )

        response_text = ""

        for chunk in streaming_handler:
            content = chunk.data
            if content:
                response_text += " " + content.get("choices", [{}])[0].get("delta", {}).get("content")
                yield content

    else:
        result = orchestrator.run({"input": user_input})
        response_text = result.output.get("content", "")
        yield response_text
