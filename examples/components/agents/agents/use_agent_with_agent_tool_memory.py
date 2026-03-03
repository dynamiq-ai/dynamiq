from dynamiq import Workflow
from dynamiq.flows import Flow
from dynamiq.memory import Memory
from dynamiq.memory.backends.in_memory import InMemory
from dynamiq.nodes.agents import Agent, SubAgentTool
from dynamiq.utils.logger import logger
from examples.llm_setup import setup_llm

RESEARCHER_ROLE = """
You are a focused researcher who digs up concise facts.
Summarize clearly and keep track of prior answers in memory when user/session ids are provided.
"""

MANAGER_ROLE = """
You are a manager agent that delegates research tasks to your Researcher tool.
Always call the Researcher tool with {"input": "<subtask>"} and then summarize the findings for the user.
Use clear task descriptions when delegating and return concise, useful summaries.
"""


def make_researcher_tool(llm: object) -> tuple[SubAgentTool, Memory]:
    researcher_memory = Memory(backend=InMemory())
    tool = SubAgentTool(
        name="Researcher",
        description='Call with {"input": "<research question>"}',
        factory=lambda: Agent(
            name="Researcher",
            role=RESEARCHER_ROLE,
            llm=llm,
            memory=researcher_memory,
            max_loops=3,
        ),
    )
    return tool, researcher_memory


def make_manager(llm: object, researcher_tool: SubAgentTool) -> Agent:
    return Agent(
        name="Manager",
        description="Delegates fact gathering to the Researcher tool and summarizes results.",
        role=MANAGER_ROLE,
        llm=llm,
        tools=[researcher_tool],
        memory=Memory(backend=InMemory()),
        max_loops=3,
    )


def run_workflow():
    """
    Demonstrates parent+child agents where user/session context is auto-propagated,
    so both agents use their own memories consistently.
    """
    llm = setup_llm()
    researcher_tool, researcher_memory = make_researcher_tool(llm)
    manager = make_manager(llm, researcher_tool)

    wf = Workflow(flow=Flow(nodes=[manager]))

    shared_ids = {"user_id": "demo-user", "session_id": "demo-session"}

    first = wf.run(
        input_data={"input": "Create a short brief on Mars with two concise facts.", **shared_ids},
    )
    first_answer = first.output[manager.id]["output"]["content"]
    logger.info("First turn:\n%s", first_answer)

    second = wf.run(
        input_data={"input": "Remind me what you said and add one new fact.", **shared_ids},
    )
    second_answer = second.output[manager.id]["output"]["content"]
    logger.info("Second turn:\n%s", second_answer)

    logger.info("Manager memory:\n%s", manager.memory.get_all_messages_as_string())
    logger.info("Researcher memory:\n%s", researcher_memory.get_all_messages_as_string())


if __name__ == "__main__":
    run_workflow()
