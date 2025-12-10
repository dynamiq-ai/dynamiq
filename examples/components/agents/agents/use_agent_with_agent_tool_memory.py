from dynamiq import Workflow
from dynamiq.flows import Flow
from dynamiq.memory import Memory
from dynamiq.memory.backends.in_memory import InMemory
from dynamiq.nodes.agents import Agent
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


def make_researcher(llm: object) -> Agent:
    return Agent(
        name="Researcher",
        description='Call with {"input": "<research question>"}',
        role=RESEARCHER_ROLE,
        llm=llm,
        memory=Memory(backend=InMemory()),
        max_loops=3,
    )


def make_manager(llm: object, researcher: Agent) -> Agent:
    return Agent(
        name="Manager",
        description="Delegates fact gathering to the Researcher tool and summarizes results.",
        role=MANAGER_ROLE,
        llm=llm,
        tools=[researcher],
        memory=Memory(backend=InMemory()),
        propagate_user_context=True,  # defaults to True; shown here for clarity
        max_loops=3,
    )


def run_workflow():
    """
    Demonstrates parent+child agents where user/session context is auto-propagated,
    so both agents use their own memories consistently.
    """
    llm = setup_llm()
    researcher = make_researcher(llm)
    manager = make_manager(llm, researcher)

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
    logger.info("Researcher memory:\n%s", researcher.memory.get_all_messages_as_string())


if __name__ == "__main__":
    run_workflow()
