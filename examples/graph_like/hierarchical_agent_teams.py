import functools
import json
from typing import Any

from dynamiq.connections import Tavily, ZenRows
from dynamiq.nodes.agents.orchestrators.graph import END, GraphOrchestrator
from dynamiq.nodes.agents.orchestrators.graph_manager import GraphAgentManager
from dynamiq.nodes.agents.react import ReActAgent
from dynamiq.nodes.tools import TavilyTool, ZenRowsTool
from dynamiq.nodes.tools.function_tool import function_tool
from dynamiq.nodes.types import InferenceMode
from dynamiq.prompts import Message, Prompt
from dynamiq.utils.logger import logger
from examples.llm_setup import setup_llm

DOCUMENT_FINAL = "document_final.txt"
DOCUMENT_NOTE = "results_note.txt"


def run_workflow():
    llm = setup_llm()

    def read_document(file_name: str) -> str:
        """Read the specified document."""
        with open(file_name) as file:
            lines = file.readlines()
        return "\n".join(lines)

    def write_document(file_name: str, content: str, mode: str = "w"):
        """Create and save a text document."""
        with open(file_name, mode) as file:
            file.write(content)
        return "Document saved updated."

    def supervisor(context: dict[str, Any], prompt_system, states):

        function_def = {
            "type": "json_schema",
            "json_schema": {
                "strict": True,
                "name": "plan_next_state",
                "schema": {
                    "type": "object",
                    "required": ["thought", "next_state"],
                    "properties": {
                        "thought": {
                            "type": "string",
                            "description": "Your reasoning about the next state.",
                        },
                        "next_state": {"type": "string", "enum": states, "description": "Name of next state."},
                    },
                    "additionalProperties": False,
                },
            },
        }

        llm_result = llm.run(
            input_data={},
            prompt=Prompt(
                messages=[
                    Message(role="system", content=prompt_system),
                    Message(
                        role="user",
                        content=(
                            f"History: {context['history']}"
                            "Given the conversation above, how to proceed or END?"
                            f"Select one of: {states}"
                        ),
                    ),
                ]
            ),
            schema=function_def,
            inference_mode=InferenceMode.STRUCTURED_OUTPUT,
        )

        return json.loads(llm_result.output["content"])["next_state"]

    def create_research_team() -> GraphOrchestrator:
        tavily_connection = Tavily()
        tavily_tool = TavilyTool(connection=tavily_connection)

        zenrows_connection = ZenRows()
        zenrows = ZenRowsTool(connection=zenrows_connection)

        search_assistant = ReActAgent(
            name="Search assistant", llm=llm, role=("You are a helpful search assistant"), tools=[tavily_tool]
        )
        web_scrapper_assistant = ReActAgent(
            name="Web Scrapper Assistant", llm=llm, role=("You are a helpfull research assistant"), tools=[zenrows]
        )

        research_team_supervisor = functools.partial(
            supervisor,
            prompt_system="You are a supervisor tasked with managing a conversation between the"
            " following workers: search, web_scrapper. Given the following user request,"
            " respond with the worker to act next. Each worker will perform a"
            " task and respond with their results and status. When finished,"
            " respond with END.",
            states=["web_scrapper", "search", END],
        )

        research_orchestrator = GraphOrchestrator(
            manager=GraphAgentManager(llm=llm), initial_state="research_team_supervisor", final_summarizer=True
        )

        research_orchestrator.add_node("search", [search_assistant])
        research_orchestrator.add_node("web_scrapper", [web_scrapper_assistant])
        research_orchestrator.add_node("research_team_supervisor", [])
        research_orchestrator.add_conditional_edge(
            "research_team_supervisor", ["search", "web_scrapper", END], research_team_supervisor
        )
        research_orchestrator.add_edge("search", "research_team_supervisor")
        research_orchestrator.add_edge("web_scrapper", "research_team_supervisor")

        return research_orchestrator

    def create_document_writing_team():
        @function_tool
        def write_document_notes(content: str):
            """
            Use notes to store valuable information.
            Provide notes in markdown format.
            """
            return write_document(DOCUMENT_NOTE, content, mode="a")

        @function_tool
        def write_document_final(content: str):
            """
            Writes final report to file.
            Provide content in markdown format.
            """
            return write_document(DOCUMENT_FINAL, content)

        @function_tool
        def read_document_notes():
            return read_document(DOCUMENT_NOTE)

        @function_tool
        def read_document_final():
            return read_document(DOCUMENT_NOTE)

        doc_writing_agent = ReActAgent(
            name="Document Writing Assistant",
            llm=llm,
            role=(
                "You are a helpfull document writing assistant to store final document."
                "Before using this agent read what is present in notes already."
            ),
            tools=[read_document_final(), write_document_final()],
        )

        notes_agent = ReActAgent(
            name="Notes Writing Assistant",
            llm=llm,
            role=(
                "You are a helpfull notes writing assistant for storing notes in the file system."
                "Call it when new portion of information comes."
            ),
            tools=[read_document_notes(), write_document_notes()],
        )

        doc_writing_team_supervisor = functools.partial(
            supervisor,
            prompt_system="You are a supervisor tasked with managing a conversation between the"
            " following workers: note_taker, doc_writer. Given the following user request,"
            " respond with the worker to act next. Each worker will perform a"
            " task and respond with their results and status. When finished,"
            " respond with END.",
            states=["note_taker", "doc_writer", END],
        )

        document_writing_orchestrator = GraphOrchestrator(
            manager=GraphAgentManager(llm=llm), initial_state="doc_writing_team_supervisor", final_summarizer=True
        )

        document_writing_orchestrator.add_node("doc_writer", [doc_writing_agent])
        document_writing_orchestrator.add_node("note_taker", [notes_agent])
        document_writing_orchestrator.add_node("doc_writing_team_supervisor", [])

        document_writing_orchestrator.add_edge("doc_writer", "doc_writing_team_supervisor")
        document_writing_orchestrator.add_edge("note_taker", "doc_writing_team_supervisor")

        document_writing_orchestrator.add_conditional_edge(
            "doc_writing_team_supervisor", ["note_taker", "doc_writer", END], doc_writing_team_supervisor
        )

        return document_writing_orchestrator

    # Create Research Team
    research_chain = create_research_team()

    # Document Writing Team
    document_writing_chain = create_document_writing_team()

    def research_team_node(context: dict[str, Any], **kwargs):
        result = research_chain.run(input_data={"input": context["history"][-1]}).output["content"]
        return result

    def document_writing_team_node(context: dict[str, Any], **kwargs):
        notes_content = read_document(DOCUMENT_NOTE)
        gathered_information = str(context["history"][-1]) + "\n" + "\n".join(notes_content)
        prompt = (
            "Previous report was refused by supervisor. \n"
            f"<Previous report>\n {read_document(DOCUMENT_FINAL)}\n</Previous report>"
            "Write a final report to a file based on gathered informatiom and notes:"
            f"<Information>{gathered_information}</Information>"
        )
        result = document_writing_chain.run(input_data={"input": prompt}).output["content"]
        return result

    graph_orchestrator = GraphOrchestrator(
        manager=GraphAgentManager(llm=llm), initial_state="supervisor", final_summarizer=True
    )

    teams_supervisor = functools.partial(
        supervisor,
        prompt_system="You are a supervisor tasked with managing a conversation between the"
        " following teams: research_team, document_writing_team. Given the following user request,"
        " respond with the worker to act next. Each worker will perform a"
        " task and respond with their results and status. If you are satisfied with final report"
        " respond with END. When ending make sure that report is not empty."
        " Final report (At the beginning it can be empty):"
        f"'{read_document(DOCUMENT_FINAL)}'",
        states=["research_team", "document_writing_team", END],
    )

    graph_orchestrator.add_node("supervisor", [])
    graph_orchestrator.add_node("research_team", [research_team_node])
    graph_orchestrator.add_node("document_writing_team", [document_writing_team_node])

    graph_orchestrator.add_conditional_edge("supervisor", ["research_team", "document_writing_team"], teams_supervisor)
    graph_orchestrator.add_edge("research_team", "supervisor")
    graph_orchestrator.add_edge("document_writing_team", "supervisor")

    return graph_orchestrator.run(
        input_data={"input": "Gather information about Shapley Values types. Keep it short."}
    ).output["content"]


if __name__ == "__main__":
    result = run_workflow()
    logger.info(result)
